"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""
import json
import os
import time
import math
import pickle
from contextlib import nullcontext
import random
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import GPTConfig, GPT
from tqdm import tqdm

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda:1' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------




# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
print(int(os.environ.get('RANK', -1)))
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    print('gradient_acc',gradient_accumulation_steps,'ddp_world_size', ddp_world_size)
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
    print('gradient_acc',gradient_accumulation_steps)
else:
    
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
print('master process',master_process)
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if 'cuda' in device_type:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y




define_method_list=['term']
define_method=''
for t in define_method_list:
    define_method+=t
sub_scale=150000
PAD=' <PAD>'
triangle_num=[1]
triangle_num_string=''
for t in triangle_num:
    triangle_num_string+=str(t)




pattern_list=['FFL']
pattern_name_list=''
for p in pattern_list:
    pattern_name_list+=p
method='adj'
k_hop='5'
model_scale={384:'small',768:'mid',1024:'large',192:'tiny',512:'baby',96:'tiny2', 1536:'huge'}
base_path=f'/egr/research-dselab/shared/daixinna/nano-pattern/tiny_models'


iter_num = 0
best_val_loss = 1e9
min_train_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, f'{base_path}/my_meta_{define_method}_{pattern_name_list}_{triangle_num_string}_{method}.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']# +1
    question_max_length=meta['max_questions']# +1
    ans_max_length=meta['max_ans']# +1
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path}) ans_max_length={ans_max_length} question_max_length={question_max_length}")
else:
    print(meta_path)
    print(os.listdir(base_path))
print('ans_max_length',ans_max_length)
stoi, itos = meta['stoi'], meta['itos']

if sub_scale==100000:
    if 'cot' not in define_method:
        batch_size = batch_size*3
if sub_scale==30000:
    max_iters =  5000
    lr_decay_iters = 5000
    n_embd = 192
if sub_scale==150000 or sub_scale==250000 or sub_scale==50000 or sub_scale==350000 or sub_scale==650000:
    if 'cot' not in define_method:
        max_iters =  15000
        lr_decay_iters = 15000
        n_embd = 192
    if sub_scale==350000:
        max_iters =  45000
        lr_decay_iters = 45000
    if method == 'edge':
        max_iters *= 2
        lr_decay_iters *= 2

out_dir=base_path+f'/model_{pattern_name_list}_{triangle_num_string}_{method}_{define_method}_{model_scale[n_embd]}_{n_layer}_{n_head}_{sub_scale}_{seed_offset}'
if os.path.exists(out_dir)==False:
    os.makedirs(out_dir)
ckpt_path = os.path.join(out_dir, f'model.pt')

    

def get_my_batch(split,loaded_dicts):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    # print(device)
    question_data=loaded_dicts['question']
    ans_data=loaded_dicts['ans']
    
    ix = torch.randint(len(ans_data), (batch_size,))
    
    x = torch.from_numpy(question_data[ix,:])
    y = torch.from_numpy(ans_data[ix,:])
    # print('x',x.shape)
    question_end=torch.nonzero(x == stoi['<END_Q>'], as_tuple=False)

    x_list=[]
    y_list=[]
    if split=='train':
        for idx in range(batch_size):
            selected_batch=0 
            end_p=0
            if 'cot' in define_method:
                if define_method == 'cot_hex' or define_method == 'cot_pet':
                    selected_batch = random.randint(1, 2)
                elif define_method == 'cot_hex2':
                    selected_batch = random.randint(1, 4)
                else:
                    selected_batch = random.randint(1, 3)
            if 'cot' in define_method:
                if define_method == 'cot_hex' or define_method == 'cot_pet':
                    end_p=1
                elif define_method == 'cot_hex2':
                    end_p=3
                else:
                    end_p=2
                if selected_batch < end_p:
                    for num_i in range(1,end_p+1):
                        # num_i=selected_batch
                        ans_begin=torch.nonzero(y[idx,:] == stoi[f'<P{num_i}>'], as_tuple=False)
                        ans_end=torch.nonzero(y[idx,:] == stoi[f'<P{num_i}_END>'], as_tuple=False)
                        selected_vis=torch.randint(ans_begin,ans_end+2,(1,))
                        ans=y[idx,:selected_vis]
                        question_nodes=x[idx,-4:]
                        if idx >= question_end.shape[0]:
                            selected_idx=question_end.shape[0] - 1
                        else:selected_idx=idx
                        question=x[selected_idx,:question_end[selected_idx,1]]
                        provided_ans=y[selected_idx,:selected_vis-1]
                        pads_tensor_x=torch.ones(question_max_length-question.shape[0]-question_nodes.shape[0])*stoi['<PAD>']
                        provide_ans_pads=torch.ones(ans_max_length-provided_ans.shape[0])*stoi['<PAD>']
                        
                        x_list.append(torch.cat((question,pads_tensor_x.to(torch.int64),question_nodes.to(torch.int64),provided_ans,provide_ans_pads.to(torch.int64)),0).unsqueeze(0))

                        pads_graph=torch.ones(question_max_length)*stoi['<PAD>']
                        pads_tensor_y=torch.ones(ans_max_length-ans.shape[0])*stoi['<PAD>']
                        y_list.append(torch.cat((pads_graph.to(torch.int64),ans,pads_tensor_y.to(torch.int64)),0).unsqueeze(0))
            else:
                ans_begin=1
            if 'cot' not in define_method or selected_batch>=end_p:
                if 'cot' in define_method:
                    ans_begin=torch.nonzero(y[idx,:] == stoi[f'<ANS>'], as_tuple=False)
                ans_end=torch.nonzero(y[idx,:] == stoi[f'<END>'], as_tuple=False)
                if ans_end.numel() == 0:
                    ans_end = torch.tensor([[ans_max_length-1]],dtype=torch.int64)
                selected_vis=torch.randint(ans_begin,ans_end+2,(1,))
                ans=y[idx,:selected_vis]
                question_nodes=x[idx,-4:]
                if idx >= question_end.shape[0]:
                    selected_idx=question_end.shape[0] - 1
                else:selected_idx=idx
                question=x[selected_idx,:question_end[selected_idx,1]]
                provided_ans=y[selected_idx,:selected_vis-1]
                pads_tensor_x=torch.ones(question_max_length-question.shape[0]-question_nodes.shape[0])*stoi['<PAD>']
                provide_ans_pads=torch.ones(ans_max_length-provided_ans.shape[0])*stoi['<PAD>']
                x_list.append(torch.cat((question,pads_tensor_x.to(torch.int64),question_nodes.to(torch.int64),provided_ans,provide_ans_pads.to(torch.int64)),0).unsqueeze(0))
                pads_graph=torch.ones(question_max_length)*stoi['<PAD>']
                pads_tensor_y=torch.ones(ans_max_length-ans.shape[0])*stoi['<PAD>']
                y_list.append(torch.cat((pads_graph.to(torch.int64),ans,pads_tensor_y.to(torch.int64)),0).unsqueeze(0))
    else:
        for idx in range(batch_size):
            if 'cot' in define_method:
                ans_begin=torch.nonzero(y[idx,:] == stoi[f'<ANS>'], as_tuple=False)
                ans_end=torch.nonzero(y[idx,:] == stoi[f'<END>'], as_tuple=False)
                if ans_end.numel() == 0:
                    ans_end = torch.tensor([[ans_max_length-1]],dtype=torch.int64)
                selected_vis=torch.randint(ans_begin,ans_end+2,(1,))
                ans=y[idx,:selected_vis]
                question_nodes=x[idx,-4:]
                if idx >= question_end.shape[0]:
                    selected_idx=question_end.shape[0] - 1
                else:selected_idx=idx
                question=x[selected_idx,:question_end[selected_idx,1]]
                provided_ans=y[selected_idx,:selected_vis-1]
                pads_tensor_x=torch.ones(question_max_length-question.shape[0]-question_nodes.shape[0])*stoi['<PAD>']
                provide_ans_pads=torch.ones(ans_max_length-provided_ans.shape[0])*stoi['<PAD>']
                x_list.append(torch.cat((question,pads_tensor_x.to(torch.int64),question_nodes.to(torch.int64),provided_ans,provide_ans_pads.to(torch.int64)),0).unsqueeze(0))
                pads_graph=torch.ones(question_max_length)*stoi['<PAD>']
                pads_tensor_y=torch.ones(ans_max_length-ans.shape[0])*stoi['<PAD>']
                y_list.append(torch.cat((pads_graph.to(torch.int64),ans,pads_tensor_y.to(torch.int64)),0).unsqueeze(0))
            else:
                ans_end = torch.nonzero(y[idx, :] == stoi['<END>'], as_tuple=False)# [0, 0]
                if ans_end.numel() == 0:
                    ans_end = torch.tensor([[ans_max_length-1]],dtype=torch.int64)
                if len(pattern_list)>1:
                    ans_end = ans_end[0, 0]
                    candidates = torch.arange(1, ans_end + 2)
                    weights = torch.linspace(1.0, ans_max_length/2, steps=candidates.shape[0])
                    selected_vis_samples = candidates[torch.multinomial(weights, 3)]
                else:
                    selected_vis_samples = torch.randint(1, ans_end + 2, (2,))
                # [0,1]
                for selected_vis in selected_vis_samples:
                    ans=y[idx,:selected_vis]
                    question_nodes=x[idx,-4:]
                    if idx >= question_end.shape[0]:
                        selected_idx=question_end.shape[0] - 1
                    else:selected_idx=idx
                    question=x[selected_idx,:question_end[selected_idx,1]]
                    provided_ans=y[selected_idx,:selected_vis-1]
                    pads_tensor_x=torch.ones(question_max_length-question.shape[0]-question_nodes.shape[0])*stoi['<PAD>']
                    provide_ans_pads=torch.ones(ans_max_length-provided_ans.shape[0])*stoi['<PAD>']
                    
                    x_list.append(torch.cat((question,pads_tensor_x.to(torch.int64),question_nodes.to(torch.int64),provided_ans,provide_ans_pads.to(torch.int64)),0).unsqueeze(0))

                    pads_graph=torch.ones(question_max_length)*stoi['<PAD>']
                    pads_tensor_y=torch.ones(ans_max_length-ans.shape[0])*stoi['<PAD>']
                    y_list.append(torch.cat((pads_graph.to(torch.int64),ans,pads_tensor_y.to(torch.int64)),0).unsqueeze(0))

    x=torch.cat(x_list,0)
    y=torch.cat(y_list,0)
    y_mask=stoi['<PAD>']

    
    
    if 'cuda' in device_type :
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y,y_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), y_mask# .pin_memory().to(device, non_blocking=True)
    else:
        x, y,y_mask = x.to(device), y.to(device),y_mask# .to(device)
    return x, y,y_mask

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)


encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ' '.join([itos[i] for i in l])

# block_size=question_max_length+ans_max_length

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line


if init_from == 'scratch':
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
for name, param in model.named_parameters():
    print(f"{name}: {tuple(param.shape)}")
# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    print('have set ddp')
    print(ddp_local_rank)
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['val']:
        losses = torch.zeros(eval_iters)
        for k in tqdm(range(eval_iters)):
            X, Y,y_mask = get_my_batch(split,loaded_val_data) # fetch the very first batch
            with ctx:
                logits, loss = model(X, Y,y_mask)
                
            losses[k] = loss.item()
        
        out[split] = losses.mean()

    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    
question_data=np.memmap(os.path.join(data_dir, f'{base_path}/train_question_{define_method}_{pattern_name_list}_{triangle_num_string}_{method}_{sub_scale}.bin'), dtype=np.uint16, mode='r')
ans_data=np.memmap(os.path.join(data_dir, f'{base_path}/train_ans_{define_method}_{pattern_name_list}_{triangle_num_string}_{method}_{sub_scale}.bin'), dtype=np.uint16, mode='r')
ans_data=ans_data.reshape(-1,ans_max_length).astype(np.int64)
question_data=question_data.reshape(-1,question_max_length).astype(np.int64)


loaded_train_data={}
loaded_train_data['question']=question_data
loaded_train_data['ans']=ans_data
loaded_val_data={}
val_question_data=np.memmap(os.path.join(data_dir, f'{base_path}/val_question_{define_method}_{pattern_name_list}_{triangle_num_string}_{method}.bin'), dtype=np.uint16, mode='r')
val_ans_data=np.memmap(os.path.join(data_dir, f'{base_path}/val_ans_{define_method}_{pattern_name_list}_{triangle_num_string}_{method}.bin'), dtype=np.uint16, mode='r')
val_ans_data=val_ans_data.reshape(-1,ans_max_length).astype(np.int64)
val_question_data=val_question_data.reshape(-1,question_max_length).astype(np.int64)
loaded_val_data['question']=val_question_data
loaded_val_data['ans']=val_ans_data
# training loop
from torch.nn import functional as F
X, Y,y_mask = get_my_batch('train',loaded_train_data) # fetch the very first batch
print('x',X)
print('y',Y) 
print(y_mask)

loss_dicts={}
loss_dicts['train']={}
loss_dicts['val']={}
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:
    # print(iter_num)
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    ckpt_path_epoch = os.path.join(out_dir, f'{iter_num}_model.pt')
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        loss_dicts['val'][iter_num]=float(losses['val'])
        print(f"step {iter_num}, val loss {losses['val']}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                # "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            # if iter_num > 0:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, ckpt_path)
        if iter_num%10000 == 0 and iter_num > 0:
            print(f"saving checkpoint to {ckpt_path_epoch}")
            torch.save(checkpoint, ckpt_path_epoch)
            
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps-1 )
        with ctx:
            logits, loss = model(X, Y,y_mask)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y,y_mask = get_my_batch('train',loaded_train_data)


        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        loss_dicts['train'][iter_num]=float(lossf)
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        # print('\n-------')
        
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        if lossf < min_train_loss:
            min_train_loss = lossf

        with open(os.path.join(out_dir,'val_loss.json'),'w') as f:
                json.dump(loss_dicts,f)
        # print('-------')
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
    # exit()

print('mini train loss', min_train_loss)

if ddp:
    destroy_process_group()

print(out_dir)