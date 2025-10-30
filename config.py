# train a miniature character-level model
# good for debugging and playing on macbooks and such

out_dir = '' # need fix
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 1
log_interval = 10 # don't print too too often


# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'char'
wandb_run_name = 'mini-gpt'

dataset = 'char'

gradient_accumulation_steps = 1 # if ddp, gradient_accumulation_steps > 1
batch_size = 2048 #128 for edge list, 1024 for adj list 1024 512


block_size = 2000 # context of up to 256 previous characters

# baby GPT model :) 
n_layer = 4# 12 #  24 # 6
n_head = 12 # 12 # 16 # 6
n_embd = 192 # 768 # 1024 # 384
dropout =  0.2

learning_rate = 5e-4 # 5e-4 for adj, 1e-3 for edge  with baby networks can afford to go a bit higher
max_iters =  40000
lr_decay_iters = 40000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 1 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
