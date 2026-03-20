seed = 42
batch_size = 32
end_epoch = 100
init_lr = 0.003
lr_milestones = [15, 30, 45, 60]
lr_decay_rate = 0.1
weight_decay = 1e-4
input_size = 384
    
root = '/mnt/data_sda/rstagg/git_repos/insects/DeWi'   # the root folder, which contains all the files
checkpoint_path = '/mnt/data_sda/rstagg/git_repos/insects/DeWi/checkpoint/'      # path of the checkpoint
dataset_path = '/mnt/data_sda/rstagg/git_repos/insects/DeWi/images'    # the path of **images** folder