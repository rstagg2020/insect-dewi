seed = 42
batch_size = 32
end_epoch = 100
init_lr = 0.0003
resume_lr = 0.00003
backbone_lr_factor = 0.1
neck_lr_factor = 0.1
lr_milestones = [15, 30, 45, 60]
lr_decay_rate = 0.1
weight_decay = 1e-4
input_size = 384

root = '/netfiles/dmlabshare1/rstagg/insect-dewi'
checkpoint_path = '/netfiles/dmlabshare1/rstagg/insect-dewi/standard/checkpoint/'
dataset_path = '/netfiles/dmlabshare1/rstagg/insect-dewi/vt_data/100KDataVT2014-2022'