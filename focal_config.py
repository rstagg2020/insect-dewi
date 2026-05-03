seed = 42
batch_size = 32
end_epoch = 250
init_lr = 0.003
resume_lr = 0.0003       # was 0.00003 — backbone was getting 3e-6 (10x too frozen)
backbone_lr_factor = 0.1
neck_lr_factor = 0.1
# Shift milestones forward — early epochs were effectively frozen at wrong LR
lr_milestones = [50, 70, 90, 115, 135, 148]
lr_decay_rate = 0.2
weight_decay = 5e-5
input_size = 384

root = '/netfiles/dmlabshare1/rstagg/insect-dewi'
checkpoint_path = '/netfiles/dmlabshare1/rstagg/insect-dewi/focal_checkpoint/'
dataset_path = '/netfiles/dmlabshare1/rstagg/insect-dewi/vt_data/100KDataVT2014-2022'