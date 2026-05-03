seed = 42
batch_size = 32
end_epoch = 250
init_lr = 0.003
backbone_lr_factor = 0.1
# Tighter early milestones; softer decay so LR isn't killed in one step.
# Epoch 40 milestone just fired when we stopped — give it room to breathe.
lr_milestones = [40, 60, 80, 110, 130, 145]
lr_decay_rate = 0.2
weight_decay = 5e-5
input_size = 384

root = '/netfiles/dmlabshare1/rstagg/insect-dewi'
checkpoint_path = '/netfiles/dmlabshare1/rstagg/insect-dewi/linear_checkpoint/'
dataset_path = '/netfiles/dmlabshare1/rstagg/insect-dewi/vt_data/100KDataVT2014-2022'
