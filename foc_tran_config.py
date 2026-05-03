seed = 42
batch_size = 32
end_epoch = 150
init_lr = 0.003        # kept for reference; AdamW LRs set per-group in train script
weight_decay = 5e-5    # used for backbone + head; neck uses 1e-2 (set in train script)
input_size = 384

root = '/netfiles/dmlabshare1/rstagg/insect-dewi'
checkpoint_path = '/netfiles/dmlabshare1/rstagg/insect-dewi/foc_tran_checkpoint/'
dataset_path = '/netfiles/dmlabshare1/rstagg/insect-dewi/vt_data/100KDataVT2014-2022'
