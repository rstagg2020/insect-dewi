#coding=utf-8
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import MultiStepLR
import shutil
import time
from utils.set_seeds import seed_everything
from utils.read_dataset_vt import read_dataset
from utils.train_model import train
from config import seed, batch_size, root, checkpoint_path, init_lr, lr_decay_rate,\
    lr_milestones, weight_decay, end_epoch, dataset_path, input_size
from utils.auto_load_resume import auto_load_resume
import os
import argparse
import wandb
from pytorch_metric_learning import losses, miners

from models.dewi import dewi_resnet50, dewi_resnet101, dewi_resnet152, dewi_resnext50_32x4d, dewi_resnext101_32x8d, dewi_resnext101_64x4d,\
    dewi_wide_resnet50_2, dewi_wide_resnet101_2

class CosineClassifier(nn.Module):
    def __init__(self, in_features, num_classes, init_scale=20.0):
        super(CosineClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        # Make the scale 's' a learnable parameter to dynamically soften/harden the distribution
        self.scale = nn.Parameter(torch.tensor([init_scale]))
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # L2-normalize both features and weights
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        # Cosine similarity scaled by a constant factor
        return F.linear(x, w) * self.scale


device = torch.device("cuda")

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="chosen model")
args = vars(ap.parse_args())

model_pool = {
    'dewi_resnet50': dewi_resnet50,
    'dewi_resnet101': dewi_resnet101,
    'dewi_resnet152': dewi_resnet152,
    'dewi_resnext50_32x4d': dewi_resnext50_32x4d,
    'dewi_resnext101_32x8d': dewi_resnext101_32x8d,
    'dewi_resnext101_64x4d': dewi_resnext101_64x4d,
    'dewi_wide_resnet50_2': dewi_wide_resnet50_2,
    'dewi_wide_resnet101_2': dewi_wide_resnet101_2,
}

pretrained_url_pool = dict.fromkeys(['dewi_resnet50'], "https://download.pytorch.org/models/resnet50-11ad3fa6.pth")
pretrained_url_pool.update(dict.fromkeys(['dewi_resnet101'], "https://download.pytorch.org/models/resnet101-cd907fc2.pth"))
pretrained_url_pool.update(dict.fromkeys(['dewi_resnet152'], "https://download.pytorch.org/models/resnet152-f82ba261.pth"))
pretrained_url_pool.update(dict.fromkeys(['dewi_resnext50_32x4d'], "https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth"))
pretrained_url_pool.update(dict.fromkeys(['dewi_resnext101_32x8d'], "https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth"))
pretrained_url_pool.update(dict.fromkeys(['dewi_resnext101_64x4d'], "https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth"))
pretrained_url_pool.update(dict.fromkeys(['dewi_wide_resnet50_2'], "https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth"))
pretrained_url_pool.update(dict.fromkeys(['dewi_wide_resnet101_2'], "https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth"))


def main():
    # count num classes
    classes_file = open(os.path.join(root, 'dataset_vt', 'classes.txt'))
    num_classes = len(classes_file.readlines())
    classes_file.close()
    
    # set all the necessary seeds
    seed_everything(seed)
    
    dataset_path_vt = os.path.join(root, "vt_data", "10KDataVT2014-2022")
    end_epoch = 100 # Increased to train longer after plateau
    # Read the dataset
    trainloader, valloader, testloader = read_dataset(input_size, batch_size, root, dataset_path_vt)

    # Initialize the model (it defaults to 102 classes in dewi.py)
    model = model_pool.get(args["model"])(pth_url=pretrained_url_pool.get(args["model"]), pretrained=True)
    
    # Modify the classification head for the new number of classes
    in_features = model.fc.in_features
    # Replace the FC layer with a new one matching out classes (Logit Normalization)
    model.fc = CosineClassifier(in_features, num_classes, init_scale=20.0)
    
    # load pretrained IP102 checkpoint if it exists so we are fine-tuning from IP102.
    pretrained_ip102_path = os.path.join(checkpoint_path, args["model"], "best_model.pth")
    if os.path.exists(pretrained_ip102_path):
        print(f"Loading IP102 pre-trained weights from {pretrained_ip102_path}")
        checkpoint = torch.load(pretrained_ip102_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        # remove the old fc weights
        state_dict = {k: v for k, v in state_dict.items() if 'fc.' not in k}
        model.load_state_dict(state_dict, strict=False)

    # define the CE loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    metric_loss = losses.TripletMarginLoss(0.2)
    miner = miners.BatchHardMiner()
    
    # Differential learning rates:
    # Separate the FC layer parameters from the rest of the model (the backbone)
    fc_params = list(map(id, model.fc.parameters()))
    base_params = filter(lambda p: id(p) not in fc_params, model.parameters())

    # define the optimizer with differential learning rates
    optimizer = torch.optim.SGD([
        {'params': base_params, 'lr': init_lr * 0.1},  # Backbone gets a 10x smaller LR
        {'params': model.fc.parameters(), 'lr': init_lr} # New classification head gets standard LR
    ], momentum=0.9, weight_decay=weight_decay)
    # define the learning rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay_rate, verbose=True)

    # loading checkpoint
    save_path = os.path.join(checkpoint_path, args["model"] + "_vt")
    if os.path.exists(save_path):
        start_epoch, best_val_acc = auto_load_resume(model, optimizer, scheduler, save_path, status='train', device=device)
        
        # User requested to change learning rate due to plateau
        new_lr = 0.00003
        print(f"Lowering learning rate to {new_lr} to handle plateau...")
        
        # Fix: Maintain the 10x differential learning rate ratio between backbone and head
        optimizer.param_groups[0]['lr'] = new_lr * 0.1  # Backbone
        optimizer.param_groups[1]['lr'] = new_lr        # Classification Head
        
        # Reset optimizer state (momentum buffers) to prevent immediate divergence
        optimizer.state.clear()

        assert start_epoch < end_epoch
    else:
        os.makedirs(save_path)
        best_val_acc = 0.0
        start_epoch = 0

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    

    time_str = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy('./config.py', os.path.join(save_path, "{}config.py".format(time_str)))
    
    # Initialize wandb
    wandb.init(project="dewi-insect-classification", name="vt-100k-finetuning", config={
        "model": args["model"],
        "batch_size": batch_size,
        "init_lr": init_lr,
        "end_epoch": end_epoch
    })

     # Train the model
    train(model=model,
          device=device,
          trainloader=trainloader,
          valloader=valloader,
          testloader=testloader,
          metric_loss=metric_loss,
          miner=miner,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          save_path=save_path,
          start_epoch=start_epoch,
          end_epoch=end_epoch,
          best_val_acc = best_val_acc)


if __name__ == '__main__':
    main()
