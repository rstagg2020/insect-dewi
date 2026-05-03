#coding=utf-8
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import shutil
import time
from utils.set_seeds import seed_everything
from utils.focal_read_dataset_vt import read_dataset
from utils.focal_train_model import train
import os
import argparse
import wandb
from pytorch_metric_learning import losses, miners

from models.dewi import dewi_resnet50, dewi_resnet101, dewi_resnet152, dewi_resnext50_32x4d, dewi_resnext101_32x8d, dewi_resnext101_64x4d,\
    dewi_wide_resnet50_2, dewi_wide_resnet101_2

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

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
    from linear_config import seed, batch_size, root, checkpoint_path, init_lr, backbone_lr_factor, lr_decay_rate,\
        lr_milestones, weight_decay, end_epoch, dataset_path, input_size
    from utils.auto_load_resume import auto_load_resume

    # count num classes
    classes_file = open(os.path.join(root, 'dataset_vt', 'classes.txt'))
    num_classes = len(classes_file.readlines())
    classes_file.close()
    
    seed_everything(seed)

    dataset_path_vt = os.path.join(root, "vt_data", "100KDataVT2014-2022")
    end_epoch = int(os.environ.get('LINEAR_END_EPOCH', end_epoch))
    _batch_size = int(os.environ.get('LINEAR_BATCH_SIZE', batch_size))
    _num_workers = int(os.environ.get('LINEAR_NUM_WORKERS', 6))
    trainloader, valloader, testloader = read_dataset(input_size, _batch_size, root, dataset_path_vt, num_workers=_num_workers)

    # Initialize the model (pretrained ImageNet backbone)
    model = model_pool.get(args["model"])(pth_url=pretrained_url_pool.get(args["model"]), pretrained=True)
    
    # Standard linear classifier head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
    nn.init.constant_(model.fc.bias, 0)

    # Focal loss
    criterion = FocalLoss(alpha=1, gamma=2)
    metric_loss = losses.TripletMarginLoss(0.2)
    miner = miners.BatchHardMiner()
    
    # Differential learning rates
    fc_params = list(map(id, model.fc.parameters()))
    base_params = filter(lambda p: id(p) not in fc_params, model.parameters())

    optimizer = torch.optim.SGD([
        {'params': base_params, 'lr': init_lr * backbone_lr_factor},
        {'params': model.fc.parameters(), 'lr': init_lr}
    ], momentum=0.9, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # loading checkpoint
    save_path = os.path.join(checkpoint_path, args["model"] + "_vt")
    if os.path.exists(save_path) and os.path.exists(os.path.join(save_path, 'current_model.pth')):
        start_epoch, best_val_acc = auto_load_resume(model, optimizer, scheduler, save_path, status='train', device=device)
        assert start_epoch < end_epoch
    else:
        os.makedirs(save_path, exist_ok=True)
        best_val_acc = 0.0
        start_epoch = 0

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    time_str = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy('./linear_config.py', os.path.join(save_path, "{}config.py".format(time_str)))
    
    wandb.init(project="dewi-insect-classification", name="linear-vt-100k", config={
        "model": args["model"],
        "batch_size": _batch_size,
        "init_lr": init_lr,
        "end_epoch": end_epoch,
        "classifier": "nn.Linear"
    })

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
          best_val_acc=best_val_acc)


if __name__ == '__main__':
    main()
