#coding=utf-8
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
import shutil
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
from utils.set_seeds import seed_everything
from utils.focal_read_dataset_vt import read_dataset
from utils.foc_tran_train_model import train
from foc_tran_config import seed, batch_size, root, checkpoint_path, init_lr, weight_decay, end_epoch, dataset_path, input_size
from utils.auto_load_resume import auto_load_resume
import os
import argparse
import wandb
from pytorch_metric_learning import losses, miners

from models.transform_dewi import dewi_resnet50, dewi_resnet101, dewi_resnet152, dewi_resnext50_32x4d, dewi_resnext101_32x8d, dewi_resnext101_64x4d,\
    dewi_wide_resnet50_2, dewi_wide_resnet101_2

class CosineClassifier(nn.Module):
    def __init__(self, in_features, num_classes, init_scale=20.0):
        super(CosineClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = nn.Parameter(torch.tensor([init_scale]))
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        logits = F.linear(x, w) * self.scale
        return logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
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
    classes_file = open(os.path.join(root, 'dataset_vt', 'classes.txt'))
    num_classes = len(classes_file.readlines())
    classes_file.close()
    
    seed_everything(seed)
    
    dataset_path_vt = os.path.join(root, "vt_data", "100KDataVT2014-2022")
    end_epoch = int(os.environ.get('FOC_TRAN_END_EPOCH', 200))
    _batch_size = int(os.environ.get('FOC_TRAN_BATCH_SIZE', batch_size))
    _num_workers = int(os.environ.get('FOC_TRAN_NUM_WORKERS', 6))
    trainloader, valloader, testloader = read_dataset(input_size, _batch_size, root, dataset_path_vt, num_workers=_num_workers)

    model = model_pool.get(args["model"])(pth_url=pretrained_url_pool.get(args["model"]), pretrained=True)
    
    in_features = model.fc.in_features
    model.fc = CosineClassifier(in_features, num_classes, init_scale=20.0)
    
    criterion = FocalLoss(alpha=1, gamma=2)
    metric_loss = losses.TripletMarginLoss(0.2)
    miner = miners.BatchHardMiner()
    
    fc_params = list(map(id, model.fc.parameters()))
    neck_params = list(map(id, model.neck.parameters()))
    base_params = list(filter(lambda p: id(p) not in fc_params and id(p) not in neck_params, model.parameters()))

    # AdamW with differential LRs:
    #   backbone  — low LR (pretrained, don't perturb)
    #   neck      — higher LR (randomly initialized transformer, needs to learn fast)
    #   head      — higher LR (classification head, also randomly initialized)
    optimizer = torch.optim.AdamW([
        {'params': base_params,                'lr': 1e-5,  'weight_decay': weight_decay},
        {'params': model.neck.parameters(),    'lr': 3e-4,  'weight_decay': 1e-2},
        {'params': model.fc.parameters(),      'lr': 3e-4,  'weight_decay': weight_decay},
    ])

    # CosineAnnealingLR — smooth, no sudden drops; T_max = full training run
    scheduler = CosineAnnealingLR(optimizer, T_max=end_epoch, eta_min=1e-7)

    save_path = os.path.join(checkpoint_path, args["model"] + "_vt")
    if os.path.exists(save_path) and os.path.exists(os.path.join(save_path, 'current_model.pth')):
        start_epoch, best_val_acc = auto_load_resume(model, optimizer, scheduler, save_path, status='train', device=device)
    else:
        os.makedirs(save_path, exist_ok=True)
        # We are starting a new foc_tran run. Load from the plateaued transformer run to continue from epoch 63
        pretrained_path = os.path.join(root, "checkpoint", args["model"] + "_vt", "current_model.pth")
        if os.path.exists(pretrained_path):
            print(f"Resuming from previous transformer checkpoint: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint['model_state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
            start_epoch = checkpoint.get('epoch', 0)
            best_val_acc = checkpoint.get('val_acc', 0.0)
            print(f"Resumed from epoch {start_epoch} with val_acc {best_val_acc}")
        else:
            start_epoch = 0
            best_val_acc = 0.0

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    time_str = time.strftime("%Y%m%d-%H%M%S")
    config_src = os.path.join(os.path.dirname(__file__), 'foc_tran_config.py')
    shutil.copy(config_src, os.path.join(save_path, "{}config.py".format(time_str)))
    
    wandb.init(project="dewi-insect-classification", name="foc-tran-vt-100k", config={
        "model": args["model"],
        "batch_size": batch_size,
        "init_lr": init_lr,
        "end_epoch": end_epoch,
        "approach": "focal+transform"
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
