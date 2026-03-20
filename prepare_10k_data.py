import os
import random

root_dir = '/mnt/data_sda/rstagg/git_repos/insects/DeWi/10k_data/10KDataVT2014-2022'
out_dir = '/mnt/data_sda/rstagg/git_repos/insects/DeWi/dataset_10k'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

classes = sorted(os.listdir(root_dir))
classes = [c for c in classes if os.path.isdir(os.path.join(root_dir, c))]

with open(os.path.join(out_dir, 'classes.txt'), 'w') as f:
    for i, c in enumerate(classes):
        f.write(f"{i} {c}\n")

train_lines = []
val_lines = []
test_lines = []

for i, c in enumerate(classes):
    img_dir = os.path.join(root_dir, c)
    imgs = sorted(os.listdir(img_dir))
    imgs = [img for img in imgs if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    random.seed(42)
    random.shuffle(imgs)
    
    n_total = len(imgs)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    if n_train == 0 and n_total > 0:
        n_train = n_total
        n_val = 0
        n_test = 0
        
    train_imgs = imgs[:n_train]
    val_imgs = imgs[n_train:n_train+n_val]
    test_imgs = imgs[n_train+n_val:]
    
    for img in train_imgs:
        train_lines.append(f"{c}/{img} {i}\n")
    for img in val_imgs:
        val_lines.append(f"{c}/{img} {i}\n")
    for img in test_imgs:
        test_lines.append(f"{c}/{img} {i}\n")

random.shuffle(train_lines)
random.shuffle(val_lines)
random.shuffle(test_lines)

with open(os.path.join(out_dir, 'train.txt'), 'w') as f:
    f.writelines(train_lines)
with open(os.path.join(out_dir, 'val.txt'), 'w') as f:
    f.writelines(val_lines)
with open(os.path.join(out_dir, 'test.txt'), 'w') as f:
    f.writelines(test_lines)

print(f"Created dataset splits in {out_dir}")
print(f"Train: {len(train_lines)}, Val: {len(val_lines)}, Test: {len(test_lines)}")
