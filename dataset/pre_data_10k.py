import numpy as np
import imageio
import os
from PIL import Image
from torchvision import transforms

class Dataset():
    def __init__(self, input_size, root, dataset_path, mode='train'):
        self.input_size = input_size
        self.root = root
        self.mode = mode

        train_img_path = val_img_path = test_img_path = dataset_path

        train_label_file = open(os.path.join(self.root, 'dataset_10k', 'train.txt'))
        val_label_file = open(os.path.join(self.root, 'dataset_10k', 'val.txt'))
        test_label_file = open(os.path.join(self.root, 'dataset_10k', 'test.txt'))

        train_img_label = []
        val_img_label = []
        test_img_label = []
        for line in train_label_file:
            parts = line.strip().split(' ')
            if len(parts) >= 2:
                train_img_label.append([os.path.join(train_img_path, parts[0]), int(parts[1])])
        for line in val_label_file:
            parts = line.strip().split(' ')
            if len(parts) >= 2:
                val_img_label.append([os.path.join(val_img_path, parts[0]), int(parts[1])])
        for line in test_label_file:
            parts = line.strip().split(' ')
            if len(parts) >= 2:
                test_img_label.append([os.path.join(test_img_path, parts[0]), int(parts[1])])

        self.train_img_label = train_img_label
        self.val_img_label = val_img_label
        self.test_img_label = test_img_label

    def __getitem__(self, index):
        if self.mode == 'train':
            img, target = imageio.imread(self.train_img_label[index][0]), self.train_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            if img.shape[2] == 4:
                img = img[:,:,:3]
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize(self.input_size + 16)(img)
            img = transforms.RandomRotation(20)(img)
            img = transforms.RandomVerticalFlip()(img)
            img = transforms.RandomCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        elif self.mode == 'val':
            img, target = imageio.imread(self.val_img_label[index][0]), self.val_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            if img.shape[2] == 4:
                img = img[:,:,:3]
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize(self.input_size + 16)(img)
            img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = imageio.imread(self.test_img_label[index][0]), self.test_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            if img.shape[2] == 4:
                img = img[:,:,:3]
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize(self.input_size + 16)(img)
            img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_img_label)
        elif self.mode == 'val':
            return len(self.val_img_label)
        else:
            return len(self.test_img_label)
