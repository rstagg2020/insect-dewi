import torch
from dataset_vt import pre_data_vt as pre_data

def read_dataset(input_size, batch_size, root, dataset_path):
    trainset = pre_data.Dataset(input_size=input_size, root=root, dataset_path=dataset_path, mode='train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=6, drop_last=True, pin_memory=False,
                                                persistent_workers=True)

    valset = pre_data.Dataset(input_size=input_size, root=root, dataset_path=dataset_path, mode='val')
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                  shuffle=False, num_workers=6, drop_last=False, pin_memory=False,
                                                  persistent_workers=True)

    testset = pre_data.Dataset(input_size=input_size, root=root, dataset_path=dataset_path, mode='test')
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=6, drop_last=False, pin_memory=False,
                                                 persistent_workers=True)

    return trainloader, valloader, testloader

