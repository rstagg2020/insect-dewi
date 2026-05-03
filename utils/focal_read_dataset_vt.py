import torch
from dataset_vt import pre_data_vt as pre_data

def read_dataset(input_size, batch_size, root, dataset_path, num_workers=6):
    persistent = num_workers > 0
    # 'forkserver' is safe with CUDA + DataParallel, unlike default 'fork'
    # which deadlocks when worker processes inherit CUDA/threading state
    mp_context = 'forkserver' if num_workers > 0 else None

    trainset = pre_data.Dataset(input_size=input_size, root=root, dataset_path=dataset_path, mode='train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers, drop_last=True,
                                                pin_memory=False, persistent_workers=persistent,
                                                multiprocessing_context=mp_context)

    valset = pre_data.Dataset(input_size=input_size, root=root, dataset_path=dataset_path, mode='val')
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers, drop_last=False,
                                                  pin_memory=False, persistent_workers=persistent,
                                                  multiprocessing_context=mp_context)

    testset = pre_data.Dataset(input_size=input_size, root=root, dataset_path=dataset_path, mode='test')
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers, drop_last=False,
                                                 pin_memory=False, persistent_workers=persistent,
                                                 multiprocessing_context=mp_context)

    return trainloader, valloader, testloader
