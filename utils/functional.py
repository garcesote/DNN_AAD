import torch
import numpy as np

def get_subject(idx, n_subjects):
    subjects = ['S' + str(i+1) for i in range(n_subjects)]
    return subjects[idx]

def get_trials(split):
    if split == 'train':
        return np.arange(0,40)
    elif split == 'val':
        return np.arange(40,50)
    elif split == 'test':
        return np.arange(50,60)
    else:
        raise ValueError('Field split must be a train/val/test value')

# turn a tensor to 0 mean and std of 1 with shape (C, T) and return shape (C)   
def normalize(tensor):

    # unsqueeze necesario para el broadcasting (10) => (10, 1)
    mean = (torch.mean(tensor, dim=1)).unsqueeze(1)
    std = torch.std(tensor, dim=1).unsqueeze(1)

    return (tensor - mean) / std

def correlation(x, y, eps=1e-8):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + eps)
    return corr