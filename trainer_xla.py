import torch
import torch.multiprocessing as mp
import random
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import numpy as np
import pickle
from torch.utils.data import Dataset
import os
import logging

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from adaIN.adaIN_xla import NSTTransform
import adaIN.net as net
from resnet_wide import WideResNet_28_4

import sys

os.environ.pop['TPU_PROCESS_ADDRESSES']

current_dir = os.path.dirname(__file__)
module_path = os.path.abspath(current_dir)

if module_path not in sys.path:
    sys.path.append(module_path)

encoder_rel_path = 'adaIN/models/vgg_normalised.pth'
decoder_rel_path = 'adaIN/models/decoder.pth'
encoder_path = os.path.join(current_dir, encoder_rel_path)
decoder_path = os.path.join(current_dir, decoder_rel_path)

class CIFAR10(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        
        self.data = []
        self.targets = []
        
        # Load data from the dataset files
        if self.train:
            for batch_id in range(1, 6):
                file_path = os.path.join(data_dir, f'data_batch_{batch_id}')
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f, encoding='bytes')
                    self.data.extend(entry[b'data'])
                    self.targets.extend(entry[b'labels'])
        else:
            file_path = os.path.join(data_dir, 'test_batch')
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='bytes')
                self.data = entry[b'data']
                self.targets = entry[b'labels']
        
        self.data = np.array(self.data, dtype=np.uint8)
        self.targets = np.array(self.targets)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.data[index].reshape(3, 32, 32)
        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        target = self.targets[index]
        
        return img, target

def load_models(device):
    vgg = net.vgg
    decoder = net.decoder
    vgg.load_state_dict(torch.load(encoder_path))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    decoder.load_state_dict(torch.load(decoder_path))

    vgg.to(device).eval()
    decoder.to(device).eval()
    return vgg, decoder

def train_loop_fn(net, loader, optimizer, criterion, device):
    net.train()
    for i, (inputs, labels) in enumerate(loader):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        xm.optimizer_step(optimizer)
        if i % 10 == 0:
            xm.mark_step()
            xm.master_print(f'Batch {i}, Loss: {loss.item():.4f}')

def test_loop_fn(net, loader, device):
    net.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    xm.master_print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def trainer_fn(net, trainloader, testloader, device, epochs=50, save_path='./cifar_net.pth'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        xm.master_print(f'Epoch {epoch + 1}/{epochs}')
        train_loop_fn(net, trainloader, optimizer, criterion, device)
        test_accuracy = test_loop_fn(net, testloader, device)
        scheduler.step()
        xm.master_print(f'Epoch {epoch + 1} Test Accuracy: {test_accuracy:.2f}%')

    xm.save(net.state_dict(), save_path)
    xm.master_print('Finished Training')

def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    device = xm.xla_device()
    
    vgg, decoder = load_models(device)
    nst_transfer = NSTTransform(style_dir='/kaggle/input/painter-by-numbers-resized', vgg=vgg, decoder=decoder)

    transform_train = transforms.Compose([nst_transfer])
    transform_test = transforms.Compose([transforms.ToTensor()])

    batch_size = 1024
    cifar_10_dir = '/kaggle/input/cifar10-python/cifar-10-batches-py/'
    
    trainset = CIFAR10(data_dir=cifar_10_dir, transform=transform_train)
    testset = CIFAR10(data_dir=cifar_10_dir, train=False, transform=transform_test)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        drop_last=True)
    
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True)

    trainloader = pl.MpDeviceLoader(trainloader, device)
    testloader = pl.MpDeviceLoader(testloader, device)

    net = WideResNet_28_4(num_classes=10).to(device)
    
    trainer_fn(net, trainloader, testloader, device, epochs=flags['num_epochs'])

if __name__ == '__main__':
    flags = {}
    flags['num_epochs'] = 50
    xmp.spawn(_mp_fn, args=(flags,), nprocs=2)



    

