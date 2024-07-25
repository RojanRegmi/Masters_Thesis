import torch
import torch.multiprocessing as mp
import random
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.optim as optim
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.init as init
from PIL import Image

import numpy as np

import pickle
from torch.utils.data import Dataset
import os
import logging

from adaIN.adaIN_v2 import NSTTransform
import adaIN.net as net
from resnet_wide import WideResNet_28_4

import sys

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

def load_models():
    vgg = net.vgg
    decoder = net.decoder
    vgg.load_state_dict(torch.load(encoder_path))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    decoder.load_state_dict(torch.load(decoder_path))

    vgg.to(device).eval()
    decoder.to(device).eval()
    return vgg, decoder


def trainer_fn(epochs: int, net, trainloader, testloader, device, save_path='./cifar_net.pth'):

    log_file = 'training.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()])
    logger = logging.getLogger()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # Initialize the scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)  # Cosine Annealing LR Scheduler

    for epoch in range(epochs):

        running_loss = 0.0
        total_train = 0
        correct_train = 0
        total = 0
        correct = 0

        net.train()

        for i, (inputs, labels) in enumerate(trainloader):

            logging.info(f'Start Training batch {i}')
            # zero the parameter gradients
            optimizer.zero_grad()

            # get the inputs
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # calculating training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if i % 10 == 0:
                logger.info(f'Number of Minibatches:{i}, total train: {total_train}, running_loss: {running_loss}')

        with torch.no_grad():
            net.eval()
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
        
        print(f'Epoch {epoch + 1} Train Accuracy: {100 * correct_train / total_train}, Test Accuracy: {100 * correct / total}')
        logger.info(f"Epoch {epoch + 1} Train Accuracy: {100 * correct_train / total_train}, Test Accuracy: {100 * correct / total}")

        scheduler.step()
    
    torch.save(net.state_dict(), save_path)
    print('Finished Training')

if __name__ == '__main__' :

    mp.set_start_method('spawn', force=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    vgg, decoder = load_models()

    nst_transfer = NSTTransform(style_dir = '/kaggle/input/painter-by-numbers-resized', vgg=vgg, decoder=decoder)

    transform_train = transforms.Compose([
        nst_transfer,
        #transforms.RandomHorizontalFlip(),
        
        #transforms.RandomCrop(32, padding=4),  
        #TrivialAugmentWide(),
        #transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 512

    cifar_10_dir = '/kaggle/input/cifar10-python/cifar-10-batches-py/'
    trainset = CIFAR10(data_dir=cifar_10_dir, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, pin_memory=True, num_workers=4)

    testset = CIFAR10(data_dir=cifar_10_dir, train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, pin_memory=True, num_workers=4)
    
    net = WideResNet_28_4(num_classes=10)
    net.to(device)

    trainer_fn(epochs=50, net=net, trainloader=trainloader, testloader=testloader, device=device, save_path='./cifar_net.pth')


    

            




    

