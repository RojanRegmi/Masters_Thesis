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

import pickle
from torch.utils.data import Dataset
import os

from adaIN.adaIN import NSTTransform
from resnet_wide import WideResNet_28_4

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


def trainer_fn(epochs: int, net, trainloader, testloader, device, save_path='./cifar_net.pth'):

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

        with torch.no_grad():
            net.eval()
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total *= labels.size(0)
                correct += (predicted==labels).sum().item()
        
        print(f'Epoch {epoch + 1} Train Accuracy: {100 * correct_train / total_train}, Test Accuracy: {100 * correct / total}')

        scheduler.step()
    
    torch.save(net.state_dict(), save_path)
    print('Finished Training')

if __name__ == '__main__' :

    mp.set_start_method('spawn', force=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    nst_transfer = NSTTransform(style_dir = '/kaggle/input/painter-by-numbers-resized')

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

    batch_size = 256

    cifar_10_dir = '/kaggle/input/cifar10-python/cifar-10-batches-py/'
    trainset = CIFAR10(data_dir=cifar_10_dir, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = CIFAR10(data_dir=cifar_10_dir, train=False, transform=transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)
    
    net = WideResNet_28_4(num_classes=10)
    net.to(device)

    trainer_fn(epochs=100, net=net, trainloader=trainloader, testloader=testloader, device=device, save_path='./cifar_net.pth')


    

            




    

