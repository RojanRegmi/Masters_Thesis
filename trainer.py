import torch
import torchvision
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.optim as optim

import torch.nn as nn
from PIL import Image

import numpy as np

import pickle
from torch.utils.data import Dataset
import os
from gen_data_utils import load_augmented_traindata

import time
import logging
import argparse

from adaIN.adaIN_v3 import NSTTransform
import adaIN.net as net
from resnet_wide import WideResNet_28_4
from utils import RandomChoiceTransforms
from adaIN.mobilnet import EncoderNet, mobnet_decoder
from mobilenet.function import remove_batchnorm

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

def load_models(device, model_type):

    if model_type == 'vgg':
        encoder = net.vgg
        decoder = net.decoder
        encoder.load_state_dict(torch.load(encoder_path))
        encoder = nn.Sequential(*list(encoder.children())[:31])
        decoder.load_state_dict(torch.load(decoder_path))

        encoder.to(device).eval()
        decoder.to(device).eval()

    elif model_type == 'mobilenet':
        mobilenet_encoder_path = '/kaggle/working/Masters_Thesis/mobilenet/models/mobilenet_v1_encoder_weights.pth'
        mobilenet_decoder_path = '/kaggle/working/Masters_Thesis/mobilenet/models/decoder_mobilenet_classic.pth.tar'
        encoder = EncoderNet()
        encoder.load_state_dict(torch.load(mobilenet_encoder_path))
        encoder = remove_batchnorm(encoder)
        encoder = nn.Sequential(*list(encoder.children())[:5])

        decoder = mobnet_decoder
        decoder.load_state_dict(torch.load(mobilenet_decoder_path))

        encoder.to(device).eval()
        decoder.to(device).eval()

    
    return encoder, decoder

def load_feat_files(feats_dir, device):

    style_feats_np = np.load(feats_dir)
    style_feats_tensor = torch.from_numpy(style_feats_np)
    style_feats_tensor = style_feats_tensor.to(device)
    return style_feats_tensor


def trainer_fn(epochs: int, net, trainloader, testloader, device, save_path='./cifar_net.pth'):

    log_file = 'training.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()])
    logger = logging.getLogger()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # Initialize the scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)  # Cosine Annealing LR Scheduler

   
    for epoch in range(epochs):

        epoch_start_time = time.time()
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
        
        train_accuracy = 100 * correct_train/total_train
        train_loss = running_loss/len(trainloader)

        with torch.no_grad():
            net.eval()
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()

        test_accuracy = 100 * correct/total

        epoch_time = (time.time() - epoch_start_time) / 60
            
        print(f'Epoch {epoch + 1} Time {epoch_time:.2f} Mins Train Accuracy: {100 * correct_train / total_train}, Test Accuracy: {100 * correct / total}, training_loss: {train_loss:.4f}')
        logger.info(f"Epoch {epoch + 1} Train Accuracy: {100 * correct_train / total_train}, Test Accuracy: {100 * correct / total}")

         # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0 or epoch == epochs:
            checkpoint_path = f'./checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy
            }, checkpoint_path)
            logger.info(f"Checkpoint saved at {checkpoint_path}")
        
        torch.cuda.empty_cache()
        
        scheduler.step()

    torch.save(net.state_dict(), save_path)
    print('Finished Training')

if __name__ == '__main__' :

    parser = argparse.ArgumentParser(description='CIFAR-10 training with style transfer')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha value for style transfer (default: 1.0)')
    parser.add_argument('--prob_ratio', type=float, default=0.5, help='probability of applying style transfer (default: 0.5)')
    parser.add_argument('--content_dir', type=str, default='/kaggle/input/cifar10-python/cifar-10-batches-py/', help='CIFAR10 Directory')
    parser.add_argument('--style_dir', type=str, default='/kaggle/input/style-feats-adain-1000/style_feats_adain_1000.npy', help='Style_feats_directory')
    parser.add_argument('--randomize_alpha', type=bool, default=False, help='Make alpha random or fixed (default: False)')
    parser.add_argument('--rand_min', type=float, default=0.2, help='lower range for random alpha when randomize_alpha is True (deafault: 0.2)')
    parser.add_argument('--rand_max', type=float, default=1.0, help='Upper range for random alpha when randomize_alpha is True (deafault: 1.0)')
    parser.add_argument('--style_transfer_model', type=str, default='vgg', help='vgg or mobilenet')
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10 or cifar100')
    parser.add_argument('--gen_data', type=str, default=False, help='Use Generated data or not')
    parser.add_argument('--gen_nst_prob', type=float, default=0.5, help='NST probability on generated data')



    args = parser.parse_args()

    mp.set_start_method('spawn', force=True) 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    encoder, decoder = load_models(device=device, model_type = args.style_transfer_model)

    style_feats = load_feat_files(feats_dir=args.style_dir, device=device)

    nst_transfer = NSTTransform(style_feats, encoder=encoder, decoder=decoder, alpha=args.alpha, probability=args.prob_ratio, randomize=args.randomize_alpha, rand_min=args.rand_min, rand_max=args.rand_max)
    nst_transfer_gen = NSTTransform(style_feats, encoder=encoder, decoder=decoder, alpha=args.alpha, probability=args.gen_nst_prob, randomize=args.randomize_alpha, rand_min=args.rand_min, rand_max=args.rand_max)

    transform1 = nst_transfer
    transform2 = transforms.TrivialAugmentWide()


    transforms_list = [transform1, transform2]
    probabilities = [0.5, 0.5]

    random_choice_transform = RandomChoiceTransforms(transforms_list, probabilities)

    
 
    transform_train = transforms.Compose([
        nst_transfer,
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        #random_choice_transform,
        #GeometricTrivialAugmentWide(),  
        #transforms.TrivialAugmentWide(),
        #transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
 
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Transforms in transform_train: ")
    for t in transform_train.transforms:
        print(type(t))

    batch_size = args.batch_size
    target_size = 50000

    cifar_10_dir = args.content_dir

    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        if args.gen_data is True:
            baseset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
            transform_gen = transforms.Compose([nst_transfer_gen, transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), #random_choice_transform, #transforms.TrivialAugmentWide(), 
                                                transforms.ToTensor(),])
            trainset = load_augmented_traindata(base_trainset=baseset, style_transfer=nst_transfer, target_size=target_size, transforms_generated=transform_gen)
        net = WideResNet_28_4(num_classes=10)

    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        net = WideResNet_28_4(num_classes=100)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, pin_memory=True, num_workers=4)

    #testset = torchvision.datasets.CIFAR100(data_dir=cifar_10_dir, train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, pin_memory=True, num_workers=4)
    

    #net = WideResNet_28_4(num_classes=100)
    net.to(device)

    trainer_fn(epochs=args.epochs, net=net, trainloader=trainloader, testloader=testloader, device=device, save_path='./cifar_net.pth')


    

            




    

