import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Subset
import torchvision.transforms.functional as TF
import numpy as np

class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform augmentations on Generated Data"""

    def __init__(self, images, labels, sources, transforms_preprocess, transforms_basic, transforms_augmentation,
                 transforms_generated=None):
        self.images = images
        self.labels = labels
        self.sources = sources
        self.preprocess = transforms_preprocess
        self.transforms_basic = transforms_basic
        self.transforms_augmentation = transforms_augmentation
        self.transforms_generated = transforms_generated #if transforms_generated else transforms_augmentation

        print("Transforms in transform_augmentation: ")
        for transform in self.transforms_augmentation.transforms:
            for t in transform.transforms:
                print(type(t))
        
        print("Transforms in transform_generated: ")
        for transform in self.transforms_generated.transforms:
            for t in transform.transforms:
                print(type(t))

        
    def __getitem__(self, i):
        x = self.images[i]
        aug_strat = self.transforms_augmentation if self.sources[i] == True else self.transforms_generated
        augment = aug_strat #transforms.Compose([self.transforms_basic, aug_strat])
        #print(self.sources[i])
        
        return augment(x), self.labels[i]
        

    def __len__(self):
        return len(self.labels)

def load_augmented_traindata(base_trainset, target_size, dataset, tf, seed=0, transforms_generated = None, generated_ratio = 0.5, robust_samples=0):

        transforms_generated = transforms_generated
        robust_samples = robust_samples
        target_size = target_size
        generated_ratio = generated_ratio
        if dataset == 'cifar10':
            generated_dataset = np.load(f'/kaggle/input/cifar10-1m-npz/1m.npz',
                                    mmap_mode='r') if generated_ratio > 0.0 else None
        elif dataset == 'cifar100':
            generated_dataset = np.load(f'/kaggle/input/cifar-100-1m-generated/1m.npz',
                                    mmap_mode='r') if generated_ratio > 0.0 else None

        flip = transforms.RandomHorizontalFlip()
        c32 = transforms.RandomCrop(32, padding=4)
        t = transforms.ToTensor()
        
        transforms_preprocess = transforms.Compose([t])
        transforms_basic = transforms.Compose([flip, c32])
        tf = tf
        transforms_augmentation = transforms.Compose([transforms_basic, tf,  transforms_preprocess])
        transform_generated = transforms.Compose([transforms_basic, transforms_generated, transforms_preprocess])

        

        #torch.manual_seed(seed)
        #np.random.seed(seed)
        #random.seed(seed)
        
        for key in generated_dataset:
            array = generated_dataset[key]
            print(f"{key}: shape = {array.shape}, length = {array.size}")

        # Prepare lists for combined data
        images = [None] * target_size
        labels = [None] * target_size
        sources = [None] * target_size
 
        if generated_dataset == None or generated_ratio == 0.0:
            images, labels = zip(*base_trainset)
            if isinstance(images[0], torch.Tensor):
                images = TF.to_pil_image(images)
            sources = [True] * len(base_trainset)
        else:
            num_generated = int(target_size * generated_ratio)
            num_original = target_size - num_generated
            # Create a single permutation for the whole epoch
            original_perm = torch.randperm(len(base_trainset))
            generated_perm = torch.randperm(len(generated_dataset['image']))

            original_indices = original_perm[0:num_original]
            generated_indices = generated_perm[0:num_generated]
            generated_images = list(map(Image.fromarray, generated_dataset['image'][generated_indices]))
            generated_labels = generated_dataset['label'][generated_indices]

            original_subset = Subset(base_trainset, original_indices)
            #original_images, original_labels = zip(*original_subset)
            original_images, original_labels = map(list, zip(*original_subset))

            if isinstance(original_images[0], torch.Tensor):
                #original_images = [TF.to_pil_image(img) for img in original_images if isinstance(img, torch.Tensor)]
                original_images = TF.to_pil_image(original_images)


            # Transform and append original data
            images[:num_original] = original_images
            labels[:num_original] = original_labels
            sources[:num_original] = [True] * num_original

            # Append NPZ data
            images[num_original:target_size] = generated_images
            labels[num_original:target_size] = generated_labels
            sources[num_original:target_size] = [False] * num_generated

        return AugmentedDataset(images, labels, sources, transforms_preprocess,
                                         transforms_basic, transforms_augmentation, transforms_generated=transform_generated,
                                        )