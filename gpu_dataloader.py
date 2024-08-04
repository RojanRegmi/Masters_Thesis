import torch
from torch.utils.data import DataLoader, Dataset
import threading
import queue
import time
from PIL import Image
import torchvision.transforms as transforms

class GPUTransformDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, 
                 pin_memory=False, drop_last=False, gpu_transform=None, 
                 prefetch_factor=2, device='cuda'):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, 
                         num_workers=num_workers, pin_memory=pin_memory, 
                         drop_last=drop_last)
        
        self.gpu_transform = gpu_transform
        self.prefetch_factor = prefetch_factor
        self.device = device
        
        self.queue = queue.Queue(maxsize=self.prefetch_factor)
        self.stream = torch.cuda.Stream()
        self.stop_event = threading.Event()
        
        self.preprocess_thread = threading.Thread(target=self._preprocess_thread)
        self.preprocess_thread.start()

    def _preprocess_thread(self):
        torch.cuda.set_device(self.device)
        while not self.stop_event.is_set():
            try:

                if not hasattr(self, 'batch_sampler_iter'):
                    time.sleep(0.1)
                    continue

                batch = next(self.batch_sampler_iter)
                with torch.cuda.stream(self.stream):
                    processed_batch = self._process_batch(batch)
                self.queue.put(processed_batch)
            except StopIteration:
                break

    def _process_batch(self, batch):
        # Debugging: Print batch structure
        print(f"Batch type: {type(batch)}, Batch contents: {batch}")
        
        # Ensure batch is a list of (img, target) tuples
        if not all(isinstance(item, tuple) and len(item) == 2 for item in batch):
            raise ValueError(f"Unexpected batch format: {batch}")
        
        # Separate images and targets
        images, targets = zip(*batch)
        
        # Convert images to a single tensor
        if isinstance(images[0], Image.Image):
            # If images are PIL Images, convert to tensors
            images = torch.stack([transforms.ToTensor()(img) for img in images])
        elif isinstance(images[0], torch.Tensor):
            # If images are already tensors, just stack them
            images = torch.stack(images)
        else:
            raise TypeError(f"Unexpected image type: {type(images[0])}")
        
        # Move images to GPU
        images = images.to(self.device)
        
        # Convert targets to tensor and move to GPU
        targets = torch.tensor(targets).to(self.device)
        
        # Apply GPU transform
        transformed_images = self.gpu_transform(images)
        
        return transformed_images, targets


    def __iter__(self):
        self.batch_sampler_iter = iter(self.batch_sampler)
        return self

    def __next__(self):
        if self.stop_event.is_set() and self.queue.empty():
            raise StopIteration
        return self.queue.get()

    def __del__(self):
        self.stop_event.set()
        self.preprocess_thread.join()