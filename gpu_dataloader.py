import torch
from torch.utils.data import DataLoader, Dataset
import threading
import queue
import time

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
        data = [self.dataset[i] for i in batch]
        data = self.collate_fn(data)
        if self.gpu_transform:
            with torch.no_grad():
                data = self.gpu_transform(data.to(self.device))
        return data

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