import torch
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
from wrn_moex import WideResNet_28_4_WithPoNoAndIN

# Define the training function
def train(train_loader_content, train_loader_style, model, criterion, optimizer, epoch, moex_prob=0.5):
    model.train()  # Set the model to training mode

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()

    style_iter = iter(train_loader_style)  # Create an iterator for the style loader
    for i, (input_content, target) in enumerate(train_loader_content):
        data_time.update(time.time() - end)
        
        input_content = input_content.cuda()
        target = target.cuda()

        # Try to get the next batch from the style dataset, loop if necessary
        try:
            input_style = next(style_iter)  # We don't care about the target from the style dataset
        except StopIteration:
            style_iter = iter(train_loader_style)
            input_style = next(style_iter)
        
        input_style = input_style.cuda()

        # Apply MoEx with a certain probability
        if torch.rand(1).item() < moex_prob:
            # Moment exchange between content and style images
            output = model(input_content, image2=input_style, moex_type='in')  # You can also use 'pono' if needed
        else:
            # No MoEx, regular forward pass
            output = model(input_content)

        loss = criterion(output, target)

        # Measure accuracy and loss
        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), input_content.size(0))
        top1.update(acc1.item(), input_content.size(0))

        # Compute gradient and perform optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader_content)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})')

# Define AverageMeter for tracking performance
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Define accuracy function
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# Main training script
def main():
    # Hyperparameters and settings
    batch_size = 256
    epochs = 100
    lr = 0.1

    # Define transformations for the datasets
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load primary (content) dataset
    train_dataset_content = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader_content = torch.utils.data.DataLoader(train_dataset_content, batch_size=batch_size, shuffle=True)

    # Load secondary (style) dataset
    train_dataset_style = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_loader_style = torch.utils.data.DataLoader(train_dataset_style, batch_size=batch_size, shuffle=True)

    # Define the model, loss, and optimizer
    model = WideResNet_28_4_WithPoNoAndIN(num_classes=10).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    cudnn.benchmark = True

    # Train the model
    for epoch in range(epochs):
        train(train_loader_content, train_loader_style, model, criterion, optimizer, epoch)

if __name__ == '__main__':
    main()
