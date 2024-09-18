import argparse
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image, ImageFile
from mobilenet import MobilenetEncoder, MobilenetDecoder  # Import MobileNet encoder and decoder

# Enable CUDNN backend and handle large images
cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Adaptive Instance Normalization (AdaIN) function
def adaptive_instance_normalization(content_feat, style_feat):
    content_mean, content_std = calc_mean_std(content_feat)
    style_mean, style_std = calc_mean_std(style_feat)
    
    normalized_feat = (content_feat - content_mean) / content_std
    adain_feat = normalized_feat * style_std + style_mean
    return adain_feat

def calc_mean_std(feat, eps=1e-5):
    N, C = feat.size()[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def train_transform():
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def adjust_learning_rate(optimizer, iteration_count, lr, lr_decay):
    """Adjust learning rate based on the iteration count."""
    new_lr = lr / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def train(encoder, decoder, content_loader, style_loader, args):
    # Set models to training mode
    decoder.train()
    encoder.eval()  # Encoder is pretrained and frozen

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)

    content_iter = iter(content_loader)
    style_iter = iter(style_loader)

    for i in range(args.max_iter):
        adjust_learning_rate(optimizer, i, args.lr, args.lr_decay)

        try:
            content_images, _ = next(content_iter)
            
        except StopIteration:
            content_iter = iter(content_loader)
            content_images, _ = next(content_iter)
            
        content_images = content_images.to(args.device)

        try:
            style_images = next(style_iter)
        except StopIteration:
            style_iter = iter(style_loader)
            style_images = next(style_iter)
        style_images = style_images.to(args.device)

        # Calculate loss
        loss, content_loss, style_loss = compute_loss(encoder, decoder, content_images, style_images, args)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log and save outputs periodically
        if (i + 1) % args.log_interval == 0 or (i + 1) == args.max_iter:
            print(f"Iteration [{i + 1}/{args.max_iter}], Loss: {loss.item():.4f}, Content Loss: {content_loss.item():.4f}, Style Loss: {style_loss.item():.4f}")
            save_image(content_images, f"{args.save_dir}/content_{i + 1}.jpg")
            save_image(style_images, f"{args.save_dir}/style_{i + 1}.jpg")
            output = decoder(t)
            save_image(output, f"{args.save_dir}/output_{i + 1}.jpg")



def compute_loss(encoder, decoder, content_image, style_image, args):
    # Extract content and style features from the encoder at multiple layers
    content_features = []
    style_features = []
    for i in range(len(encoder.btnecks)):
        content_features.append(encoder(i, content_image))
        style_features.append(encoder(i, style_image))

    # Perform AdaIN on the deepest layer
    t = adaptive_instance_normalization(content_features[-1], style_features[-1])

    # Reconstruct image through the decoder
    output = decoder(t)

    # Compute content loss
    output_content_features = encoder(len(encoder.btnecks) - 1, output)
    content_loss = nn.MSELoss()(output_content_features, content_features[-1])

    # Compute style loss over multiple layers
    style_loss = 0
    for of, sf in zip([encoder(i, output) for i in range(len(encoder.btnecks))], style_features):
        of_mean, of_std = calc_mean_std(of)
        sf_mean, sf_std = calc_mean_std(sf)
        style_loss += nn.MSELoss()(of_mean, sf_mean) + nn.MSELoss()(of_std, sf_std)

    # Total loss
    loss = args.content_weight * content_loss + args.style_weight * style_loss
    return loss, content_loss, style_loss



def main():
    parser = argparse.ArgumentParser()

    # Basic options
    parser.add_argument('--content_dir', type=str, required=True, help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', type=str, required=True, help='Directory path to a batch of style images')
    parser.add_argument('--save_dir', default='./outputs', help='Directory to save the output images')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5, help='Learning rate decay')
    parser.add_argument('--max_iter', type=int, default=160000, help='Maximum number of training iterations')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--style_weight', type=float, default=10.0, help='Weight for style loss')
    parser.add_argument('--content_weight', type=float, default=1.0, help='Weight for content loss')
    parser.add_argument('--n_threads', type=int, default=4, help='Number of threads for data loading')
    parser.add_argument('--log_interval', type=int, default=100, help='Interval for logging and saving outputs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    
    args = parser.parse_args()

    # Create necessary directories
    Path(args.save_dir).mkdir(exist_ok=True, parents=True)

    # Initialize models
    encoder = MobilenetEncoder().to(args.device)
    decoder = MobilenetDecoder().to(args.device)

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor()
    ])

    # Data loading
    content_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    style_dataset = FlatFolderDataset(args.style_dir, transform_train)

    content_loader = data.DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)
    style_loader = data.DataLoader(style_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)

    # Train the model
    train(encoder, decoder, content_loader, style_loader, args)



if __name__ == '__main__':
    main()
