from torchvision import transforms
import torch

def downsample_image(image, size=(32, 32)):
    """
    Downsample the given PIL image to the specified size using nn.Upsample.
    """
    # Convert the PIL image to a tensor
    transform_to_tensor = transforms.ToTensor()
    image_tensor = transform_to_tensor(image).unsqueeze(0)  # Add batch dimension

    # Define the upsample transformation
    upsample = torch.nn.Upsample(size=size, mode='bilinear', align_corners=True)

    # Apply the upsample transformation
    downsampled_tensor = upsample(image_tensor)

    # Convert back to PIL image
    transform_to_pil = transforms.ToPILImage()
    downsampled_image = transform_to_pil(downsampled_tensor.squeeze(0))  # Remove batch dimension

    return downsampled_image