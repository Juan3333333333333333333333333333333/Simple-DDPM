import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def load_dataset(image_size=64,batch_size=32)->DataLoader:
    tranforms=transforms.Compose([transforms.Resize((image_size,image_size)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Lambda(lambda x: 2*x-1),])
    dataset = datasets.ImageFolder(root='ae4dc-main/anime-data/anime-data/anime-faces', transform=tranforms)
    return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,drop_last=True)

def show_tensor_image(image):
    reverse_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )

    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))