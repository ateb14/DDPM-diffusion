import torch
from torchvision.utils import save_image
from ddpm import Diffusion
from utils import get_data

import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.batch_size = 1  # 5
args.image_size = 64
args.dataset_path = r"./cifar10png/train"

dataloader = get_data(args)

device = 'cpu'
diff = Diffusion(device=device)
image, label = next(iter(dataloader))
imgae = image.to(device)
t = torch.Tensor([0, 50, 100, 150, 200, 300, 600, 700, 999]).long().to(device)

noised_image, _ = diff.noisy_images(image, t)
save_image(noised_image.add(1).mul(0.5), f"noise_{label.item()}.jpg")