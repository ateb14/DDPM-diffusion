import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                    level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_step=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cpu"):
        self.noise_step = noise_step
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule()
        self.alpha = 1 - self.beta
        self.alpha_hat = self.alpha.cumprod(dim=0)

    # schedule the variance of the noise
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_step).to(self.device)

    # add noise to the image given the timestep, return the noisy image and the sampled noise
    def noisy_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[
            :, None, None, None]  # fit the shape of x
        sqrt_one_minus_alpha = torch.sqrt(
            1 - self.alpha_hat[t])[:, None, None, None]

        epsilon = torch.randn_like(x)

        return sqrt_alpha_hat * x + sqrt_one_minus_alpha * epsilon, epsilon

    def sample_timestep(self, batchsize):
        return torch.randint(1, self.noise_step, (batchsize,)).to(self.device)

    def sample(self, model, batchsize):
        logging.info(f"Sampling {batchsize} images...")
        with torch.no_grad():
            x = torch.randn(batchsize, 3, self.img_size, self.img_size).to(
                self.device)  # sample from Gaussian
            for i in tqdm(reversed(range(1, self.noise_step)), position=0):
                t = torch.tensor(i).long().repeat(batchsize).to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:  # We don't need to sample noise for the first timestep, as we will get out final clean image
                    noise = torch.zeros_like(x)
                x = 1/(torch.sqrt(alpha)) * (x - beta/torch.sqrt(1-alpha_hat)
                                             * predicted_noise) + noise * torch.sqrt(beta)

        # Rescale to [0, 1], and then to [0, 255]
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timestep(images.shape[0]).to(device)
            x_t, noise = diffusion.noisy_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_prefix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), epoch * l + i)

        # for each epoch, we try to sample 12 images to see how the model is doing
        sampled_images = diffusion.sample(model, images.shape[0])
        save_images(sampled_images, os.path.join(
            "results", args.run_name, f"epoch_{epoch+1}.jpg"))
        torch.save(model.state_dict(), os.path.join(
            "checkpoints", args.run_name, f"epoch_{epoch+1}.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    args.dataset_path = r"~/~/~/~"
    args.device = "cpu"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()
