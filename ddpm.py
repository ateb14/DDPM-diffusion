import os
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from utils import *
from modules import UNet, UNet_with_class, EMA
import logging
import copy
# from torch.utils.tensorboard import SummaryWriter

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

    def sample(self, model, batchsize, labels, cfg_scale=3):
        logging.info(f"Sampling {batchsize} images...")
        with torch.no_grad():
            x = torch.randn(batchsize, 3, self.img_size, self.img_size).to(
                self.device)  # sample from Gaussian
            for i in reversed(range(1, self.noise_step)):
                t = torch.tensor(i).long().repeat(batchsize).to(self.device)
                predicted_noise = model(x, t, labels)

                if cfg_scale > 0:
                    # bilinear interpolation
                    uncond_pred_noise = model(x, t, None)
                    predicted_noise = torch.lerp(
                        uncond_pred_noise, predicted_noise, cfg_scale)

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
    # model = UNet().to(device)
    model = UNet_with_class(num_classes=args.num_classes, device=device)
    model = nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    # logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        # pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timestep(images.shape[0]).to(device)
            x_t, noise = diffusion.noisy_images(images, t)

            if np.random.rand() < 0.1:
                labels = None

            # predicted_noise = model(x_t, t)
            predicted_noise = model(x_t, t, labels)
            loss = mse(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            # pbar.set_postfix(MSE=loss.item())
            if i % 64 == 0 or i == l-1:
                logging.info(f"MSE {loss.item()}, epoch {epoch + 1}, iteration {i}")
            # logger.add_scalar("MSE", loss.item(), epoch * l + i)

        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch + 1}, saving checkpoint...")
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, batchsize=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, batchsize=len(labels), labels=labels)
            # plot_images(sampled_images)
            save_images(sampled_images, os.path.join(
                "results", args.run_name, f"epoch_{epoch+1}.jpg"))
            save_images(ema_sampled_images, os.path.join(
                "results", args.run_name, f"epoch_{epoch+1}_ema.jpg"))
            
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), os.path.join(
                    "models", args.run_name, f"epoch_{epoch+1}.pt"))
                torch.save(ema_model.module.state_dict(), os.path.join("models", args.run_name, f"epoch_{epoch+1}_ema.pt"))
            else:
                torch.save(model.state_dict(), os.path.join(
                    "models", args.run_name, f"epoch_{epoch+1}.pt"))
                torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"epoch_{epoch+1}_ema.pt"))

            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_condtional"
    args.epochs = 300
    args.batch_size = 160
    args.image_size = 64
    args.num_classes = 10  # cifar10
    args.dataset_path = r"./cifar10png/train"
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.lr = 3e-4
    if torch.cuda.is_available():
        print("CUDA enabled!")
        gpu_cnt = torch.cuda.device_count()
        print(gpu_cnt, 'gpus are working, which are:')
        for i in range(gpu_cnt):
            print('  ', torch.cuda.get_device_name(i))
    else:
        print("Turtle speed cpu")

    train(args)


if __name__ == '__main__':
    launch()
