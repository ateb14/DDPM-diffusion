import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ddpm import Diffusion
from modules import UNet_with_class
from utils import save_images
import os
import datetime

date = datetime.datetime.now().strftime("%Y_%m_%d")
# mkdir if not exist
os.makedirs("sampled_images", exist_ok=True)
os.makedirs(os.path.join("sampled_images", date), exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet_with_class(device=device, num_classes=10)
model.load_state_dict(torch.load('models/DDPM_condtional/epoch_291_ema.pt'))
model = model.to(device)
model = nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()

diff = Diffusion(device=device)

# input user command(int)
print('Model loaded, input image categories to generate images')
print('0: airplane')
print('1: automobile')
print('2: bird')
print('3: cat')
print('4: deer')
print('5: dog')
print('6: frog')
print('7: horse')
print('8: ship')
print('9: truck')
id2name = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
           4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
while True:
    command = input("Input image categories: ")
    command = int(command)
    if command < 0 or command > 9:
        break
    batchsize = input('Input batchsize: ')
    batchsize = int(batchsize)
    labels = torch.tensor([command] * batchsize).long().to(device)
    sampled_images = diff.sample(model=model, batchsize=batchsize, labels=labels)
    time = datetime.datetime.now().strftime('_%H_%M_%S')
    save_images(sampled_images, os.path.join(
                "sampled_images", date, id2name[command]+time+".jpg"))
