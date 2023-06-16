import torch
import torch.nn as nn

print(torch.cuda.is_available())
print(torch.cuda.device_count())

for i in range(4):
    print(torch.cuda.get_device_name(i))