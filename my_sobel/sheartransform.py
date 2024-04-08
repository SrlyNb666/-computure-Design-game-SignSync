import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch.nn.functional as F
import torch
import torchvision
import ptwt
torch.cuda.set_device(0)
import math
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def calculate_non_black_ratio(batch):
    # img = batch[0]/2 + 0.5
    # img = img.cpu()
    # npimg = img.numpy()
    # plt.imsave('image0.png', np.transpose(npimg, (1, 2, 0)))
    image_gray = 0.2989 * batch[:, 0, :, :] + 0.5870 * batch[:, 1, :, :] + 0.1140 * batch[:, 2, :, :]
    # 归一化
    image_gray = (image_gray - image_gray.min()) / (image_gray.max() - image_gray.min())
    torchvision.utils.save_image(image_gray[0], 'image1.png')
    return image_gray

def calculate_num_blocks(pixels_per_block):
    return int(math.sqrt(pixels_per_block))

def split_image_into_blocks(image_gray, pixels_per_block):
    block_height=image_gray.shape[1]//pixels_per_block
    block_width=image_gray.shape[2]//pixels_per_block
    blocks = image_gray.unfold(1, block_height, block_height).unfold(2, block_width, block_width) # 调整张量的形状，使每个块成为一个通道 
    blocks = blocks.contiguous().view(image_gray.shape[0], -1, block_height, block_width) 
    return blocks

def concatenate_blocks(blocks, num_blocks):
    original_height = num_blocks * blocks.shape[2]
    original_width = num_blocks * blocks.shape[3]
    blocks = blocks.view(-1, num_blocks, num_blocks, blocks.shape[2], blocks.shape[3]).permute(0, 1, 3, 2, 4)
    image_gray_reconstructed = blocks.contiguous().view(-1, original_height, original_width).unsqueeze(1).expand(-1, 3, -1, -1)
    return image_gray_reconstructed

def process_blocks(blocks, num_blocks):
    coeffs = ptwt.wavedec2(blocks, 'haar',level=1 )
    brightness_mean = coeffs[0].mean(dim=(2, 3), keepdim=True)
    # 创建一个只包含明系数的系数张量
    bright_coeffs = [torch.where(c >= brightness_mean, c, 0) if not isinstance(c, tuple) else tuple(torch.where(sub_c >= brightness_mean, sub_c, 0) for sub_c in c) for c in coeffs]
    bright_recon = ptwt.waverec2(bright_coeffs, 'haar')
    bright_recon = bright_recon[:,:,:blocks.shape[2], :blocks.shape[3]]
    binary_image = torch.where(bright_recon > 0, torch.tensor([1], device=bright_recon.device), torch.tensor([0], device=bright_recon.device))
    num_channels = binary_image.size(1)
    kernel = torch.ones((num_channels, 1, 2, 2), device="cuda")
    eroded_image = F.conv2d((1 - binary_image).float().to(kernel.device), kernel, padding=1, groups=num_channels) > 0
    dilated_image = F.conv2d(binary_image.float().to(kernel.device), kernel, padding=1, groups=num_channels) > 0
    boundary_binary = dilated_image ^ eroded_image
    boundary_binary = boundary_binary.narrow(2, 0, blocks.shape[2])
    boundary_binary = boundary_binary.narrow(3, 0, blocks.shape[3])
    blocks = torch.where(boundary_binary, blocks,torch.tensor(255.0, device=boundary_binary.device) )    
    blocks = concatenate_blocks(blocks, num_blocks)
    #torchvision.utils.save_image(blocks[0].float(), 'image2.png')
    return blocks

