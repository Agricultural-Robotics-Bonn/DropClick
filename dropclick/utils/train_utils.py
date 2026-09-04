# Adapted by Patrick Zimmer from https://github.com/amitrana001/DynaMITe
import math
import torch
import numpy as np
import copy
import cv2
import random

                  
def get_spatiotemporal_embeddings(pos_tensor, positional_embeddings):
        
        scale = 2 * math.pi
        if positional_embeddings == "temporal":
            dim_t = torch.arange(256, dtype=torch.float, device=pos_tensor.device)
            dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / 256)
            t_embed = pos_tensor[:, :, 2] * scale
            pos_t = t_embed[:, :, None] / dim_t
            pos_t[:, :, 0::2][torch.where(pos_t[:, :, 0::2] < 0)] = 0.0
            pos_t[:, :, 1::2][torch.where(pos_t[:, :, 1::2] < 0)] = math.pi/2
            pos_t = torch.stack((pos_t[:, :, 0::2].sin(), pos_t[:, :, 1::2].cos()), dim=3).flatten(2)
            return pos_t
        dim_t = torch.arange(128, dtype=torch.float, device=pos_tensor.device)
        dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / 128)
        x_embed = pos_tensor[:, :, 1] * scale
        y_embed = pos_tensor[:, :, 0] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x[:, :, 0::2][torch.where(pos_x[:, :, 0::2] < 0)] = 0.0
        pos_x[:, :, 1::2][torch.where(pos_x[:, :, 1::2] < 0)] = math.pi/2
        pos_y[:, :, 0::2][torch.where(pos_y[:, :, 0::2] < 0)] = 0.0
        pos_y[:, :, 1::2][torch.where(pos_y[:, :, 1::2] < 0)] = math.pi/2
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

        if positional_embeddings == "spatial":
            return torch.cat((pos_y, pos_x), dim=2)
        elif positional_embeddings == "spatio_temporal":
            t_embed = pos_tensor[:, :, 2] * scale
            pos_t = t_embed[:, :, None] / dim_t
            pos_t[:, :, 0::2][torch.where(pos_t[:, :, 0::2] < 0)] = 0.0
            pos_t[:, :, 1::2][torch.where(pos_t[:, :, 1::2] < 0)] = math.pi/2
            pos_t = torch.stack((pos_t[:, :, 0::2].sin(), pos_t[:, :, 1::2].cos()), dim=3).flatten(2)
            
            pos = torch.cat((pos_y, pos_x, pos_t), dim=2)
            return pos
    
def get_pos_tensor_coords(fg_coords, num_queries, height, width, device, max_timestamp=None):

    #fg_coords: batch x (list of list of fg coords) [y,x,t]

    # return
    # points: Bs x num_queries x 3 
    B = len(fg_coords)
    
    pos_tensor = []
    
    for i, fg_coords_per_image in enumerate(fg_coords):
        coords_per_image  = []
        if max_timestamp is not None:
            t = max(max_timestamp[i],500)
        for fg_coords_per_mask in fg_coords_per_image:
            for coords in fg_coords_per_mask:
                if max_timestamp is not None:
                    coords_per_image.append([coords[0]/height, coords[1]/width, coords[2]/t])
                else:
                    coords_per_image.append([coords[0]/height, coords[1]/width, coords[2]])
        coords_per_image.extend([[-1.0,-1.0,-1.0]] * (num_queries-len(coords_per_image)))
        pos_tensor.append(torch.tensor(coords_per_image,device=device))
    pos_tensor = torch.stack(pos_tensor)
    return pos_tensor