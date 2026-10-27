import numpy as np

import torch
import torchvision

from timm.models.layers.helpers import to_3tuple

import pdb

def patches3d_to_grid(patches, patch_size=16, grid_size=8, in_chans=4, n_group=3, hidden_axis='d', slice_pos_list=[0.4, 0.45, 0.5, 0.55, 0.6]):











    B, L, C = patches.shape
    patch_size = to_3tuple(patch_size)
    grid_size = to_3tuple(grid_size)
    assert np.prod(grid_size) == L and np.prod(patch_size) * in_chans == C, "Shape of input doesn't match parameters"


    patches = patches.reshape(B, *grid_size, *patch_size, in_chans)

    image = patches.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, in_chans, 1,
                                                            grid_size[0] * patch_size[0],
                                                            grid_size[1] * patch_size[1],
                                                            grid_size[2] * patch_size[2])

    assert B % n_group == 0
    n_per_row = len(slice_pos_list) * in_chans * B // n_group

    if hidden_axis == 'd':
        slice_list = []
        for slice_pos in slice_pos_list:
            slice_list.append(image[..., :, :, int(image.size(-1) * slice_pos)])
        image = torch.cat(slice_list, dim=2)
    else:
        raise ValueError(f"Only support D for now")
    visH, visW = image.size(-2), image.size(-1)
    grid_of_images = torchvision.utils.make_grid(image.reshape(B * len(slice_pos_list) * in_chans, 1, visH, visW), nrow=n_per_row)

    grid_of_images.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    return grid_of_images

def images3d_to_grid(image, n_group=3, hidden_axis='d', slice_pos_list=[0.3, 0.4, 0.5, 0.6, 0.7]):








    B, C, H, W, D = image.shape

    assert B % n_group == 0
    n_per_row = B // n_group
    list_of_grid_images = []
    for slice_pos in slice_pos_list:
        if hidden_axis == 'd':
            image_slice = image[..., :, :, int(D * slice_pos)]
        else:
            raise ValueError(f"Only support D for now")

        grid_of_images = torchvision.utils.make_grid(image_slice, nrow=n_per_row)
        grid_of_images.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        list_of_grid_images.append(grid_of_images)

    return list_of_grid_images
