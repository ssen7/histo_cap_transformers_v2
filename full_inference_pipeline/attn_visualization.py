import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image
import h5py
import matplotlib.pyplot as plt

import sys
# sys.path.append('../gen_patch/')

# from ..gen_patch.wsi_core.wsi_utils import StitchCoords, DrawGrid
from gen_patch.wsi_core.WholeSlideImage import WholeSlideImage

# adapted from https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/utils/image.py
def show_cam_on_img(patch, attn, image_weight=0.5, colormap=cv2.COLORMAP_JET):
    
    w,h,c = patch.shape
    attn_cam = np.tile(np.tile(attn, w),h).reshape(w,h)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_cam), colormap)
    heatmap = np.float32(heatmap) / 255
    
    img = np.float32(patch) / 255
    
    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

# some code borrowed from https://github.com/mahmoodlab/CLAM/blob/master/wsi_core/wsi_utils.py
def gen_attn_viz(wsi_full_path, patch_path, attentions):

    # normalize attention values
    # attentions=attentions/np.max(attentions)
    # attentions=np.array([x if x>0.0 else 0 for x in attentions])

    wsi_object = WholeSlideImage(wsi_full_path)
    wsi = wsi_object.getOpenSlide()

    vis_level = wsi.get_best_level_for_downsample(64)
    w, h = wsi.level_dimensions[vis_level]

    downsamples = wsi_object.wsi.level_downsamples[vis_level]

    file = h5py.File(patch_path, 'r')
    dset = file['coords']
    coords = dset[:]
   
    patch_size = dset.attrs['patch_size']
    patch_level = dset.attrs['patch_level']
    patch_size = tuple((np.array((patch_size, patch_size)) * wsi.level_downsamples[patch_level]).astype(np.int32))
    
    heatmap = Image.new(size=(w,h), mode="RGB", color=(0,0,0))
    canvas= np.array(heatmap)
    
    indices = np.arange(len(coords))
    total = len(indices)
    patch_size = tuple(np.ceil((np.array(patch_size)/np.array(downsamples))).astype(np.int32))
    full_img=np.array(wsi.get_thumbnail(wsi.level_dimensions[vis_level]).convert('RGB'))
    
    for idx in range(total):
        patch_id = indices[idx]
        attn = attentions[idx]
        coord = coords[patch_id]

        patch = np.array(wsi_object.wsi.read_region(tuple(coord), vis_level, patch_size).convert("RGB"))
        patch = show_cam_on_img(patch, attn)

        coord = np.ceil(coord / downsamples).astype(np.int32)

        canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3].shape[:2]

        canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
        full_img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = 0

        sup_img = full_img+canvas
    
    # close h5py file
    file.close()
    return sup_img
#     plt.figure(figsize=(15,8))
#     plt.imshow(sup_img, cmap='jet')
#     plt.colorbar()