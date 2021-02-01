
import cv2
import numpy as np
import numba
import torch

from oct2py import octave


# download this pkg
# octave.eval('pkg install "./image-2.10.0.tar.gz"')
octave.addpath('./matlab_modules/')
octave.eval('pkg load image')

# MegaDepth net output

# megadepth_prediction = 'pred_depth.pth'
# depth_image = torch.load(megadepth_prediction).detach().cpu().numpy()

# H, W = depth_image.shape

# mask = cv2.imread('mask0.png', 0)
# mask = cv2.resize(mask, (W, H), cv2.INTER_AREA)

im = cv2.imread('DSC03170.jpg')
# im = cv2.resize(im, (W, H), cv2.INTER_AREA)
rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

focal = octave.findFocal(rgb)
print(f"focal {focal}")

# @numba.jit
# def create_point_cloud(depth_image, f, scale):
#     shape = depth_image.shape;
#     rows = shape[0];
#     cols = shape[1];

#     cx, cy = rows / 2, cols / 2

#     print(shape, cx, cy)
#     points = np.zeros((rows * cols, 3), np.float32);

#     # Linear iterator for convenience
#     i = 0
#     # For each pixel in the image...
#     for r in range(0, rows):
#         for c in range(0, cols):
#             # Get the depth in bytes
#             depth = depth_image[r, c];

#             # If the depth is 0x0 or 0xFF, its invalid.
#             # By convention it should be replaced by a NaN depth.

#             z = depth * scale;

#             # Get the x, y, z coordinates in units of the pixel
#             points[i, 0] = cx + (c - cx) * (z / f);
#             points[i, 1] = cy + (r - cy) * (z / f);
#             points[i, 2] = z
#             # else:
#             #     # Invalid points have a NaN depth
#             #     points[i, 2] = np.nan;
#             i = i + 1

#     print(z)
#     print(points.max(axis=0))
#     return points.astype(np.uint16)



# points_cloud = create_point_cloud(depth_image * (mask > 0) * 1,  f=focal, scale=1000)