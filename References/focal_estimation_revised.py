import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d

# octave.addpath('./matlab_modules/')
# octave.eval('pkg load image')

def read_file(fname):
	if fname.endswith('.npy'):
		return np.load(fname)
	elif fname.endswith('.jpg') or fname.endswith('.jpeg'):
		img = plt.imread(fname)
		img = img.astype(np.float32) / 255.0
		return img			
	elif fname.endswith('.png'):
		img = plt.imread(fname)
		return img

def create_point_cloud(depth_image, f, scale, index):
    shape = depth_image.shape
    rows = shape[0]
    cols = shape[1]
    cx, cy = rows / 2, cols / 2

    count = np.count_nonzero(index)
    points = np.zeros((count, 3), np.float32)

    # Linear iterator for convenience
    i = 0
    # For each pixel in the image...
    for r in range(0, rows):
        for c in range(0, cols):
            # Skip non grandstand area
            if (not index[r, c]):
                continue

            # Get the depth in bytes
            depth = depth_image[r, c]

            # If the depth is 0x0 or 0xFF, its invalid.
            # By convention it should be replaced by a NaN depth.
            z = (1-depth) * scale

            # Get the x, y, z coordinates in units of the pixel
            points[i, 0] = cx + (c - cx) * (z / f)
            points[i, 1] = cy + (r - cy) * (z / f)
            points[i, 2] = z
            # else:
            #     # Invalid points have a NaN depth
            #     points[i, 2] = np.nan;
            i = i + 1
    return points


def main():
    rgb = read_file('../Images/wikicommons_field.jpg')
    depth = read_file('../Images/wikicommons_field_inv_depth.png')
    mask = read_file('../Images/pred_wikicommons_field.png')
    focal = 580.8723715937349

    depth = np.squeeze(depth)
    if np.amax(depth) > 1.0:
        depth = depth / np.amax(depth)
    h, w, _ = rgb.shape

    # Adjust depth image size
    dh, dw = depth.shape
    if dh != h or dw != w:
        dimg = Image.fromarray(depth)
        dimg_resized = dimg.resize((w, h), Image.BICUBIC)
        depth = np.asarray(dimg_resized)
    # Adjust mask image size
    mh, mw, _ = mask.shape
    if mh != h or mw != w:
        mimg = Image.fromarray(mask)
        mimg_resized = mimg.resize((w, h), Image.BICUBIC)
        mask = np.asarray(mimg_resized)

    # Get mask from PSPNet Result
    index0 = mask[:, :, 0] == 31/255
    index1 = mask[:, :, 1] == 255/255
    index2 = mask[:, :, 1] == 0/255
    index = np.logical_and(index0, index1, index2)
    
    # rgb[np.bitwise_not(index)] = 0
    # print(rgb[index].shape)

    # Estimate focal length
    # focal = octave.findFocal(rgb)
    print(f"1. Find focal: {focal}")

    # Create points from depth by pinhole model
    points_cloud = create_point_cloud(depth, f=focal, scale=1, index=index)
    print(f"2. Create points_cloud shape: {points_cloud.shape}")

    # Set open3d point cloud data
    pcd = o3d.geometry.PointCloud()
    r = rgb[index][:, 0].flatten()
    g = rgb[index][:, 1].flatten()
    b = rgb[index][:, 2].flatten()
    
    myc = np.stack([r, g, b], axis=1)
    pcd.points = o3d.utility.Vector3dVector(points_cloud)
    pcd.colors = o3d.utility.Vector3dVector(myc)
    pcd.transform([
        [1, 0, 0, 0],
        [0,-1, 0, 0],
        [0, 0,-1, 0],
        [0, 0, 0, 1]
        ]) # flip, otherwise the pcd will be upside down
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()