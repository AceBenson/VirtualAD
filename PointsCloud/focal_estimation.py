import cv2
import numpy as np
from oct2py import octave
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image

octave.addpath('./matlab_modules/')
octave.eval('pkg load image')

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

def create_point_cloud(depth_image, f, scale):
    shape = depth_image.shape
    rows = shape[0]
    cols = shape[1]
    cx, cy = rows / 2, cols / 2

    points = np.zeros((rows * cols, 3), np.float32)

    # Linear iterator for convenience
    i = 0
    # For each pixel in the image...
    for r in range(0, rows):
        for c in range(0, cols):
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

    depth = np.squeeze(depth)
    if np.amax(depth) > 1.0:
        depth = depth / np.amax(depth)
    h, w, _ = rgb.shape
    dh, dw = depth.shape
    if dh != h or dw != w:
        dimg = Image.fromarray(depth)
        dimg_resized = dimg.resize((w, h), Image.BICUBIC)
        depth = np.asarray(dimg_resized)

    # Estimate focal length
    focal = octave.findFocal(rgb)
    print(f"focal {focal}")

    points_cloud = create_point_cloud(depth,  f=focal, scale=1)
    print("points_cloud: ")
    print(points_cloud.shape)
    print(points_cloud)

    pcd = o3d.geometry.PointCloud()
    r = rgb[:, :, 0].flatten()
    g = rgb[:, :, 1].flatten()
    b = rgb[:, :, 2].flatten()
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