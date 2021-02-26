import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d

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

    i = 0
    for r in range(0, rows):
        for c in range(0, cols):
            if (not index[r, c]):
                continue
            depth = depth_image[r, c]
            z = (1-depth) * scale
            points[i, 0] = cx + (c - cx) * (z / f)
            points[i, 1] = cy + (r - cy) * (z / f)
            points[i, 2] = z
            i = i + 1
    return points

def find_plane(pcd):
	plane_model, inliers = pcd.segment_plane(distance_threshold=0.015,
											 ransac_n=3,
											 num_iterations=1000)
	[a, b, c, d] = plane_model
	print(f'Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0')

	inlier_clouds = pcd.select_by_index(inliers)
	inlier_clouds.paint_uniform_color([1.0, 0.0, 0.0])
	outlier_clouds = pcd.select_by_index(inliers, invert=True)
	return [inlier_clouds, outlier_clouds]

def main(args):
    # Read files
    rgb = read_file(args.image)
    depth = read_file(args.depth)
    mask = read_file(args.mask)
    focal = args.focal

    # Deal with depth input
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
    index2 = mask[:, :, 2] == 0/255
    index = np.logical_and(index0, index1, index2)

    # Create points from depth by pinhole model
    points_cloud = create_point_cloud(depth, f=focal, scale=1, index=index)

    # Set point cloud color
    r = rgb[index][:, 0].flatten()
    g = rgb[index][:, 1].flatten()
    b = rgb[index][:, 2].flatten()
    myc = np.stack([r, g, b], axis=1)

    # PointCloud 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_cloud)
    pcd.colors = o3d.utility.Vector3dVector(myc)
    pcd.transform([
        [1, 0, 0, 0],
        [0,-1, 0, 0],
        [0, 0,-1, 0],
        [0, 0, 0, 1]
        ]) # flip, otherwise the pcd will be upside down
    o3d.visualization.draw_geometries([pcd])

    # Fit Plane
    vis_pcd = find_plane(pcd)
    o3d.visualization.draw_geometries(vis_pcd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default='./Images/test_resized.jpeg')
    parser.add_argument("--depth", type=str, default='./Images/test_depth.png')
    parser.add_argument("--mask", type=str, default='./Images/pred_test_resized.png')
    parser.add_argument("--focal", type=float, default=1324.0)
    args = parser.parse_args()
    main(args)