import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d
from pprint import pprint
import cv2

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
    cx, cy = cols / 2, rows / 2
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
	plane_model, inliers = pcd.segment_plane(distance_threshold=5,
											 ransac_n=3,
											 num_iterations=1000)
	[a, b, c, d] = plane_model
	print(f'Crowd Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0')

	inlier_clouds = pcd.select_by_index(inliers)
	inlier_clouds.paint_uniform_color([1.0, 0.0, 0.0])
	outlier_clouds = pcd.select_by_index(inliers, invert=True)
	return [inlier_clouds, outlier_clouds], [a, b, c, d]

def find_alignment(depth_image, f, scale, index):
    shape = depth_image.shape
    rows = shape[0]
    cols = shape[1]
    binaryImg = (index*255).astype(np.uint8)
    cannyImg = cv2.Canny(binaryImg, 50, 150)
    lines = cv2.HoughLines(cannyImg,1,np.pi/180,50)

    # Check if detect at least one line
    # if lines is not None:
    line = lines[0]
    rho,theta = line[0]
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a*rho, b*rho
    x1, y1 = 0, int(y0 + (x0 / b)*(a))
    x2, y2 = (cols-1), int(y0 + ((x0 - (cols-1)) / b)*(a))
    x0, y0 = int((x1 + x2) / 2), int((y1 + y2) / 2)
    x = [x0, x1, x2]
    y = [y0, y1, y2]

    # Alignment plane
    points = np.zeros((3, 3), np.float32)
    cx, cy = cols / 2, rows / 2

    for i in range(3):
        d = depth_image[y[i], x[i]]
        z = (1-d) * scale
        points[i, 0] = cx + (x[i] - cx) * (z / f)
        points[i, 1] = cy + (y[i] - cy) * (z / f)
        points[i, 2] = z

    # Plane Equation
    p1, p2, p3 = points
    v1 = p3 - p1
    v2 = p2 - p1
    cp = np.cross(v1, v2)
    a, b, c = cp
    d = -np.dot(cp, p3)
    print(f'Alignment Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0')
    
    # Sample points
    points_cloud = np.zeros((100*100, 3), np.float32)
    i = 0
    for x in np.linspace(p2[0], p3[0], 100):
        for y in np.linspace(p2[1], p3[1], 100):
            z = -((d + a*x + b*y) / c)
            points_cloud[i, 0] = x
            points_cloud[i, 1] = y
            points_cloud[i, 2] = z
            i = i + 1
    return points_cloud, [a, b, c, d]

def reference_axis():
    points_cloud_reference = np.zeros((100*3, 3), np.float32)
    i = 0
    for x in range(3):
        for y in np.linspace(0, 100, 100):
            if x == 0:
                points_cloud_reference[i, 0] = y
                points_cloud_reference[i, 1] = 0
                points_cloud_reference[i, 2] = 0
            elif x == 1:
                points_cloud_reference[i, 0] = 0
                points_cloud_reference[i, 1] = y
                points_cloud_reference[i, 2] = 0
            elif x == 2:
                points_cloud_reference[i, 0] = 0
                points_cloud_reference[i, 1] = 0
                points_cloud_reference[i, 2] = y
            i = i + 1
    pcd_reference = o3d.geometry.PointCloud()
    pcd_reference.points = o3d.utility.Vector3dVector(points_cloud_reference)
    pcd_reference.transform([
        [1, 0, 0, 0],
        [0,-1, 0, 0],
        [0, 0,-1, 0],
        [0, 0, 0, 1]
        ]) # flip, otherwise the pcd will be upside down
    x_pcd = pcd_reference.select_by_index(list(range(0, 100, 1)))
    x_pcd.paint_uniform_color([1.0, 0.0, 0.0])
    y_pcd = pcd_reference.select_by_index(list(range(100, 200, 1)))
    y_pcd.paint_uniform_color([0.0, 1.0, 0.0])
    z_pcd = pcd_reference.select_by_index(list(range(200, 300, 1)))
    z_pcd.paint_uniform_color([0.0, 0.0, 1.0])
    return x_pcd, y_pcd, z_pcd

def plane_intersect(a, b):
    """
    a, b   4-tuples/lists
           Ax + By +Cz + D = 0
           A,B,C,D in order  

    output: 2 points on line of intersection, np.arrays, shape (3,)
    """
    a_vec, b_vec = np.array(a[:3]), np.array(b[:3])

    aXb_vec = np.cross(a_vec, b_vec)

    A = np.array([a_vec, b_vec, aXb_vec])
    d = np.array([-a[3], -b[3], 0.]).reshape(3,1)

    # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error
    p_inter = np.linalg.solve(A, d).T

    return p_inter[0], (p_inter + aXb_vec)[0]

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

    # Reference points
    x_pcd, y_pcd, z_pcd = reference_axis()
    
    # Find alignment line
    alignment_plane_points, alignment_plane_model = find_alignment(depth_image=depth, f=focal, scale=1000, index=index)
    alignment_pcd = o3d.geometry.PointCloud()
    alignment_pcd.points = o3d.utility.Vector3dVector(alignment_plane_points)
    alignment_pcd.paint_uniform_color([0.0, 0.0, 1.0])
    alignment_pcd.transform([
        [1, 0, 0, 0],
        [0,-1, 0, 0],
        [0, 0,-1, 0],
        [0, 0, 0, 1]
        ]) # flip, otherwise the pcd will be upside down
    o3d.visualization.draw_geometries([alignment_pcd, x_pcd, y_pcd, z_pcd])

    # Create points from depth by pinhole model
    points = create_point_cloud(depth_image=depth, f=focal, scale=1000, index=index)
    # Set point cloud color
    r = rgb[index][:, 0].flatten()
    g = rgb[index][:, 1].flatten()
    b = rgb[index][:, 2].flatten()
    myc = np.stack([r, g, b], axis=1)
    # PointCloud 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(myc)
    pcd.transform([
        [1, 0, 0, 0],
        [0,-1, 0, 0],
        [0, 0,-1, 0],
        [0, 0, 0, 1]
        ]) # flip, otherwise the pcd will be upside down
    o3d.visualization.draw_geometries([pcd, alignment_pcd, x_pcd, y_pcd, z_pcd])

    # Fit Plane
    [inlier_clouds, outlier_clouds], crowd_plane_model = find_plane(pcd)
    o3d.visualization.draw_geometries([inlier_clouds, outlier_clouds])
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default='./Images/test_resized.jpeg')
    parser.add_argument("--depth", type=str, default='./Images/test_depth.png')
    parser.add_argument("--mask", type=str, default='./Images/pred_test_resized.png')
    parser.add_argument("--focal", type=float, default=1324.0)
    args = parser.parse_args()
    main(args)