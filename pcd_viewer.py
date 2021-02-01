import open3d as o3d
import os 
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
from PIL import Image
from pprint import pprint

depthMax = 1000

def read_file(fname):
	if fname.endswith('.npy'):
		return np.load(fname)
	elif fname.endswith('.jpg'):
		img = plt.imread(fname)
		img = img.astype(np.float32) / 255.0
		# try:
		# 	img2 = Image.open(fname)
		# except IOError:
		# 	print('fail to load image!')
		# img2 = np.asarray(img2).astype(np.float32) / 255.0
		# print(f'image shape: {img.shape}')
		# print(f'image data type: {img.dtype}')
		# print(img == img2)
		# plt.imshow(img)
		# plt.show()
		return img			
	elif fname.endswith('.png'):
		img = plt.imread(fname)
		# try:
		# 	img2 = Image.open(fname)
		# 	print(f'depth image data type: {np.array(img2).dtype}')
		# except IOError:
		# 	print('fail to load image!')
		# img2 = np.asarray(img2).astype(np.float32) / 255.0
		# print(f'depth image shape: {img.shape}')
		# print(f'depth image data type: {img.dtype}')
		# print(img == img2)
		# plt.imshow(img)
		# plt.show()
		return img

def find_plane(pcd):
	print('<-- start plane fitting by pcd -->')
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
	rgb = read_file(args.image)
	depth = read_file(args.depth)
	print(f'[before] depth shape: {depth.shape}, rgb shape: {rgb.shape}')
	depth = np.squeeze(depth)
	pprint(depth)
	if np.amax(depth) > 1.0:
		print('normalize to range [0,1]')
		depth = depth / np.amax(depth)
	print(np.amin(depth))
	print(np.amax(depth))
	# assert False

	# if len(rgb.shape) != 3:
	# 	rgb = rgb[:, :, :3]
	h, w, _ = rgb.shape

	# if len(depth.shape) != 2:
	# 	depth = depth[:, :, 0]
	assert len(depth.shape) == 2, 'depth shape must be H*W'
	dh, dw = depth.shape
	print(f'[after] depth shape: {depth.shape}, rgb shape: {rgb.shape}')
	# assert False

	# base on rgb points
	if dh != h or dw != w:
		print('depth must be reshaped to rgb')
		dimg = Image.fromarray(depth)
		dimg_resized = dimg.resize((w, h), Image.BICUBIC)
		depth = np.asarray(dimg_resized)

	xy = np.stack(np.meshgrid(range(w), range(h)), axis=-1)
	xy = xy.astype(np.float64)

	# set camera intrinsic from fov
	fov = 90.0 # +- 45 degree
	cx, cy = w / 2, h / 2

	hfov = fov / 360.0 * 2.0 * np.pi
	fx = w / (2.0 * np.tan(hfov / 2.0))

	vfov = 2. * np.arctan(np.tan(hfov / 2) * h / w)
	fy = h / (2.0 * np.tan(vfov / 2.0))

	for v in range(h):
		for u in range(w):
			xy[v, u, 0] = (u - cx) * (1.0 - depth[v, u]) / fx
			xy[v, u, 1] = (v - cy) * (1.0 - depth[v, u]) / fy

	r = rgb[:, :, 0].flatten()
	g = rgb[:, :, 1].flatten()
	b = rgb[:, :, 2].flatten()
	x = xy[:, :, 0].flatten()
	y = xy[:, :, 1].flatten()
	z = (1.0 - depth).flatten()

	myc = np.stack([r, g, b], axis=1)
	xyz = np.stack([x, y, z], axis=1)
	# assert False
	print('<-- start original point clouds render -->')
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(xyz) # float64 to open3d format
	pcd.colors = o3d.utility.Vector3dVector(myc)
	pcd.transform([
		[1, 0, 0, 0],
		[0,-1, 0, 0],
		[0, 0,-1, 0],
		[0, 0, 0, 1]
		]) # flip, otherwise the pcd will be upside down
	o3d.visualization.draw_geometries([pcd])
	vis_pcd = find_plane(pcd)
	o3d.visualization.draw_geometries(vis_pcd)

	# while pcd.has_points() and len(pcd.points) > 20000:
	# 	inliers_pcd, outliers_pcd = vis_pcd
	# 	dists = pcd.compute_point_cloud_distance(inliers_pcd)
	# 	dists = np.asarray(dists)
	# 	ind = np.where(dists > 0.01)[0]
	# 	remain_pcd = pcd.select_by_index(ind)
	# 	o3d.visualization.draw_geometries([remain_pcd])
	# 	vis_pcd = find_plane(remain_pcd)
	# 	pcd = remain_pcd
	# 	o3d.visualization.draw_geometries(vis_pcd)
	
	# print()
	# print('<-- start to create convex hull by inliers pcd -->')
	# hull, _ = inlier_clouds.compute_convex_hull()
	# hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
	# hull_ls.paint_uniform_color([0.0, 0.0, 1.0])
	# o3d.visualization.draw_geometries([inlier_clouds, hull_ls])

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='point clouds rendering')
	parser.add_argument('-c', "--image", type=str, default='')
	parser.add_argument('-d', "--depth", type=str, default='')
	args = parser.parse_args()
	main(args)