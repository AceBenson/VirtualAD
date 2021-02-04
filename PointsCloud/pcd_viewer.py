import open3d as o3d
import os 
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
from PIL import Image
from pprint import pprint

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
	
def main(args):
	rgb = read_file(args.image)
	depth = read_file(args.depth)

	depth = np.squeeze(depth)
	if np.amax(depth) > 1.0:
		depth = depth / np.amax(depth)

	h, w, _ = rgb.shape
	dh, dw = depth.shape
	# base on rgb points
	if dh != h or dw != w:
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

	# set camera intrinsic from focal length estimation
	# fx = 580.8723715937349
	# fy = 580.8723715937349

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
	print("\npoints_cloud: ")
	print(f"shape: {xyz.shape}")
	print(xyz)
	
	print('\n<-- start original point clouds render -->')
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

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='point clouds rendering')
	parser.add_argument('-c', "--image", type=str, default='../Images/wikicommons_field.jpg')
	parser.add_argument('-d', "--depth", type=str, default='../Images/wikicommons_field_inv_depth.png')
	args = parser.parse_args()
	main(args)