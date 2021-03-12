import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d
from pprint import pprint
import cv2
from oct2py import octave
import os

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

def readFilesInFolder(folder):
    for fileName in os.listdir(folder):
        if fileName.find('depth') != -1:
            depthImg = read_file(os.path.join(folder, fileName))
        elif fileName.find('segment') != -1:
            segmentImg = read_file(os.path.join(folder, fileName))
        else:
            rgbImg = read_file(os.path.join(folder, fileName))
    return depthImg, segmentImg, rgbImg

class Images2D:
    def __init__(self, rgb_image, depth_image, segment_image, advertisement_image):
        self.rgb_image = rgb_image
        self.depth_image = depth_image
        self.segment_image = segment_image
        self.advertisement_image = advertisement_image
        self.imageHeight, self.imageWidth, _ = rgb_image.shape
        self.mask = self.getBinaryMaskBySegment()
        self.initImages()

    def initImages(self):
        self.normalizeDepth()
        self.resizeDepthAndSegment()

    def normalizeDepth(self):
        self.depth_image = np.squeeze(self.depth_image)
        if np.amax(self.depth_image) > 1.0:
            self.depth_image = self.depth_image / np.amax(self.depth_image)
    
    def resizeDepthAndSegment(self):
        dh, dw = self.depth_image.shape
        if dh != self.imageHeight or dw != self.imageWidth:
            dimg = Image.fromarray(self.depth_image)
            dimg_resized = dimg.resize((self.imageWidth, self.imageHeight), Image.BICUBIC)
            self.depth_image = np.asarray(dimg_resized)
        mh, mw, _ = self.segment_image.shape
        if mh != self.imageHeight or mw != self.imageWidth:
            mimg = Image.fromarray(self.segment_image)
            mimg_resized = mimg.resize((self.imageWidth, self.imageHeight), Image.BICUBIC)
            self.segment_image = np.asarray(mimg_resized)

    def getBinaryMaskBySegment(self):
        indexR = self.segment_image[:, :, 0] == 31/255
        indexG = self.segment_image[:, :, 1] == 255/255
        indexB = self.segment_image[:, :, 2] == 0/255
        return np.logical_and(indexR, indexG, indexB)

def reconstructPointsCloud(rgb_image, depth_image, focal, scale=1000):
    def initColor():
        r = rgb_image[:, :, 0].flatten()
        g = rgb_image[:, :, 1].flatten()
        b = rgb_image[:, :, 2].flatten()
        return np.stack([r, g, b], axis=1)
    def initPoints():
        shape = depth_image.shape
        rows = shape[0]
        cols = shape[1]
        cx, cy = cols / 2, rows / 2
        points = np.zeros((rows*cols, 3), np.float32)

        i = 0
        for r in range(0, rows):
            for c in range(0, cols):
                depth = depth_image[r, c]
                z = (1-depth) * scale
                points[i, 0] = cx + (c - cx) * (z / focal)
                points[i, 1] = cy + (r - cy) * (z / focal)
                points[i, 2] = z
                i = i + 1
        return points
    color = initColor()
    points = initPoints()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(color)
    pcd.transform([
        [1, 0, 0, 0],
        [0,-1, 0, 0],
        [0, 0,-1, 0],
        [0, 0, 0, 1]
        ])
    return pcd

def filterGrandStandPointsCloud(mainPCD, mask):
    return mainPCD.select_by_index( list(np.where(mask.flatten())[0]) )

def identifyAlignmentPlaneEquation(mask, depth_image, f, scale=1000):
    def findLines():
        binaryImg = (mask*255).astype(np.uint8)
        cannyImg = cv2.Canny(binaryImg, 50, 150)
        lines = cv2.HoughLines(cannyImg,1,np.pi/180,50)
        return lines
    def sample3dPoints(lines, rows, cols):
        line = lines[0]
        rho,theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a*rho, b*rho
        x1, y1 = 0, int(y0 + (x0 / b)*(a))
        x2, y2 = (cols-1), int(y0 + ((x0 - (cols-1)) / b)*(a))
        x0, y0 = int((x1 + x2) / 2), int((y1 + y2) / 2)
        x = [x0, x1, x2]
        y = [y0, y1, y2]
        return x, y
    def deriveEquation(rows, cols, x, y):
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
        return [a, b, c, d]

    shape = depth_image.shape
    rows = shape[0]
    cols = shape[1]
    lines = findLines()
    if lines is None:
        print("No line be found!")
    x, y = sample3dPoints(lines, rows, cols)
    equation = deriveEquation(rows, cols, x, y)
    return equation

def fitCrowdPlaneEquation(pcd, th):
    plane_model, inliers = pcd.segment_plane(distance_threshold=th, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model
    inlier_clouds = pcd.select_by_index(inliers)
    outlier_clouds = pcd.select_by_index(inliers, invert=True)
    return [a, -b, -c, d]

def findCorners(alignmentPlaneEquation, crowdPlaneEquation, x_max, x_min, AD_width, AD_Height, assetResize):
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm

    def planeIntersect(a, b):
        a_vec, b_vec = np.array(a[:3]), np.array(b[:3])
        aXb_vec = np.cross(a_vec, b_vec)

        A = np.array([a_vec, b_vec, aXb_vec])
        d = np.array([-a[3], -b[3], 0.]).reshape(3,1)
        p_inter = np.linalg.solve(A, d).T

        return p_inter[0], (p_inter - aXb_vec)[0]

    def findUV():
        n = crowdPlaneEquation[:3]
        v = p2-p1
        v = normalize(v)
        n = normalize(n)
        u = np.cross(n, v)
        return u, v
    def pivotPoint():
        x1 = (x_max + x_min) / 2
        t = (x1 - p1[0]) / (p2[0] - p1[0])
        y1 = p1[1] + (p2[1] - p1[1]) * t
        z1 = p1[2] + (p2[2] - p1[2]) * t
        return np.array([x1, y1, z1])

    p1, p2 = planeIntersect(alignmentPlaneEquation, crowdPlaneEquation)
    u, v = findUV()
    p = pivotPoint()
    
    p1 = p - v * int(AD_width/2*assetResize)
    p2 = p + v * int(AD_width/2*assetResize)
    p3 = p1 + u * int(AD_Height*assetResize)
    p4 = p2 + u * int(AD_Height*assetResize)
    return [p1, p2, p3, p4]

def putAsset(p, rgb, ad, f):
    def projectPoints():
        h, w, _ = rgb.shape
        cx = w / 2
        cy = h / 2
        pts_dst = np.zeros((4, 2))
        for i in range(4):
            c = (p[i][0] - cx) * f / p[i][2] + cx
            r = (p[i][1] - cy) * f / p[i][2] + cy
            pts_dst[i] = [c, r] # (x,y)
        return pts_dst
    h, w, _ = rgb.shape
    pts_dst = projectPoints()
    pts_src = np.array([ [0, ad.shape[0]-1], [ad.shape[1]-1, ad.shape[0]-1], [0, 0], [ad.shape[1]-1, 0] ])

    M, status = cv2.findHomography(pts_src, pts_dst)
    ad_homography = cv2.warpPerspective(ad, M, (w, h))

    index = ad_homography[:, :, 0] != 0
    rgb[index] = ad_homography[index]

    return rgb

def axisPointsCloud():
    points = np.zeros((100*3, 3), np.float32)
    i = 0
    for axis in range(3):
        for p in np.linspace(0, 100, 100):
            if axis == 0:
                points[i, 0] = p
                points[i, 1] = 0
                points[i, 2] = 0
            elif axis == 1:
                points[i, 0] = 0
                points[i, 1] = p
                points[i, 2] = 0
            elif axis == 2:
                points[i, 0] = 0
                points[i, 1] = 0
                points[i, 2] = p
            i = i + 1
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.transform([
        [1, 0, 0, 0],
        [0,-1, 0, 0],
        [0, 0,-1, 0],
        [0, 0, 0, 1]
        ])
    x_pcd = pcd.select_by_index(list(range(0, 100, 1)))
    x_pcd.paint_uniform_color([1.0, 0.0, 0.0])
    y_pcd = pcd.select_by_index(list(range(100, 200, 1)))
    y_pcd.paint_uniform_color([0.0, 1.0, 0.0])
    z_pcd = pcd.select_by_index(list(range(200, 300, 1)))
    z_pcd.paint_uniform_color([0.0, 0.0, 1.0])
    return x_pcd + y_pcd + z_pcd

def samplePlanePointsCloud(xBegin, zBegin, xEnd, zEnd, equation, color):
    a, b, c, d = equation
    points_cloud = np.zeros((100*100, 3), np.float32)
    i = 0
    for x in np.linspace(xBegin, xEnd, 100):
        for z in np.linspace(zBegin, zEnd, 100):
            y = -((d + a*x + c*z) / b)
            points_cloud[i, 0] = x
            points_cloud[i, 1] = y
            points_cloud[i, 2] = z
            i = i + 1
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_cloud)
    pcd.paint_uniform_color(color)
    pcd.transform([
        [1, 0, 0, 0],
        [0,-1, 0, 0],
        [0, 0,-1, 0],
        [0, 0, 0, 1]
        ])
    return pcd

def getUVPointsCloud(p):
    def sampleBetween2Points(p1, p2):
        points_cloud = zip(np.linspace(p1[0], p2[0], 100), np.linspace(p1[1], p2[1], 100), np.linspace(p1[2], p2[2], 100))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_cloud)
        pcd.paint_uniform_color([0.0, 0.0, 1.0])
        pcd.transform([
            [1, 0, 0, 0],
            [0,-1, 0, 0],
            [0, 0,-1, 0],
            [0, 0, 0, 1]
            ])
        return pcd
    bottom_pcd = sampleBetween2Points(p[0], p[1])
    left_pcd = sampleBetween2Points(p[0], p[2])
    top_pcd = sampleBetween2Points(p[3], p[1])
    right_pcd = sampleBetween2Points(p[3], p[2])
    return bottom_pcd + left_pcd + top_pcd + right_pcd

def estimateFocal(rgb):
    octave.addpath('./matlab_modules/')
    octave.eval('pkg load image')
    focal = octave.findFocal(rgb)
    return focal

def main(args):
    depthImg, segmentImg, rgbImg = readFilesInFolder(args.folder)
    images = Images2D(rgbImg, depthImg, segmentImg, read_file(args.ad))
    # focal = estimateFocal(images.rgb_image)
    focal = 1000.0
    mainPCD = reconstructPointsCloud(images.rgb_image, images.depth_image, focal)
    grandStandPCD = filterGrandStandPointsCloud(mainPCD, images.mask)

    alignmentPlaneEquation = identifyAlignmentPlaneEquation(images.mask, images.depth_image, focal)
    crowdPlaneEquation = fitCrowdPlaneEquation(grandStandPCD, images.imageWidth/10)

    points = np.asarray(grandStandPCD.points)
    corners = findCorners(alignmentPlaneEquation, crowdPlaneEquation, np.max(points[:, 0]), np.min(points[:, 0]), \
        images.advertisement_image.shape[1], images.advertisement_image.shape[0], args.assetResize)

    images.result = putAsset(corners, images.rgb_image, images.advertisement_image, focal)
    plt.imshow(images.result)
    plt.show()

    if args.showPointCloud:
        referencePCD = axisPointsCloud()
        alignmentPCD = samplePlanePointsCloud(0, 0, 1000, 1000, alignmentPlaneEquation, [0.0, 1.0, 0.0])
        crowdPCD = samplePlanePointsCloud(0, 0, 1000, 1000, crowdPlaneEquation, [1.0, 0.0, 0.0])
        uvPCD = getUVPointsCloud(corners)

        o3d.visualization.draw_geometries([grandStandPCD, referencePCD, alignmentPCD, crowdPCD, uvPCD])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default='./Images/test/')
    parser.add_argument("--ad", type=str, default='./Images/AD3.jpg')
    parser.add_argument("--assetResize", type=float, default=0.5)
    parser.add_argument("-showPointCloud", action='store_true')
    args = parser.parse_args()
    main(args)