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

def getFullPointsCloud(rgb_image, depth_image, focal, scale=1000):
    def initColor(rgb_image):
        r = rgb_image[:, :, 0].flatten()
        g = rgb_image[:, :, 1].flatten()
        b = rgb_image[:, :, 2].flatten()
        return np.stack([r, g, b], axis=1)
    def initPoints(depth_image, focal, scale):
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
    color = initColor(rgb_image)
    points = initPoints(depth_image, focal, scale)
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

def getGrandStandPointsCloud(mainPCD, mask):
    return mainPCD.select_by_index( list(np.where(mask.flatten())[0]) )

def getReferencePointsCloud():
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

def getAlignmentPlaneEquation(mask, depth_image, f, scale=1000):
    def findLines(mask):
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
    def deriveEquation(depth_image, rows, cols, x, y):
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
    lines = findLines(mask)
    if lines is None:
        print("No line be found!")
    x, y = sample3dPoints(lines, rows, cols)
    equation = deriveEquation(depth_image, rows, cols, x, y)
    return equation

def getCrowdPlaneEquation(pcd):
    plane_model, inliers = pcd.segment_plane(distance_threshold=10, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model
    inlier_clouds = pcd.select_by_index(inliers)
    outlier_clouds = pcd.select_by_index(inliers, invert=True)
    return [a, -b, -c, d]

def getPlanePointsCloud(x1, z1, x2, z2, equation, color):
    a, b, c, d = equation
    points_cloud = np.zeros((100*100, 3), np.float32)
    i = 0
    for x in np.linspace(x1, x2, 100):
        for z in np.linspace(z1, z2, 100):
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
    points_cloud = zip(np.linspace(p[0][0], p[1][0], 100), np.linspace(p[0][1], p[1][1], 100), np.linspace(p[0][2], p[1][2], 100))
    v_pcd = o3d.geometry.PointCloud()
    v_pcd.points = o3d.utility.Vector3dVector(points_cloud)
    v_pcd.paint_uniform_color([0.0, 0.0, 1.0])
    v_pcd.transform([
        [1, 0, 0, 0],
        [0,-1, 0, 0],
        [0, 0,-1, 0],
        [0, 0, 0, 1]
        ])
    points_cloud = zip(np.linspace(p[0][0], p[2][0], 100), np.linspace(p[0][1], p[2][1], 100), np.linspace(p[0][2], p[2][2], 100))
    u_pcd = o3d.geometry.PointCloud()
    u_pcd.points = o3d.utility.Vector3dVector(points_cloud)
    u_pcd.paint_uniform_color([0.0, 1.0, 1.0])
    u_pcd.transform([
        [1, 0, 0, 0],
        [0,-1, 0, 0],
        [0, 0,-1, 0],
        [0, 0, 0, 1]
        ])
    return v_pcd + u_pcd

def get4Corners(alignmentPlaneEquation, crowdPlaneEquation, x_max, x_min, AD_width, AD_Height):
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

    p1, p2 = planeIntersect(alignmentPlaneEquation, crowdPlaneEquation)

    n = crowdPlaneEquation[:3]
    v = p2-p1
    v = normalize(v)
    n = normalize(n)
    u = np.cross(n, v)

    x1 = (x_max + x_min) / 2
    t = (x1 - p1[0]) / (p2[0] - p1[0])
    y1 = p1[1] + (p2[1] - p1[1]) * t
    z1 = p1[2] + (p2[2] - p1[2]) * t

    p = np.array([x1, y1, z1])
    p1 = p - v * int(AD_width/2/2)
    p2 = p + v * int(AD_width/2/2)
    p3 = p1 + u * int(AD_Height/2)
    p4 = p2 + u * int(AD_Height/2)
    return [p1, p2, p3, p4]

def putAsset(p, rgb, ad, f):
    h, w, _ = rgb.shape
    cx = w / 2
    cy = h / 2
    pts_dst = np.zeros((4, 2))
    for i in range(4):
        c = (p[i][0] - cx) * f / p[i][2] + cx
        r = (p[i][1] - cy) * f / p[i][2] + cy
        pts_dst[i] = [c, r] # (x,y)

    pts_src = np.array([ [0, ad.shape[0]-1], [ad.shape[1]-1, ad.shape[0]-1], [0, 0], [ad.shape[1]-1, 0] ])

    M, status = cv2.findHomography(pts_src, pts_dst)
    ad_homography = cv2.warpPerspective(ad, M, (w, h))

    index = ad_homography[:, :, 0] != 0
    rgb[index] = ad_homography[index]

    return rgb

def main(args):
    images = Images2D(read_file(args.image), read_file(args.depth), read_file(args.mask), read_file(args.ad))
    mainPCD = getFullPointsCloud(images.rgb_image, images.depth_image, args.focal)
    grandStandPCD = getGrandStandPointsCloud(mainPCD, images.mask)
    referencePCD = getReferencePointsCloud()
    alignmentPlaneEquation = getAlignmentPlaneEquation(images.mask, images.depth_image, args.focal)
    crowdPlaneEquation = getCrowdPlaneEquation(grandStandPCD)

    points = np.asarray(grandStandPCD.points)

    # TODO
    # Set boundary by image size
    alignmentPCD = getPlanePointsCloud(0, 0, 1000, 1000, alignmentPlaneEquation, [0.0, 1.0, 0.0])
    crowdPCD = getPlanePointsCloud(0, 0, 1000, 1000, crowdPlaneEquation, [1.0, 0.0, 0.0])

    corners = get4Corners(alignmentPlaneEquation, crowdPlaneEquation, np.max(points[:, 0]), np.min(points[:, 0]), \
        images.advertisement_image.shape[1], images.advertisement_image.shape[0])
    uvPCD = getUVPointsCloud(corners)

    o3d.visualization.draw_geometries([grandStandPCD, referencePCD, alignmentPCD, crowdPCD, uvPCD])

    result = putAsset(corners, images.rgb_image, images.advertisement_image, args.focal)
    plt.imshow(result)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default='./Images/test_resized.jpeg')
    parser.add_argument("--depth", type=str, default='./Images/test_depth.png')
    parser.add_argument("--mask", type=str, default='./Images/pred_test_resized.png')
    parser.add_argument("--ad", type=str, default='./Images/AD3.jpg')
    parser.add_argument("--focal", type=float, default=1324.0)
    args = parser.parse_args()
    main(args)