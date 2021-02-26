import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    
def drawLine(img, rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    #把直线显示在图片上
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imshow('img', img)
    cv2.waitKey(0)

def findAlignmentLine(segImg):
    index0 = segImg[:, :, 0] == 31/255
    index1 = segImg[:, :, 1] == 255/255
    index2 = segImg[:, :, 2] == 0/255
    index = np.logical_and(index0, index1, index2)

    binaryImg = (index*255).astype(np.uint8)
    cannyImg = cv2.Canny(binaryImg, 50, 150)

    lines = cv2.HoughLines(cannyImg,1,np.pi/180,50)
    if lines is not None:
        line = lines[0]
        rho,theta = line[0]
        
        return rho, theta
    else:
        return None

def main():
    segImg = read_file('../Images/pred_test_resized.png')
    rho, theta = findAlignmentLine(segImg)
    drawLine(segImg, rho, theta)

if __name__ == '__main__':
    main()