import cv2
import numpy as np

def main():
    test_resized = cv2.imread('../Images/test_resized.jpeg')
    img = cv2.imread('../Images/AD3.jpg')

    h, w, _ = img.shape
    # [w, h]
    pts_src = np.array([ [0, h-1], [w-1, h-1], [0, 0], [w-1, 0] ])
    pts_dst = np.array([[198.96515467, 298.99967657],
       [628.62351448, 298.99966469],
       [218.7680448 , 118.61319268],
       [556.10976095, 122.40038052]])

    M, status = cv2.findHomography(pts_src, pts_dst)

    img_out = cv2.warpPerspective(img, M, (test_resized.shape[1], test_resized.shape[0]))

    # img_out = cv2.line(img_out, tuple(pts_dst[0]),tuple(pts_dst[1]), (255, 0, 0), 2)
    # img_out = cv2.line(img_out, tuple(pts_dst[0]),tuple(pts_dst[3]), (255, 0, 0), 2)
    # img_out = cv2.line(img_out, tuple(pts_dst[2]),tuple(pts_dst[1]), (255, 0, 0), 2)
    # img_out = cv2.line(img_out, tuple(pts_dst[2]),tuple(pts_dst[3]), (255, 0, 0), 2)

    index = img_out[:, :, 0] != 0
    test_resized[index] = img_out[index]

    cv2.imshow('img', img)
    cv2.imshow('test', test_resized)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
