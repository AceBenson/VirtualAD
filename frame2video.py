import os
import cv2
import numpy as np
from os.path import isfile, join

pathIn = './OutputVideoFrames/resnet101_upernet/'
pathOut = 'video25_resnet101_upernet.mp4'

files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
files.sort(key = lambda x: int(x[8:-4]))
print(files)

frames = []

for fileName in files:
    name = join(pathIn, fileName)
    img = cv2.imread(name)
    frames.append(img)
    height, width, _ = img.shape

out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (width, height))
for img in frames:
    out.write(img)
out.release()
