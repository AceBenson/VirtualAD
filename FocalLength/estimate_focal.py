import numpy as np
from oct2py import octave
import matplotlib.pyplot as plt

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

def estimate_focal(rgb):
    focal = octave.findFocal(rgb)
    print(f"Focal Length: {focal}")
    return focal

def main():
    rgb = read_file('../Images/video16.png')
    estimate_focal(rgb)

if __name__ == '__main__':
    main()