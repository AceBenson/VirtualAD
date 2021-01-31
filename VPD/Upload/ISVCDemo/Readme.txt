This is a demo for "Simultaneous Vanishing Point Detection and Camera Calibration from Single Images".

getLine.m is a Matlab m-file that detects line segments in a given image.
demo_final.m is a Matlab m-file that the algortithm proposed in our paper. 

This demo is implemented in Matlab 2009b, Windows 7. 
Put these two m-files in one folder and open demo_final.m in Matlab. Press F5 to run the demo.
Some images are provided with this demo. To test different images, change the file path in the imread() function in Line 5. 
	f = imread('<file path>');

Notice that our algorithm is based on the constraints provided by three orthogonal vanishing points. So there should be three significant orthogonal vanishing points in your test image. 

May 5th, 2010