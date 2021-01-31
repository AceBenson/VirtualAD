f = imread('fj1.jpg');
line = getLines(rgb2gray(f),20);
vp = getVP3(line,size(f,1),size(f,2));
draw(f,vp,zeros(3),line);