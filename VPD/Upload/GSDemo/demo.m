    f = imread('fj1.jpg');
    line = getLines(rgb2gray(f),20);
    [vp foc time] = getVP(f,line);
    vp0 = vp;
    draw(f,vp0,zeros(3),line);
    