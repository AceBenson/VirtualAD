f = imread('fj1.jpg');
    line = getLines(rgb2gray(f),20);
    tic
    [v2, sigma, p, hpos] = getVp(line, size(f),0);
    vp = v2;
    vp(:,1:2) = v2(:,2:-1:1) - repmat([320 240],size(v2,1),1);
    time = toc
    vp0 = vp;
    draw(f,vp0,zeros(3),line);
