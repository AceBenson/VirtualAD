function [pos, tilt, conf, y, py] = vp2horizon(v, vars, p, imsize)



ind = find([v(:, 2) > -0.25 & v(:, 2) < 1.25]);

varthresh = 0.005;
ind(find(vars(ind) > varthresh)) = [];

memberthresh = 6;
nmembers = sum(p(:, ind), 1);
ind(find(nmembers < memberthresh)) = [];

conf = length(ind);

v = v(ind, :);
vars = vars(ind);

if length(ind)==0
    pos = NaN;
    tilt = NaN;
    conf = NaN;
elseif length(ind)>0

    v(:, 1) = v(:, 1) - 0.5*imsize(2)/imsize(1);
    v(:, 2) = v(:, 2) - 0.5;
    for i = 1:length(ind)
        vars(i) = vars(i)*v(i,2)^2 / (v(i,1)^2+v(i,2)^2);    
    end    
    pos = sum(v(:, 2) ./ vars') / sum(1./vars');

    if nargout > 3
        y = [pos-0.25:0.005:pos+0.25];
        py = zeros(size(y));
        for i = 1:size(v, 1)
            py = py + log(1 / sqrt(2*pi*vars(i))*exp(-1/2/vars(i)*(y-v(i,2)).^2));
        end
        py = exp(py);
        py = py / sum(py);

        y = y + 0.5;

        ind = find(py> (1E-4)/length(py));
        y = y(ind);
        py = py(ind);
        py = py / sum(py);
    end
    
    pos = pos + 0.5;
end



