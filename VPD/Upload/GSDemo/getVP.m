function [vp, foc time] = getVP(f,line)
f = rgb2gray(f);

tic;
used = zeros(size(line,1),1);
vp_theta = zeros(3,1);
vp_phi = zeros(3,1);
vp_radius = zeros(3,1);
vp_weight = zeros(3,1);

foc0 = min(size(f));
for i = 1:3
    point = getIntersections(line(find(~used),:));
    point(:,1) = point(:,1) - size(f,2) / 2;
    point(:,2) = point(:,2) - size(f,1) / 2;
    radius = sqrt(point(:,1).*point(:,1) + point(:,2).*point(:,2));
    theta = atan2(point(:,2),point(:,1));
    phi = atan2(radius,foc0 * ones(length(radius),1));
    [vp_theta(i) vp_phi(i) vp_weight(i)] = findMax(theta,phi);

    vp_radius = foc0*tan(vp_phi);
    vp = [vp_radius.*cos(vp_theta) vp_radius.*sin(vp_theta) ones(3,1)];

    
    ix = findVLine(line,vp_theta(i),vp_radius(i),size(f,1),size(f,2));
    used(ix) = 1;
end
vp_radius = foc0*tan(vp_phi);
vp = [vp_radius.*cos(vp_theta) vp_radius.*sin(vp_theta) ones(3,1)];
foc = 0;
time = toc
vp_theta = atan2(vp(:,2),vp(:,1));
vp_radius = sqrt(vp(:,1).^2 + vp(:,2).^2);
end