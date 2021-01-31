function [vp_theta vp_phi vp_weight] = findMax(theta,phi)
w = 0.01;
theta = theta + pi;
phi = phi;
T = 0:w:2*pi + w;
P = 0:w:pi/2 + w;
mat = zeros(length(T),length(P));

for i = 2:length(theta)
    mat(ceil(theta(i)/w + eps),ceil(phi(i)/w + eps)) = mat(ceil(theta(i)/w + eps),ceil(phi(i)/w + eps)) + 1;
end
mat = imfilter(mat,fspecial('disk',5));
[v,ix] = max(mat(:));
r = mod((ix - 1),size(mat,1)) + 1;
c = floor((ix - 1)/size(mat,1)) + 1;

vp_theta = T(r) - pi;
vp_phi = P(c) - 0;
vp_weight = v;
end