
clear;
close all;

imgs = cell(1,2);
imgs{1} = imread('images/1_001.jpg');
imgs{2} = imread('images/1_002.jpg');


n_planes = 2;
[plane_pts, plane_H, match1, match2] = findPlanes(imgs{1}, imgs{2}, n_planes);

vps = findVPs(imgs{1});
% compute K using 3 orthogonal vanishing points
v1 = vps(1,:); v2 = vps(2,:); v3 = vps(3,:); 
A = [v1(1)*v2(1)+v1(2)*v2(2), v1(3)*v2(1)+v1(1)*v2(3), v1(3)*v2(2) + v1(2)*v2(3); ...
     v1(1)*v3(1)+v1(2)*v3(2), v1(3)*v3(1)+v1(1)*v3(3), v1(3)*v3(2)+v1(2)*v3(3); ...
     v2(1)*v3(1)+v2(2)*v3(2), v2(3)*v3(1)+v2(1)*v3(3), v2(3)*v3(2)+v2(2)*v3(3)];

w = A \ [-1; -1; -1];
W = [w(1) 0 w(2); 0 w(1) w(3); w(2) w(3) 1];

K_inv = chol(W);
K = inv(K_inv);
K = K/K(3,3);

% get total matches from the planes
match1 = [];
match2 = [];
for i=1:n_planes
    pl_pts = plane_pts{i};
    match1 = [match1; pl_pts(:,1:2)];
    match2 = [match2; pl_pts(:,3:4)];
end

[F, inliers] = estimateFundamentalMatrix(match1,...
    match2,'Method','RANSAC',...
    'NumTrials', 2000,'DistanceThreshold',0.005);

%showMatchedFeatures(imgs{1}, imgs{2}, match1(inliers,:), match2(inliers,:),'montage');


M1 = [eye(3), zeros(3,1)];
M2 = findM2(K, F, match1(inliers,:), match2(inliers,:));


% get surface normals
normals = zeros(4, n_planes);
normals2 = zeros(3, n_planes);
%figure; hold on
for k=1:2
    pl_pts = plane_pts{k};
    pts1 = [pl_pts(:,1) pl_pts(:,2)];
    pts2 = [pl_pts(:,3) pl_pts(:,4)];
    [P, err] = triang(K*M1, pts1, K*M2, pts2);
    [~, ~, V] = svd([P, ones(size(P,1),1)]);
    
    normal = V(:,end);
    normals(:,k) = normal;
    
   % pcshow(P, 'MarkerSize', 70);
end



% reconstruct
figure; hold on
for i=1:2
    pl_pts = plane_pts{i};
    tx = double(pl_pts(:,1));
    ty = double(pl_pts(:,2));
    k = convhull(tx, ty);
    bw = poly2mask(tx(k), ty(k), size(imgs{1},1), size(imgs{1},2));

    %apply mask
    cImg = im2double(imgs{1});
    for c=1:3
        cImg(:,:,c) = cImg(:,:,c) .* bw;
    end

    % get non-black pixels
    [r, c, l] = find(cImg(:,:,1) > 0);
    
    n = normals(1:3,i);
     
    % backproject
    pix_coords = [c'; r'; ones(1,size(r,1))];
    world_coords = inv(K)*pix_coords;
    % w is scale factor
    w = (-n'*world_coords)/normals(4,i);
    new_coords = bsxfun(@rdivide, world_coords, w);
    xx = new_coords(1,:);
    yy = new_coords(2,:);
    zz = new_coords(3,:);
    
    % plot the surface
    pixcolors = impixel(cImg, c, r);
    pcshow([xx(:), yy(:), zz(:)], pixcolors, ...
       'VerticalAxis', 'y', 'VerticalAxisDir', 'down');
end




