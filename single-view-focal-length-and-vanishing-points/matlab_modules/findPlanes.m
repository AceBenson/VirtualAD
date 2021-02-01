function [ plane_pts, plane_H, all_match1, all_match2 ] = findPlanes( im1, im2, n )

% inputs:
% im1, im2: rgb images
% n: number of planes

% outputs:
% plane_pts: cell containing NX4 matched feature points for the plane 
% plane_H: cell containing H for each plane

n_planes = n;

im1 = rgb2gray(im1);
im2 = rgb2gray(im2);

surf1 = detectSURFFeatures(im1, 'MetricThreshold', 100);
surf2 = detectSURFFeatures(im2, 'MetricThreshold', 100);

[feat1, pts1] = extractFeatures(im1, surf1);
[feat2, pts2] = extractFeatures(im2, surf2);

index_pairs = matchFeatures(feat1, feat2);

match1 = pts1(index_pairs(:,1), :);
match2 = pts2(index_pairs(:,2), :);

match1_loc = match1.Location;
match2_loc = match2.Location;

% showMatchedFeatures(im1, im2, match1_loc, match2_loc);

match1_h = [match1_loc'; ones(1,size(match1_loc,1))];
match2_h = [match2_loc'; ones(1,size(match2_loc,1))];

area_thresh = 1000;
dist_thresh = 20;

plane_pts = {};
plane_H = {};

all_match1 = match1_loc;
all_match2 = match2_loc;

while 1
    
    if size(plane_pts,2) == n_planes
        break
    end
    
    randidx = randperm(size(match1_loc,1), 4);
    subs = match1_loc(randidx,:);
    
    % check if areas formed by triangles are ok
    a1 = polyarea(subs([2 3 4],1), subs([2 3 4],2));
    a2 = polyarea(subs([1 3 4],1), subs([1 3 4],2));
    a3 = polyarea(subs([1 2 4],1), subs([1 2 4],2));
    a4 = polyarea(subs([1 2 3],1), subs([1 2 3],2));

    if any([a1 a2 a3 a4] < area_thresh)
        continue
    end

    subs1 = match1_loc(randidx,:);
    subs2 = match2_loc(randidx,:);

    H21 = computeH(subs1', subs2');

    % check all points, find matches consistent with H
    tp1 = H21*match2_h;
    tp1 = [tp1(1,:) ./ tp1(3,:); tp1(2,:) ./ tp1(3,:)];

    dists = sqrt(sum((tp1 - match1_loc').^2));

    % number of pairs that satisfy the distance threshold
    compat = dists < 3;
    n_good = sum(compat);
	if n_good < 20
        continue
    end
    
    % now reestimate H iteratively
    err_thresh = 3;
    failed = 0;
    while 1
        if err_thresh <= 0.03
            break
        end
        refH = computeH(match1_loc(compat,:)', match2_loc(compat,:)');
        tp1 = refH*match2_h;
        tp1 = [tp1(1,:) ./ tp1(3,:); tp1(2,:) ./ tp1(3,:)];

        compat = (sqrt(sum((tp1 - match1_loc').^2)) < err_thresh);
        if sum(compat) < 8
            failed = 1;
            break
        end
        
        err_thresh = err_thresh * 0.8;
    end
    
    if failed == 1
        continue
    end
    
    %new_im1_ref = applyH(im2, refH);

    plane_pts_idx = (sqrt(sum((tp1 - match1_loc').^2)) < 1);
    
    figure;
    imshow(im1); hold on;
    scatter(match1_loc(plane_pts_idx,1), match1_loc(plane_pts_idx,2));
    
    tx = double(match1_loc(plane_pts_idx,1));
    ty = double(match1_loc(plane_pts_idx,2));
    k = convhull(tx, ty);
    plot(tx(k), ty(k));
    
    remove_pts = inpolygon(match1_loc(:,1), match1_loc(:,2), tx(k), ty(k));
%     figure; imshow(im1); hold on;
%     scatter(match1_loc(remove_pts,1), match1_loc(remove_pts,2));
    
    plane_pts = [plane_pts, [match1_loc(plane_pts_idx,:), match2_loc(plane_pts_idx,:)]];
    match1_loc = match1_loc(~remove_pts,:);
    match2_loc = match2_loc(~remove_pts,:);
    match1_h = [match1_loc'; ones(1,size(match1_loc,1))];
    match2_h = [match2_loc'; ones(1,size(match2_loc,1))];

    plane_H = [plane_H, refH];
end


end

