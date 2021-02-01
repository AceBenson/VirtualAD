function [ M2 ] = findM2( K, F, pts1, pts2 )
%findM2: returns the second camera matrix

% inputs:
% K: intrinsics
% F: fundamental
% pts1, pts2: matching points in both images


E = K'*F*K;
M2s = camera2(E);
M1 = [eye(3), zeros(3,1)];
for i=1:4
   
    curr_M2 = M2s(:,:,i);
    [P, ~] = triang(K*M1, pts1, K*curr_M2, pts2);
    


    %check if points if positive z value
    P2 = M2s(:,:,i)*[P ones(size(P,1), 1)]';
    P2 = P2';
    if all(P(:,3) > 0) && all(P2(:,3) > 0)
        M2 = M2s(:,:,i);
        break
    end
    
end


% ptCloud = pointCloud(P);
% cameraSize = 0.1;
% figure; hold on; grid on;
% 
% plotCamera('Size', cameraSize, 'Color', 'r', 'Label', '1', 'Opacity', 0);
% plotCamera('Location', M2(:,4), 'Orientation', M2(:,1:3), 'Size', cameraSize, ...
%     'Color', 'b', 'Label', '2', 'Opacity', 0);

% % Visualize the point cloud
% pcshow(ptCloud, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
%     'MarkerSize', 45);



end

