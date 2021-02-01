function [ P, error ] = triang( M1, p1, M2, p2 )
% triangulate:
%       M1 - 3x4 Camera Matrix 1
%       p1 - Nx2 set of points
%       M2 - 3x4 Camera Matrix 2
%       p2 - Nx2 set of points

N = size(p1, 1);

P = [];
error = 0;


for i=1:N
    
   A  = [p1(i,1) * M1(3,:) - M1(1,:);
         p1(i,2) * M1(3,:) - M1(2,:);
         p2(i,1) * M2(3,:) - M2(1,:);
         p2(i,2) * M2(3,:) - M2(2,:)];
   
   [~, ~, V] = svd(A);
   
   %least squares triangulated 3D point
   X = V(:,end)';

   %unscale
   PP = X ./ X(4);
   
   %accumulate reprojection error
   proj1 = M1 * PP';
   proj2 = M2 * PP';
   
   %unscale
   proj1 = proj1(1:2) ./ proj1(3);
   proj2 = proj2(1:2) ./ proj2(3);
   
   error = error + (sum((proj1' - p1(i,:)).^2) + sum((proj2' - p2(i,:)).^2));
   
   if (sum((proj1' - p1(i,:)).^2) + sum((proj2' - p2(i,:)).^2)) < 1000
     P = [P; PP(1:3)];
   end
     
end



end

