function d = epipolarLineDistance(F,p1,p2)

% Distance of point from epipolar line
% Calculates the distance of the homogeneous points p2 (3xM) from the epipolar lines
% due to homogeneous points p1 (3xN) where F (3x3) is a fundamental matrix relating the
% views containing image points p1 and p2. d (NxM) is the distance matrix
% where element d(i,j) is the distance from the point p2(j) to the epipolar
% line due to point p1(i).

% author: Peter Corke (2015)
% edited: Maximilian Enthoven (11/2016)


l = F*p1;
for ii = 1:size(p1,2)
    d(ii) = abs(l(1,ii)*p2(1,ii) + l(2,ii)*p2(2,ii) + l(3,ii)) ./...
                    sqrt(l(1,ii)^2 + l(2,ii)^2);
end

end

