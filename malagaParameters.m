global harris_patch_size;
global harris_kappa;
global num_keypoints;
global nonmaximum_supression_radius;
global descriptor_radius;
global match_lambda;
global triangulationTolerance;
global p3pIterations;
global p3pTolerance;
global p3pSample;
global triangulationIterations;
global initializationIterations;
global triangulationSample;
global dataBaseSize;
global max_dif;
global min_dif;
global triangulationRansac;

% Size of Harris search patch
harris_patch_size = 9;
% Magic number in range (0.04 to 0.15)
harris_kappa = 0.05;
% Number of harris corners to find
num_keypoints = 2000;
% Size of patch to suppress around keypoint
nonmaximum_supression_radius = 5;
% Size of harris descriptor
descriptor_radius = 9;
% Matching parameter for harris corners, 
% a higher number means more matches of lower quality
match_lambda = 5;
% Pixel distance from epipolar line that is acceptable for
% a newly triangulated landmark
triangulationTolerance = 1;
% Number of iterations to perform RANSAC for p3p pose estimation
p3pIterations = 2000;
% Pixel margin for p3p RANSAC
p3pTolerance = 3;
% Number of sample points for p3p RANSAC
p3pSample = 3;
% Number of iterations to perform RANSAC for triangulating new points
triangulationIterations = 2000;
% Number of iterations to perform RANSAC for initialization
initializationIterations = 2000;
% Number of sample points for triangulation
triangulationSample = 10;
% How many past frames to save for triangulation
dataBaseSize = 3;
% Boundary box around camera where newly triangulated points
% are considered too close and are rejected
max_dif = [ 0.5; 0.5; 0.5];
min_dif = [-0.5; -0.5; -0.5];
% Whether or not to use ransac for triangulating new landmarks
triangulationRansac = false;

