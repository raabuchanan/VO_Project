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
harris_kappa = 0.06;
num_keypoints = 1000;
nonmaximum_supression_radius = 5;
descriptor_radius = 9;
match_lambda = 2;
triangulationTolerance = 1;


p3pIterations = 2000;
p3pTolerance = 3;
p3pSample = 3;
triangulationIterations = 2000;
initializationIterations = 2000;

triangulationSample = 10;

dataBaseSize = 2;

max_dif = [ 0; 0; 0];
min_dif = [0; 0; 0];
triangulationRansac = false;




