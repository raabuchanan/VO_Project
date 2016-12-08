% Tuning Parameters


global harris_patch_size;
global harris_kappa;
global num_keypoints;
global nonmaximum_supression_radius;
global descriptor_radius;
global match_lambda;
global use_p3p;
global pose_dist_threshold;


%% Pose Estimation
use_p3p = true;

%% Harris Corner Detector Parameters
% Randomly chosen parameters that seem to work well
harris_patch_size = 9;
harris_kappa = 0.07; % Magic number in range (0.04 to 0.15)
num_keypoints = 2000;
nonmaximum_supression_radius = 8;
descriptor_radius = 9;
match_lambda = 6;
pose_dist_threshold = 0.10; % 10% used by Google Tango

