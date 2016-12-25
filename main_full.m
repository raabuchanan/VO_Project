%%%%%%%%%% Mini Project: Vision Algorithms for Mobile Robotics %%%%%%%%%%%
% by Alex Lieber, Carl Strömbeck, Russell Buchanan, Maximilian Enthoven
% ETH Zurich / UZH, HS 2016

%% Clearing workspaces, Closing windows & Clearing commandwindow
clear all;
close all;
clc;
%% initialize variables

dataset = 2; % 0: KITTI, 1: Malaga, 2: parking

% Tuning Parameters

global K_parking; % declared when setting up paths
global K_malaga;
global K_kitti;
global harris_patch_size;
global harris_kappa;
global num_keypoints;
global nonmaximum_supression_radius;
global descriptor_radius;
global match_lambda;
global use_p3p;
global pose_dist_threshold;
global num_KLT_patches;
global random_KLT_patches;

% Pose Estimation
use_p3p = true;

% Harris Corner Detector Parameters
% Randomly chosen parameters that seem to work well
harris_patch_size = 9;
harris_kappa = 0.07; % Magic number in range (0.04 to 0.15)
num_keypoints = 800;
nonmaximum_supression_radius = 6;
descriptor_radius = 9;
match_lambda = 5;
pose_dist_threshold = 0.10; % 10% used by Google Tango
num_KLT_patches = 50; %number evenly spaced KLT patches
random_KLT_patches = false;


%% set up relevant paths

kitti_path = '/kitti';
malaga_path = '/malaga-urban-dataset-extract-07/';
parking_path = '/parking';

% Necessary paths
addpath(genpath('all_solns'))
addpath(genpath('src'))
addpath(genpath('testdata')) % here not necessary

if dataset == 0
    
    assert(exist(kitti_path) ~= 0);
    ground_truth = load([kitti_path '/poses/00.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    last_frame = 4540;
    K_kitti = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
elseif dataset == 1
    % Path containing the many files of Malaga 7.
    assert(exist(malaga_path) ~= 0);
    images = dir([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    last_frame = length(left_images);
    K_malaga = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
elseif dataset == 2
    % Path containing images, depths and all...
    %assert(exist(parking_path) ~= 0);
    last_frame = 598;
    K_parking = load([parking_path '/K.txt']);
     
    ground_truth = load([parking_path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
else
    assert(false);
end

%% bootstrap / initialization of keypoint matching between adjacent frames

bootstrap_frames = [1 3]; % first and  frame

if dataset == 0
    img0 = imread([kitti_path '/00/image_0/' ...
        sprintf('%06d.png',bootstrap_frames(1))]);
    img1 = imread([kitti_path '/00/image_0/' ...
        sprintf('%06d.png',bootstrap_frames(2))]);
elseif dataset == 1
    img0 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(1)).name]));
    img1 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(2)).name]));
elseif dataset == 2
    img0 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png',bootstrap_frames(1))]));
    img1 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png',bootstrap_frames(2))]));
else
    assert(false);
end

% 1 for ransac 'yes'
ransac = 1;
monocular_initialization_carl;


% %% Continuous operation
% range = (bootstrap_frames(2)+1):last_frame;
% for i = range
%     fprintf('\n\nProcessing frame %d\n=====================\n', i);
%     if i == 1
%         [firstState,firstLandmarks] = monocular_intialization(img0,img1,1,dataset);
%         break
%     end
%     if dataset == 0
%      
%         prevImage = imread([kitti_path '/00/image_0/' sprintf('%06d.png',i)]);
%         currImage = imread([kitti_path '/00/image_0/' sprintf('%06d.png',i+1)]);
%         % code goes here
%         
%         
%     elseif dataset == 1
%         prevImage = rgb2gray(imread([malaga_path ...
%             '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
%             left_images(i).name]));
%         currImage = rgb2gray(imread([malaga_path ...
%             '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
%             left_images(i+1).name]));
%         % code goes here
%         
%         
%     elseif dataset == 2
%         prevImage = im2uint8(rgb2gray(imread([parking_path ...
%             sprintf('/images/img_%05d.png',i)])));
%         currImage = 
%         
%         % code goes here
%         
%         
%     else
%         assert(false);
%     end
%     % Makes sure that plots refresh.    
%     pause(0.01);
%     
%     prev_img = image;
% end
