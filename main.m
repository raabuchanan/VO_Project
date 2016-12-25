%%%%%%%%%% Mini Project: Vision Algorithms for Mobile Robotics %%%%%%%%%%%
% by Alex Lieber, Carl Str�mbeck, Russell Buchanan, Maximilian Enthoven
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

kitti_path = 'kitti';
malaga_path = 'malaga-urban-dataset-extract-07/';
parking_path = 'parking';

% Necessary paths
addpath(genpath('all_solns'))
addpath(genpath('src'))
addpath(genpath('testdata')) % here not necessary

if dataset == 0
    
    assert(exist(kitti_path) ~= 0);
    ground_truth = load([kitti_path '/poses/00.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    last_frame = 4540;
    K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
elseif dataset == 1
    % Path containing the many files of Malaga 7.
    assert(exist(malaga_path) ~= 0);
    images = dir([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    last_frame = length(left_images);
    K = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
elseif dataset == 2
    % Path containing images, depths and all...
    %assert(exist(parking_path) ~= 0);
    last_frame = 598;
    K = load([parking_path '/K.txt']);
     
    ground_truth = load([parking_path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
else
    assert(false);
end

%% bootstrap / initialization of keypoint matching between adjacent frames

bootstrap_frames = [1 3]; % first and third frame

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

 fprintf('\n\nProcessing frame %d\n=====================\n', 1);
[prevState,firstLandmarks] = monocular_intialization(img0,img1,ransac,K);
prevImage = img1;

figure;
%subplot(1, 3, 3); %uncomment to display images
scatter3(firstLandmarks(1, :), firstLandmarks(2, :), firstLandmarks(3, :), 3,'b');
set(gcf, 'GraphicsSmoothing', 'on');
view(0,0);
axis equal;
axis vis3d;
axis([-15 10 -10 5 -1 40]);


%% Continuous operation
%range = (bootstrap_frames(2)+1):last_frame;
range = 1:last_frame;
dataBase = [];
for i = 2:last_frame
    fprintf('\n\nProcessing frame %d\n=====================\n', i);
    if dataset == 0
        currImage = imread([kitti_path '/00/image_0/' sprintf('%06d.png',i)]);
        
    elseif dataset == 1
        currImage = rgb2gray(imread([malaga_path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(i).name]));        
    elseif dataset == 2
        currImage = im2uint8(rgb2gray(imread([parking_path ...
            sprintf('/images/img_%05d.png',i)])));
        [currState, currPose, dataBase] = processFrame(prevState, prevImage, currImage, K, dataBase);
    else
        assert(false);
    end
    
    R_C_W = currPose(:,1:3)
    t_C_W = currPose(:,4)

    % Distinguish success from failure.
    if (numel(R_C_W) > 0)
        plotCoordinateFrame(R_C_W', -R_C_W'*t_C_W, 2,['r';'r';'r']);
        hold on
       
%         % Ground truth
%         truePose = reshape(groundTruth(i+1, :), 4, 3)';
%         rot = truePose(1:3,1:3);
%         trans = truePose(:,4);
%         plotCoordinateFrame(rot', rot*trans, 2,['b';'b';'b']);
%         hold on

        scatter3(currState(3, :), currState(4, :), currState(5, :), 5,'r','filled');
        view(0,0);
        hold off
        
%         disp(['Rotation '])
%         rotation_error = abs(R_C_W - rot)
%         disp(['Translation '])
%         translate_error = abs(t_C_W + trans)
        
    else
        disp(['Frame ' num2str(i) ' failed to localize!']);
    end

    prevState = currState;
    prevImage = currImage;
    
    % Makes sure that plots refresh.    
    pause(0.01);
    
end
