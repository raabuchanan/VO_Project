%%%%%%%%%% Mini Project: Vision Algorithms for Mobile Robotics %%%%%%%%%%%
% by Alex Lieber, Carl Str�mbeck, Russell Buchanan, Maximilian Enthoven
% ETH Zurich / UZH, HS 2016

%% Clearing workspaces, Closing windows & Clearing commandwindow
clear all;
close all;
clc;
%% initialize variables
format shortG
dataset = 0; % 0: KITTI, 1: Malaga, 2: parking

rng(1);

% Tuning Parameters

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
global pixel_threshold;

% Pose Estimation
use_p3p = true;

% Harris Corner Detector Parameters
% Randomly chosen parameters that seem to work well
harris_patch_size = 9;
harris_kappa = 0.08; % Magic number in range (0.04 to 0.15)
num_keypoints = 2000;
nonmaximum_supression_radius = 6;
descriptor_radius = 9;
match_lambda = 5;
pose_dist_threshold = 0.10; % 10% used by Google Tango
num_KLT_patches = 50; %number evenly spaced KLT patches
random_KLT_patches = false;
pixel_threshold = 1;


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
    groundTruth = load('kitti/poses/00.txt');
    
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
% bootstrap_frames = [95 97];

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

 fprintf('\n\nProcessing frame %d\n=====================\n', bootstrap_frames(1));
[firstKeypoints,firstLandmarks] = monocular_intialization(img0,img1,ransac,K,eye(3,4));
prevImage = img1;
prevState = [firstKeypoints;firstLandmarks(1:3,:)];

figure
set(gcf,'Position',[-1854 1 1855 1001])
subplot(1, 3, 3); %uncomment to display images
scatter3(firstLandmarks(1, :), firstLandmarks(2, :), firstLandmarks(3, :), 3,'b');
set(gcf, 'GraphicsSmoothing', 'on');
view(0,0);
axis equal;
axis vis3d;
axis([-15 15 -20 5 -20 30]);


%% Continuous operation
range = 1:last_frame;
dataBase = cell(3,3);
for i = 1:last_frame
    fprintf('\n\nProcessing frame %d\n=====================\n', i);
    if dataset == 0
        currImage = imread([kitti_path '/00/image_0/' sprintf('%06d.png',i)]);
        [currState, currPose, dataBase] = processFrame(prevState, prevImage, currImage, K, dataBase);
    elseif dataset == 1
        currImage = rgb2gray(imread([malaga_path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(i).name]));
        [currState, currPose, dataBase] = processFrame(prevState, prevImage, currImage, K, dataBase);
    elseif dataset == 2
        currImage = im2uint8(rgb2gray(imread([parking_path ...
            sprintf('/images/img_%05d.png',i)])));
        [currState, currPose, dataBase] = processFrame(prevState, prevImage, currImage, K, dataBase);
    else
        assert(false);
    end
    
    %check to see if we're lose and if so, re initialize
    if(isempty(currState))
        twoImagesAge = imread([kitti_path '/00/image_0/' sprintf('%06d.png',i-2)]);
        currPose = reshape(dataBase{3,2},3,4);%two poses ago
        [firstKeypoints,firstLandmarks] = monocular_intialization(twoImagesAge,currImage,ransac,K,currPose);
        currState = [firstKeypoints;firstLandmarks(1:3,:)];
        prevState = ones(5,size(currState,2));
    end
    
    R_C_W = currPose(:,1:3)
    t_C_W = currPose(:,4)

    % Distinguish success from failure.
    if (numel(R_C_W) > 0)
        
        subplot(1, 3, 3);
        plotCoordinateFrame(R_C_W', -R_C_W'*t_C_W, 2,['r';'r';'r']);
        hold on
%         if (dataset==0)
%             % Plot ground truth based on Dataset
%             truePose = reshape(groundTruth(i+1, :), 4, 3)';
%             rot = truePose(1:3,1:3);
%             trans = truePose(:,4);
%             plotCoordinateFrame(rot', rot*trans, 2,['b';'b';'b']);
%             hold on    
%         end
        delete(findobj(gca, 'type', 'patch'));
        if(exist('currentLandmarks'))
            delete(currentLandmarks)
        end
        pos = -R_C_W'*t_C_W;
        idx = ~ismember(currState(3:5,:)',prevState(3:5,:)','rows');
        test = currState(3:5,idx);
        currentLandmarks = scatter3(test(1, :), test(2, :), test(3, :), 5,'r','filled');
        axis([pos(1)-25 pos(1)+25 pos(2)-20 pos(2)+5 pos(3)-10 pos(3)+30]);
        view(0,0);
        hold off
        
        subplot(1, 3, [1 2]);
        imshow(currImage);
    else
        disp(['Frame ' num2str(i) ' failed to localize!']);
    end

    prevState = currState;
    prevImage = currImage;
    
    % Makes sure that plots refresh.    
    pause(0.01);
    
end
