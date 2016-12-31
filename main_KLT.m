%%%%%%%%%% Mini Project: Vision Algorithms for Mobile Robotics %%%%%%%%%%%
% by Alex Lieber, Carl Str�mbeck, Russell Buchanan, Maximilian Enthoven
% ETH Zurich / UZH, HS 2016

%% Clearing workspaces, Closing windows & Clearing commandwindow
clear all;
close all;
clc;
%% initialize variables

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
global templateSize;

% Pose Estimation
use_p3p = true;

% Harris Corner Detector Parameters
% Randomly chosen parameters that seem to work well
harris_patch_size = 9;
harris_kappa = 0.15; % Magic number in range (0.04 to 0.15)
num_keypoints = 1000;
nonmaximum_supression_radius = 8;
descriptor_radius = 9;
match_lambda = 4;
pose_dist_threshold = 0.10; % 10% used by Google Tango
num_KLT_patches = 100; %number evenly spaced KLT patches
random_KLT_patches = false;
pixel_threshold = 1;
templateSize = [20, 20];

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
[firstKeypoints,firstLandmarks] = monocular_intialization(img0,img1,ransac,K);
prevImage = img1;

figure
set(gcf,'Position',[-1854 1 1855 1001])
subplot(1, 3, 3); %uncomment to display images
scatter3(firstLandmarks(1, :), firstLandmarks(2, :), firstLandmarks(3, :), 3,'b');
set(gcf, 'GraphicsSmoothing', 'on');
view(0,0);
axis equal;
axis vis3d;
axis([-15 15 -20 5 -20 30]);

%% Get initial KLT templates

[M,N] = size(prevImage);

%remove keypoints too close to edge of image
%could be better
const1 = (firstKeypoints(1,:)>templateSize(1));
const2 = (firstKeypoints(1,:)<M-templateSize(1));
const3 = (firstKeypoints(2,:)>templateSize(2));
const4 = (firstKeypoints(2,:)<N-templateSize(2));

indx = logical(const1.*const2.*const3.*const4);

goodKeypoints = firstKeypoints(:,indx);
goodlandmarks = firstLandmarks(1:3,indx);

% [goodKeypoints_sample, idx] = datasample(goodKeypoints, num_KLT_patches, 2, 'Replace', false);
% goodlandmarks_sample = goodlandmarks(:,idx);

%Are they already sorted? The first x have histes harris score?
goodKeypoints_sample = goodKeypoints(:,1:num_KLT_patches);
goodlandmarks_sample = goodlandmarks(:,1:num_KLT_patches);

prevState = cell(3,num_KLT_patches);
for k=1:num_KLT_patches
    center = goodKeypoints_sample(:,k); % center of the template, in image coordinates.
    prevState{1,k} = [0 0 0 0 center(1) center(2)]; % [V;U]
    prevState{2,k} = im2double(prevImage(center(1)-templateSize(1):center(1)+templateSize(1),...
                           center(1)-templateSize(1):center(1)+templateSize(1))); % pixels of the template
    prevState{3,k} = goodlandmarks_sample(:,k);
end

dataBase = cell(3,5);

% Make a colormap
cmap = hot(256);

%% Continuous operation

for i = 2:last_frame
    fprintf('\n\nProcessing frame %d\n=====================\n', i);
    if dataset == 0
        currImage = imread([kitti_path '/00/image_0/' sprintf('%06d.png',i)]);
        [currState, currPose, dataBase] = processFrame_KLT(prevState, prevImage, currImage, K, dataBase);
    elseif dataset == 1
        currImage = rgb2gray(imread([malaga_path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(i).name]));
        [currState, currPose, dataBase] = processFrame_KLT(prevState, prevImage, currImage, K, dataBase);
    elseif dataset == 2
        currImage = im2uint8(rgb2gray(imread([parking_path ...
            sprintf('/images/img_%05d.png',i)])));
        [currState, currPose, dataBase] = processFrame_KLT(prevState, prevImage, currImage, K, dataBase);
    else
        assert(false);
    end
    
    R_C_W = currPose(:,1:3)
    t_C_W = currPose(:,4)

    % Distinguish success from failure.
    if (numel(R_C_W) > 0)
        
        subplot(1, 3, 3);
        plotCoordinateFrame(R_C_W', -R_C_W'*t_C_W, 2,['r';'r';'r']);
%         hold on
%         delete(findobj(gca, 'type', 'patch'));
%         if(exist('currentLandmarks'))
%             delete(currentLandmarks)
%         end
%         pos = -R_C_W'*t_C_W;
%         currentLandmarks = scatter3(currState.landmark(3, :), currState.landmark(4, :), currState.landmark(5, :), 5,'r','filled');
%         axis([pos(1)-15 pos(1)+15 pos(2)-20 pos(2)+5 pos(3)-10 pos(3)+30]);
        view(0,0);
        hold off
        
        subplot(1, 3, [1 2]);
        imshow(currImage);
        hold on
        
        % Display the tracked templates
        centers = cell2mat(currState(1,:));
        
        plot(centers(6,:),centers(5,:),'go',...
            'MarkerFaceColor',cmap(round(255 * k/length(currState))+1,:));
        hold off

%         % Get vertices of the template
%         [template_height, template_width] = size(currState{2,1});
%         halfw = template_width/2 - 20;
%         halfh = template_height/2 - 20;
%         v = [halfw, -halfw, -halfw,  halfw;
%              halfh,  halfh, -halfh, -halfh];
%         % Compute the position of the vertives of the (warped) template...
%         v = v + [currState(k).p(6); currState(k).p(5)] * ones(1,4);
% 
%         % Plot templates
%         color = rand(32,3);
%         % Plot vertices of (warped) template
%         plot_quadrilateral(v, color(k,:));

        drawnow
        
    else
        disp(['Frame ' num2str(i) ' failed to localize!']);
    end

    prevState = currState;
    prevImage = currImage;
    
    % Makes sure that plots refresh.    
    pause(0.01);
    
end
