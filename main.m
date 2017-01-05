%%%%%%%%%% Mini Project: Vision Algorithms for Mobile Robotics %%%%%%%%%%%
% by Alex Lieber, Carl Str�mbeck, Russell Buchanan, Maximilian Enthoven
% ETH Zurich / UZH, HS 2016

%% Clearing workspaces, Closing windows & Clearing commandwindow
clear all;
close all;
clc;
%% initialize variables
format shortG
warning off
dataset = 0; % 0: KITTI, 1: Malaga, 2: parking 3: tram
tic
%rng(1);
global dataBaseSize;

%% set up relevant paths

kitti_path = 'kitti';
malaga_path = 'malaga-urban-dataset-extract-07/';
parking_path = 'parking';
tram_path = 'tram';

% Necessary paths
addpath(genpath('src'))

if dataset == 0
    run kittiParameters
    assert(exist(kitti_path) ~= 0);
    ground_truth = load([kitti_path '/poses/00.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    last_frame = 4540;
    K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
    prevImage = imread([kitti_path '/00/image_0/' sprintf('%06d.png',1)]);
elseif dataset == 1
    run malagaParameters
    % Path containing the many files of Malaga 7.
    assert(exist(malaga_path) ~= 0);
    images = dir([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    last_frame = length(left_images);
    K = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
    prevImage = rgb2gray(imread([malaga_path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(1).name]));
elseif dataset == 2
    run parkingParameters
    last_frame = 598;
    K = load([parking_path '/K.txt']);
    ground_truth = load([parking_path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    prevImage = rgb2gray(imread([parking_path ...
            sprintf('/images/img_%05d.png',1)]));
elseif dataset == 3
    run tramParameters
    last_frame = 715;
    load([tram_path '/calibration/cameraParams.mat']);
    K = (cameraParams.IntrinsicMatrix)';
    prevImage = rgb2gray(imread([tram_path ...
        sprintf('/images/scene%05d.jpg',1)]));
else
    assert(false);
end

%% bootstrap / initialization of keypoint matching between adjacent frames

% 1 for ransac 'yes'
ransac = 1;

 fprintf('\n\nProcessing frame %d\n=====================\n', 1);
[firstKeypoints,firstLandmarks] = autoMonoInitialization(dataset,ransac,K,eye(3,4));
prevState = [firstKeypoints;firstLandmarks(1:3,:)];


%% Continuous operation
dataBase = cell(3,dataBaseSize);
for ii = 2:last_frame
    fprintf('\n\nProcessing frame %d\n=====================\n', ii);
    if dataset == 0
        currImage = imread([kitti_path '/00/image_0/' sprintf('%06d.png',ii)]);
        [currState, currPose, dataBase] = processFrame(prevState, prevImage, currImage, K, dataBase);
    elseif dataset == 1
        currImage = rgb2gray(imread([malaga_path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(ii).name]));
        [currState, currPose, dataBase] = processFrame(prevState, prevImage, currImage, K, dataBase);
    elseif dataset == 2
        currImage = rgb2gray(imread([parking_path ...
            sprintf('/images/img_%05d.png',ii)]));
        [currState, currPose, dataBase] = processFrame(prevState, prevImage, currImage, K, dataBase);
    elseif dataset == 3
        currImage = rgb2gray(imread([tram_path '/images/' sprintf('scene%05d.jpg',5*(ii-1)+1)]));
        [currState, currPose, dataBase] = processFrame(prevState, prevImage, currImage, K, dataBase);
    else
        assert(false);
    end
    
    %check to see if we're close and if so, re initialize
    if(isempty(currState))
        disp('Lost, will have to reinitialize from last pose')
        if dataset == 0
            twoImagesAgo = imread([kitti_path '/00/image_0/' sprintf('%06d.png',ii-2)]);
        elseif dataset == 1
            twoImagesAgo = rgb2gray(imread([malaga_path ...
                '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
                left_images(ii-2).name]));
        elseif dataset == 2
            twoImagesAgo = im2uint8(rgb2gray(imread([parking_path ...
                sprintf('/images/img_%05d.png',ii-2)])));
        elseif dataset == 3
            twoImagesAgo = rgb2gray(imread([tram_path ...
                '/images/' sprintf('scene%05d.jpg',5*(ii-3)+1)]));
        else
            assert(false);
        end
        
        emptyColumns = find(cellfun(@isempty,dataBase(1,:)));
        if(isempty(emptyColumns))
            idx = dataBaseSize-1;
        else
            idx = min(emptyColumns) - 1;
        end
        
        currPose = reshape(dataBase{3,idx},3,4);%two poses ago
        [firstKeypoints,firstLandmarks] = monoInitialization(twoImagesAgo,currImage,ransac,K,currPose);
        currState = [firstKeypoints;firstLandmarks(1:3,:)];
        dataBase = cell(3,dataBaseSize);
    end
    
    R_C_W = currPose(:,1:3);
    t_C_W = currPose(:,4);

    prevState = currState;
    prevImage = currImage;
    
    if ii> 5
    run Plot_all
    end
    
    % Makes sure that plots refresh.    
    pause(0.01);
    
end

toc
