%%%%%%%%%% Mini Project: Vision Algorithms for Mobile Robotics %%%%%%%%%%%
% by Alex Lieber, Carl Str???mbeck, Russell Buchanan, Maximilian Enthoven
% ETH Zurich / UZH, HS 2016

%% Clearing workspaces, Closing windows & Clearing commandwindow
clear all;
close all;
clc;
%% initialize variables
format shortG
warning off
dataset = 3; % 0: KITTI, 1: Malaga, 2: parking, 3: tram
tic
rng(1);

% Tuning Parameters

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

global dataBaseSize;


%% set up relevant paths

kitti_path = '../kitti';
malaga_path = '../malaga-urban-dataset-extract-07/';
parking_path = '../parking';
tram_path = '../tram';

% Necessary paths
addpath(genpath('all_solns'))
addpath(genpath('src'))
addpath(genpath('testdata')) % here not necessary

if dataset == 0
    run kittiParameters
    assert(exist(kitti_path) ~= 0);
    ground_truth = load([kitti_path '/poses/00.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    last_frame = 4540;
    K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
    
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
    
elseif dataset == 2
    run parkingParameters
    last_frame = 598;
    K = load([parking_path '/K.txt']);
    ground_truth = load([parking_path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    
elseif dataset == 3
    run tramParameters
    last_frame = 715;
    load([tram_path '/calibration/cameraParams.mat']);
    K = (cameraParams.IntrinsicMatrix)';
else
    assert(false);
end

%% bootstrap / initialization of keypoint matching between adjacent frames

bootstrap_frames = [1 3]; % first and third frame
% bootstrap_frames = [170 173];

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
elseif dataset == 3
    img0 = rgb2gray(imread([tram_path ...
        sprintf('/images/scene%05d.jpg',5*(bootstrap_frames(1)-1)+1)]));
    img1 = rgb2gray(imread([tram_path ...
        sprintf('/images/scene%05d.jpg',5*(bootstrap_frames(2)-1)+1)]));
else
    assert(false);
end

% 1 for ransac 'yes'
ransac = 1;

fprintf('\n\nProcessing frame %d\n=====================\n', bootstrap_frames(1));
[firstKeypoints,firstLandmarks] = initializeVO(img0,img1,ransac,K,eye(3,4));

prevImage = img1;
prevState = [firstKeypoints;firstLandmarks(1:3,:)];

% figure
% set(gcf,'Position',[-1854 1 1855 1001])
% subplot(1, 3, 3); %uncomment to display images
% set(gcf, 'GraphicsSmoothing', 'on');
% view(0,0);
% axis equal;
% axis vis3d;
% axis([-15 15 -20 5 -20 30]);


%% Continuous operation
range = 1:last_frame;
dataBase = cell(3,dataBaseSize);
for i = 2:last_frame
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
    elseif dataset == 3
        currImage = rgb2gray(imread([tram_path '/images/' sprintf('scene%05d.jpg',5*(i-1)+1)]));
        [currState, currPose, dataBase] = processFrame(prevState, prevImage, currImage, K, dataBase);
    else
        assert(false);
    end
    
    %check to see if we're close and if so, re initialize
    if(isempty(currState))
        disp('Lost, will have to reinitialize from last pose')
        if dataset == 0
            twoImagesAgo = imread([kitti_path '/00/image_0/' sprintf('%06d.png',i-2)]);
        elseif dataset == 1
            twoImagesAgo = rgb2gray(imread([malaga_path ...
                '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
                left_images(i-2).name]));
        elseif dataset == 2
            twoImagesAgo = im2uint8(rgb2gray(imread([parking_path ...
                sprintf('/images/img_%05d.png',i-2)])));
        elseif dataset == 3
            twoImagesAgo = rgb2gray(imread([tram_path ...
                '/images/' sprintf('scene%05d.jpg',5*(i-3)+1)]));
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
        [firstKeypoints,firstLandmarks] = initializeVO(twoImagesAgo,currImage,ransac,K,currPose);
        currState = [firstKeypoints;firstLandmarks(1:3,:)];
        prevState = ones(5,size(currState,2));
        dataBase = cell(3,dataBaseSize);
    end
    
    R_C_W = currPose(:,1:3);
    t_C_W = currPose(:,4);

    % Distinguish success from failure.
%    if (numel(R_C_W) > 0)
        
        
        
%         subplot(1, 3, 3);
%         % plotCoordinateFrame(R_C_W', -R_C_W'*t_C_W, 2,['k';'k';'k']);
%         origin = -R_C_W'*t_C_W;
%         scatter3(origin(1),origin(2),origin(3),'k','filled','s');
%         hold on
% %         if (dataset==0)
% %             % Plot ground truth based on Dataset
% %             truePose = reshape(groundTruth(i+1, :), 4, 3)';
% %             rot = truePose(1:3,1:3);
% %             trans = truePose(:,4);
% %             plotCoordinateFrame(rot', rot*trans, 2,['b';'b';'b']);
% %             hold on    
% %         end
%         delete(findobj(gca, 'type', 'patch'));
%         if(exist('currentLandmarks'))
%             delete(currentLandmarks)
%         end
%         
%         pos = -R_C_W'*t_C_W;
%         idx = ~ismember(currState(3:5,:)',prevState(3:5,:)','rows');
%         test = currState(3:5,idx);
%         currentLandmarks = scatter3(currState(3, :), currState(4, :), currState(5, :), 5,'r','filled');
%         axis([pos(1)-25 pos(1)+25 pos(2)-20 pos(2)+5 pos(3)-10 pos(3)+30]);
%         view(0,0);
%         hold on
%         
%         subplot(1, 3, [1 2]);
%         imshow(currImage);
        
        
       
        
%    else
%        disp(['Frame ' num2str(i) ' failed to localize!']);
%    end

    prevState = currState;
    prevImage = currImage;
    
    plot_all
    
    % Makes sure that plots refresh.    
    drawnow
    pause(0.01);
    
end

toc
