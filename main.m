clear
close all
clc

addpath(genpath('src'));
addpath(genpath('all_solns'));

parameters;

K = load('data/K.txt');
intitialKeypoints = load('data/keypoints.txt')'; % [v; u]
initialLandmarks = load('data/p_W_landmarks.txt')';% [X,Y,Z]
groundTruth = load('data/poses/00.txt');

% Declaring tempstate for triangulation
tempState = [];

figure;
%subplot(1, 3, 3); %uncomment to display images
scatter3(initialLandmarks(1, :), initialLandmarks(2, :), initialLandmarks(3, :), 3,'b');
set(gcf, 'GraphicsSmoothing', 'on');
view(0,0);
axis equal;
axis vis3d;
axis([-15 10 -10 5 -1 40]);

prevState = [intitialKeypoints; initialLandmarks];
prevImage = imread(sprintf('data/%06d.png',0));
for i = 1:20
    disp(['Frame ' num2str(i) ' being processed!']);
    currImage = imread(sprintf('data/%06d.png',i));
    
    %newkp= sum(~ismember(prevState(3:5,:)',p_W_landmarks(1:3,:)', 'rows'));
    
    [ currState, currPose, tempState ] = processFrame_new(...
      prevState, prevImage, currImage, K, tempState);
    
    R_C_W = currPose(:,1:3);
    t_C_W = currPose(:,4);

    % Distinguish success from failure.
    if (numel(R_C_W) > 0)
        %subplot(1, 3, 3);
        plotCoordinateFrame(R_C_W', -R_C_W'*t_C_W, 2,['r';'r';'r']);
        hold on
       
        % Ground truth
        truePose = reshape(groundTruth(i+1, :), 4, 3)';
        rot = truePose(1:3,1:3);
        trans = truePose(:,4);
        plotCoordinateFrame(rot', rot*trans, 2,['b';'b';'b']);
        hold on
        scatter3(currState(3, :), currState(4, :), currState(5, :), 5,'r','filled');
        view(0,0);
        hold off
    else
        disp(['Frame ' num2str(i) ' failed to localize!']);
    end
    
%     subplot(1, 3, [1 2]);
%     imshow(currImage);
    
    prevState = currState;
    prevImage = currImage;

    pause(0.01);
    disp(['New landmarks:' num2str(sum(...
        ~ismember(prevState(3:5,:)',initialLandmarks(1:3,:)', 'rows')))]);
    
end