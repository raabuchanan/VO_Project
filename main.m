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
pastPoints = [];%formerly known as tempstate

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
    
    [ currState, currPose, pastPoints] = processFrame(prevState, prevImage, currImage, K, pastPoints);
    
    R_C_W = currPose(:,1:3);
    t_C_W = currPose(:,4);

    % Distinguish success from failure.
    if (numel(R_C_W) > 0)
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
        
        disp(['Rotation '])
        rotation_error = abs(R_C_W - rot)
        disp(['Translation '])
        translate_error = abs(t_C_W + trans)
        
    else
        disp(['Frame ' num2str(i) ' failed to localize!']);
    end
    
    prevState = currState;
    prevImage = currImage;

    pause(0.01);
    
end