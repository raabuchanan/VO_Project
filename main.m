clear
close all
clc

addpath(genpath('src'));

parameters;


% Data from exercise 6
K = load('data/K.txt');
keypoints = load('data/keypoints.txt')';
p_W_landmarks = load('data/p_W_landmarks.txt')';

prevState = [keypoints; p_W_landmarks];

prevImage = imread(sprintf('data/%06d.png',0));

%% Part 4 - Same, for all frames

% Declaring tempstate for triangulation
tempState = [];
fakepose = [0,0,0]';

figure(5);
%subplot(1, 3, 3);
scatter3(p_W_landmarks(1, :), p_W_landmarks(2, :), p_W_landmarks(3, :), 5);
set(gcf, 'GraphicsSmoothing', 'on');
view(0,0);
axis equal;
axis vis3d;
axis([-15 10 -10 5 -1 40]);
for i = 1:20
    disp(['Frame ' num2str(i) ' being processed!']);
    currImage = imread(sprintf('data/%06d.png',i));
    
    [ currState, currPose, tempState,fakepose ] = processFrame_new(...
        prevState, prevImage, currImage, K, tempState,fakepose );
    
    R_C_W = currPose(:,1:3);
    t_C_W = currPose(:,4);

    % Distinguish success from failure.
    if (numel(R_C_W) > 0)
        %subplot(1, 3, 3);
        plotCoordinateFrame(R_C_W', -R_C_W'*t_C_W, 2);
        hold on
        scatter3(currState(3, :), currState(4, :), currState(5, :), 5);
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
    
end