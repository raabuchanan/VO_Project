function [ currState, currPose, pastPoints] = processFrame(...
    prevState, prevImage, currImage, K, pastPoints)
% prevState is a 5xk matrix where the columns corespond to 2D points on top
% of the coresponding 3d points. k is the number of keypoints/landmarks

addpath(genpath('../../all_solns'))

global harris_patch_size;
global harris_kappa;
global num_keypoints;
global nonmaximum_supression_radius;
global descriptor_radius;
global match_lambda;
global use_p3p;
global pose_dist_threshold;

%% P3P or DLT
if use_p3p
    num_iterations = 400;
    pixel_tolerance = 5;
    k = 3;
else
    num_iterations = 2000;
    pixel_tolerance = 5;
    k = 6;
end

%% Process prevImage

prevKeypoints = prevState(1:2,:); % [V;U]
prevLandmarks = prevState(3:5,:); %[X;Y;Z]

prevDescriptors = describeKeypoints(prevImage, prevKeypoints, descriptor_radius);

%% Process currImage
% Calculate Harris scores
currHarrisScores = harris(currImage, harris_patch_size, harris_kappa);
assert(min(size(currHarrisScores) == size(currImage)));

% Select keypoints [V; U]
currKeypoints = selectKeypoints(...
    currHarrisScores, num_keypoints, nonmaximum_supression_radius);

% Get Descriptors
currDescriptors = describeKeypoints(currImage, currKeypoints, descriptor_radius);

%% Find Matches in two images

matches = matchDescriptors( currDescriptors, prevDescriptors, match_lambda);
% [V;U]
matchedCurrKeypoints = currKeypoints(:, matches > 0);
matchedLandmarks = prevLandmarks(:, matches(matches > 0));

%% RANSAC

% Initialize RANSAC.
%inlier_mask = zeros(1, size(matchedCurrKeypoints, 2));
matchedCurrKeypoints = flipud(matchedCurrKeypoints); % [U;V]
max_num_inliers_history = zeros(1, num_iterations);
max_num_inliers = 0;

for i = 1:num_iterations
    [landmark_sample, idx] = datasample(matchedLandmarks, k, 2, 'Replace', true);
    keypoint_sample = matchedCurrKeypoints(:, idx);


    normalized_bearings = K\[keypoint_sample; ones(1, 3)];
    for ii = 1:3
        normalized_bearings(:, ii) = normalized_bearings(:, ii) / ...
            norm(normalized_bearings(:, ii), 2);
    end
    
    poses = p3p(landmark_sample, normalized_bearings);
    R_C_W_guess = zeros(3, 3, 2);
    t_C_W_guess = zeros(3, 1, 2);
    for ii = 0:1
        R_W_C_ii = real(poses(:, (2+ii*4):(4+ii*4)));
        t_W_C_ii = real(poses(:, (1+ii*4)));
        R_C_W_guess(:,:,ii+1) = R_W_C_ii';
        t_C_W_guess(:,:,ii+1) = -R_W_C_ii'*t_W_C_ii;
    end

    
    % Count inliers for guess 1
    projected_points = projectPoints(...
        (R_C_W_guess(:,:,1) * matchedLandmarks) + repmat(t_C_W_guess(:,:,1), ...
        [1 size(matchedLandmarks, 2)]), K);
    difference = matchedCurrKeypoints - projected_points;
    errors = sum(difference.^2, 1);
    inliers = errors < pixel_tolerance^2;
    guess = 1;
    
    % Count inliers for guess 2
    projected_points = projectPoints(...
        (R_C_W_guess(:,:,2) * matchedLandmarks) + repmat(t_C_W_guess(:,:,2), ...
        [1 size(matchedLandmarks, 2)]), K);
    difference = matchedCurrKeypoints - projected_points;
    errors = sum(difference.^2, 1);
    inliers_guess_2 = errors < pixel_tolerance^2;
    
    if nnz(inliers_guess_2) > nnz(inliers)
        inliers = inliers_guess_2;
        guess = 2;
    end
    
    if nnz(inliers) > max_num_inliers && nnz(inliers) >= 6
        max_num_inliers = nnz(inliers);
        best_R_C_W_guess = R_C_W_guess(:,:,guess);
        best_T_C_W_guess = t_C_W_guess(:,:,guess);
    end
    
    max_num_inliers_history(i) = max_num_inliers;
end

if max_num_inliers == 0
    disp(['Impossible to create new Pose']);
    R_C_W = [];
    t_C_W = [];
else
    R_C_W = best_R_C_W_guess;
    t_C_W = best_T_C_W_guess;
end

% back to [V;U]
matchedCurrKeypoints = flipud(matchedCurrKeypoints);

currState = [matchedCurrKeypoints; matchedLandmarks];

currPose = [R_C_W, t_C_W];

%% START OF TRIANGULATION PART

% Retrieves the Descriptors % Keypoints without a Landmark-match
unMatchedIndices = ~ismember(currKeypoints', matchedCurrKeypoints','rows');
% [V;U]
currTriKeypoints = currKeypoints(:,unMatchedIndices);
currTriDescriptors = currDescriptors(:,unMatchedIndices);

    %First time running processFrame
if isempty(pastPoints)%formerly known as tempstate
    %appends current image information to pastPoints
    pastPoints = [double(currTriKeypoints);...
        im2double(currTriDescriptors);...
        repmat(currPose(:),1,length(currTriKeypoints(1,:)))];    
else
    %After first time
    prevTriKeypoints = (pastPoints(1:2,:));%%pull previous keypoints from tempstate [v;u]
    prevTriDescriptors = im2uint8(pastPoints(3:end-12,:));%pull descriptor for each prev keypoint
    prevTriPoses = pastPoints(end-11:end,:);%pull all previous poses
    
    matches = matchDescriptors( prevTriKeypoints, currTriKeypoints, match_lambda);
    % [V;U]
    % For Triangulation
    matchesList = matches(matches > 0);
    matchedCurrTriKeypoints = currTriKeypoints(:, matchesList);
    matchedCurrTriDescriptors = currTriDescriptors(:, matchesList);
    % [V;U]
    % ONLY Matched previous keypoints are kept:
    prevTriKeypoints = prevTriKeypoints(:, matches > 0);
    prevTriDescriptors = prevTriDescriptors(:, matches > 0);
    prevTriPoses = prevTriPoses(:, matches > 0);
    % [V;U]
    % Unmatched current keypoints are saved for future triangulation
    unmatchedCurrTriKeypoints = currTriKeypoints(:, setdiff(1:size(currTriKeypoints,2),matchesList));
    unmatchedCurrTriDescriptors = currTriDescriptors(:, setdiff(1:size(currTriKeypoints,2),matchesList));
    
    
    % now triangulate new landmarks
    
end



    

end
