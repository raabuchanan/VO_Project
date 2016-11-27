function [ currState, currPose ] = processFrame( prevState, prevImage, currImage, K )
% prevState is a 5xk matrix where the columns corespond to 2D points on top
% of the coresponding 3d points. k is the number of keypoints/landmarks

global harris_patch_size;
global harris_kappa;
global num_keypoints;
global nonmaximum_supression_radius;
global descriptor_radius;
global match_lambda;
global use_p3p;

%% P3P or DLT
if use_p3p
    num_iterations = 200;
    pixel_tolerance = 10;
    k = 3;
else
    num_iterations = 2000;
    pixel_tolerance = 10;
    k = 6;
end

%% For testing, get state from prevImage
if (prevState == 0)
    
    % Calculate Harris scores
    testHarrisScores = harris(prevImage, harris_patch_size, harris_kappa);
    assert(min(size(testHarrisScores) == size(prevImage)));
    % Select keypoints
    prevState = selectKeypoints(...
        testHarrisScores, num_keypoints, nonmaximum_supression_radius);
    prevState = [prevState;zeros(3,num_keypoints)];
end

%% Process prevImage

prevKeypoints = prevState(1:2,:);
p_W_landmarks = prevState(3:5,:);

prevDescriptors = describeKeypoints(prevImage, prevKeypoints, descriptor_radius);

%% Process currImage
% Calculate Harris scores
currHarrisScores = harris(currImage, harris_patch_size, harris_kappa);
assert(min(size(currHarrisScores) == size(currImage)));

% Select keypoints
currKeypoints = selectKeypoints(...
    currHarrisScores, num_keypoints, nonmaximum_supression_radius);

% Get Descriptors
currDescriptors = describeKeypoints(currImage, currKeypoints, descriptor_radius);

%% Find Matches in two images
% Match Descriptors
matches = matchDescriptors( currDescriptors, prevDescriptors, match_lambda);

matchedCurrKeypoints = currKeypoints(:, matches > 0);
matchesList = matches(matches > 0);
matchedLandmarks = p_W_landmarks(:, matchesList);

%% RANSAC

% Initialize RANSAC.
inlier_mask = zeros(1, size(matchedCurrKeypoints, 2));
matchedCurrKeypoints = flipud(matchedCurrKeypoints);
max_num_inliers_history = zeros(1, num_iterations);
max_num_inliers = 0;

% RANSAC
for i = 1:num_iterations
    [landmark_sample, idx] = datasample(matchedLandmarks, k, 2, 'Replace', false);
    keypoint_sample = matchedCurrKeypoints(:, idx);
    
    if use_p3p
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
    else
        M_C_W_guess = estimatePoseDLT(...
            keypoint_sample', landmark_sample', K);
        R_C_W_guess = M_C_W_guess(:, 1:3);
        t_C_W_guess = M_C_W_guess(:, end);
    end
    
    % Count inliers:
    projected_points = projectPoints(...
        (R_C_W_guess(:,:,1) * matchedLandmarks) + ...
        repmat(t_C_W_guess(:,:,1), ...
        [1 size(matchedLandmarks, 2)]), K);
    difference = matchedCurrKeypoints - projected_points;
    errors = sum(difference.^2, 1);
    is_inlier = errors < pixel_tolerance^2;
    
    if use_p3p
        projected_points = projectPoints(...
            (R_C_W_guess(:,:,2) * matchedLandmarks) + ...
            repmat(t_C_W_guess(:,:,2), ...
            [1 size(matchedLandmarks, 2)]), K);
        difference = matchedCurrKeypoints - projected_points;
        errors = sum(difference.^2, 1);
        alternative_is_inlier = errors < pixel_tolerance^2;
        if nnz(alternative_is_inlier) > nnz(is_inlier)
            is_inlier = alternative_is_inlier;
        end
    end
    
    if nnz(is_inlier) > max_num_inliers && nnz(is_inlier) >= 6
        max_num_inliers = nnz(is_inlier);        
        inlier_mask = is_inlier;
    end
    
    max_num_inliers_history(i) = max_num_inliers;
end

if max_num_inliers == 0
    R_C_W = [];
    t_C_W = [];
else
    M_C_W = estimatePoseDLT(...
        matchedCurrKeypoints(:, inlier_mask>0)', ...
        matchedLandmarks(:, inlier_mask>0)', K);
    R_C_W = M_C_W(:, 1:3);
    t_C_W = M_C_W(:, end);
end


end

