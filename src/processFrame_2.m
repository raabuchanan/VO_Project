function [ currState, currPose ] = processFrame_2( prevState, prevImage, currImage, K )
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

%% EXTRACTS TRIANGULATION PART
% Triangulation part starts with NaN in first row
tempState = prevState(:,find(isnan(prevState(1,:))));
prevState = prevState(1:5,find(prevState(1,:)));

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
matchedCurrDescriptors = currDescriptors(:, matches > 0);
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

%% Output

currState = [flipud(matchedCurrKeypoints); p_W_landmarks(:, matchesList)];

currPose = [R_C_W, t_C_W];

%% START OF TRIANGULATION PART

% Retrieves the Descriptors % Keypoints without a Landmark-match
currKeypoints = currKeypoints(:,...
    find(~ismember(currKeypoints', matchedCurrKeypoints(:,inlier_mask==0)','rows')));
currDescriptors = currDescriptors(:,...
    find(~ismember(currKeypoints', matchedCurrKeypoints(:,inlier_mask==0)','rows')));
currPose = currPose;
%IF no information for triangulation exists
if isempty(tempState)
    %appends current image information to tempState
    tempState = [NaN(1,length(currKeypoints(1,:)));...
        currKeypoints;...
        currDescriptors;...
        repmat(currPose(:),1,length(currKeypoints(1,:)))];
else
    %A previous set of keypoints exists, now we need to determine which of
    %these have matches with the new keypoints: If a match is found the
    %keypoint is kept, if no match it is discarded. The new Keypoints that
    %are matched are kept for triangulation, the others are kept for future
    %triangulation
    prevKeypoints = tempState(2:3,:);
    prevDescriptors = tempState(4:end-12,:);
    prevPose = tempState(end-11:end,:);
    
    %% Find Matches Between NEW Image and OLD Images
    % Match Descriptors
    matches = matchDescriptors( currDescriptors, prevDescriptors, match_lambda);
    
    % For Triangulation
    matchedCurrKeypoints = currKeypoints(:, matches > 0);
    matchedCurrDescriptors = currDescriptors(:, matches > 0);
    
    % For FUTURE Triangulation
    unmatchedCurrKeypoints = currKeypoints(:, matches == 0);
    unmatchedCurrDescriptors = currDescriptors(:, matches == 0);
    
    % Matched old keypoints are kept:
    matchesList = matches(matches > 0);
    prevKeypoints = prevKeypoints(:, matchesList);
    prevDescriptors = prevDescriptors(:, matchesList);
    prevPose = prevPose(:, matchesList);
    
    % Calculation the Minimum Pose displacement required for triangulation
    distance2pw = mean(currState(5,:)-prevPose(end,1));
    pose2pose_threshold = distance2pw * pose_dist_threshold;
    
    %% Calculate what Keypoints that can be used for triangulation
    
    % Grouping Old Keypoints with same Pose, Segments = number of same pose
    values = unique(prevPose(end,:));
    segments = histc(prevPose,values);
    
    %% Performing RANSAC on Segments that fulfill pose2pose_threshold
    segment_val=0;
    for i = 1:length(segments)
        
        % Calculating Distance from Segment Pose to current Pose
        %segment of keypoints are set by segment_val_min & segment_val
        segment_val_min = segment_val +1;
        segment_val = segment_val + segments(i);
        p2p_distance = sqrt(...
            (prevPose(end-2,segment_val)-currPose(end-2))^2+...
            (prevPose(end-1,segment_val)-currPose(end-1))^2+...
            (prevPose(end,segment_val)-currPose(end))^2);
        
        if (p2p_distance > pose2pose_threshold)
            
            p1 = prevKeypoints(:,segment_val_min:segment_val);
            M1 =  reshape(prevPose(:,segment_val),3,4);
            p2 = matchedCurrKeypoints;
            M2 = currPose;
            
            % RANSAC
            %...
            
            % Dummy initialization of RANSAC variables
            num_iterations = 1000; % chosen by me
            pixel_threshold = 1; % specified in pipeline
            k = 50; % for non-p3p use
            
            % Triangulate a point cloud using the final transformation (R,T)
            P = linearTriangulation(p1,p2,M1,M2);
            
            num_inliers_history = zeros(1,num_iterations);
            max_num_inliers_history = zeros(1,num_iterations);
            % to fit all candidate points in matrix
            best_guess_history = zeros(3,k*num_iterations,2);
            
            for ii = 1:num_iterations
                
                % choose random data from landmarks
                [landmark_sample, idx] = datasample(P(1:3,:),k,2,'Replace',false);
                p1_sample = p1(:, idx);
                p2_sample = p2(:, idx);
                
                F_candidate = fundamentalEightPoint_normalized(p1_sample,p2_sample);
                % calculate epipolar line distance
                d = diag(epipolarLineDistance(F_candidate,p1_sample,p2_sample));
                % all relevant elements on diagonal
                inlierind = find(d<pixel_threshold);
                % inliers = d(inlierind);
                inliercount = length(inlierind);
                
                if ii == 1
                    max_num_inliers_history(ii) = inliercount;
                    counter = 1;
                elseif ii > 1
                    if inliercount > max(max_num_inliers_history)
                        max_num_inliers_history(ii) = inliercount;
                        
                        p1_inliers = p1_sample(:,inlierind);
                        p2_inliers = p2_sample(:,inlierind);
                        
                        d_2 = diag(epipolarLineDistance(F_candidate,...
                            p1_inliers,p2_inliers));
                        inlierind_2 = find(d_2<pixel_threshold);
                        
                        best_guess_1 = p1_inliers(:,inlierind_2);
                        best_guess_2 = p2_inliers(:,inlierind_2);
                        
                    elseif inliercount <= max(max_num_inliers_history)
                        % set to previous value
                        max_num_inliers_history(ii) = ...
                            max_num_inliers_history(ii-1);
                        
                    end
                    
                end
            end
            p1 = best_guess_1;
            p2 = best_guess_2;
            P = linearTriangulation(p1,p2,M1,M2);
            
            if ~isempty(newcurrState)
                newcurrState = [newcurrState,[p2;P]];
            else
                newcurrState = [p2;P];
            end
            currState = [currState,newcurrState];
            % IF Keypoint segment pose2pose distance is too small the
            % keypoints are saved for future triangulation
        else
            tempState = [NaN(1,length(prevPose(1,segment_val_min:segment_val)));...
                prevKeypoints(:,segment_val_min:segment_val);...
                prevDescriptors(:,segment_val_min:segment_val);...
                prevPose(:,segment_val_min:segment_val)];
        end
    end
    %For new keypoints that did not match they are saved for future
    %matching
    tempState = [tempState,[NaN(1,length(unmatchedCurrKeypoints(1,:)));...
        unmatchedCurrKeypoints;...
        unmatchedCurrDescriptors;...
        repmat(currPose(:),1,length(unmatchedCurrKeypoints(1,:)))]];
end

%UPDATING currState With the descriptors still unmatched but qualified
zeroState = zeros(...
    max(length(tempState(:,1)),5),...
    length(tempState(1,:)) + length(currState(1,:)));
zeroState(1:5,1:length(currState(1,:))) = currState;
zeroState(:,length(currState(1,:))+1:end) = tempState;
currState = zeroState;

end

%% WORKING