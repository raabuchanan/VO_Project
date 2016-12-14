function [ currState, currPose, tempState] = processFrame_new(...
    prevState, prevImage, currImage, K, tempState)
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
    num_iterations = 200;
    pixel_tolerance = 10;
    k = 3;
else
    num_iterations = 2000;
    pixel_tolerance = 5;
    k = 6;
end

%% Process prevImage

prevKeypoints = prevState(1:2,:); % [V;U]
p_W_landmarks = prevState(3:5,:);

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
% Match Descriptors both in [V;U]
matches = matchDescriptors( currDescriptors, prevDescriptors, match_lambda);

matchedCurrKeypoints = currKeypoints(:, matches > 0);
matchesList = matches(matches > 0);
matchedLandmarks = p_W_landmarks(:, matchesList);

%% RANSAC

% Initialize RANSAC.
inlier_mask = zeros(1, size(matchedCurrKeypoints, 2));
matchedCurrKeypoints = flipud(matchedCurrKeypoints); % !! [U;V] !! Flipped!
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
    disp(['Impossible to create new Pose']);
else
    M_C_W = estimatePoseDLT(...
        matchedCurrKeypoints(:, inlier_mask>0)', ...
        matchedLandmarks(:, inlier_mask>0)', K);
    R_C_W = M_C_W(:, 1:3);
    t_C_W = M_C_W(:, end);
end

%% Output
% back to [V;U]
matchedCurrKeypoints = flipud(matchedCurrKeypoints);

currState = [matchedCurrKeypoints; p_W_landmarks(:, matchesList)];

currPose = [R_C_W, t_C_W];
tempPose = currPose;


%% START OF TRIANGULATION PART

% Retrieves the Descriptors % Keypoints without a Landmark-match

unMatchedIndices = ~ismember(currKeypoints', matchedCurrKeypoints','rows');

currKeypoints = currKeypoints(:,unMatchedIndices);
currDescriptors = currDescriptors(:,unMatchedIndices);


%IF no information for triangulation exists
if isempty(tempState)
    %appends current image information to tempState
    tempState = [double(currKeypoints);...
        im2double(currDescriptors);...
        repmat(currPose(:),1,length(currKeypoints(1,:)))];
    
else
    %A previous set of keypoints exists, now we need to determine which of
    %these have matches with the new keypoints: If a match is found the
    %keypoint is kept, if no match it is discarded. The new Keypoints that
    %are matched are kept for triangulation, the others are kept for future
    %triangulation
    
    currPose = [reshape(tempState(end-11:end,end),3,4);0,0,0,1]...
        * [currPose;0,0,0,1];
    currPose = currPose(1:3,1:4);
    
    prevKeypoints = (tempState(1:2,:));
    prevDescriptors = im2uint8(tempState(3:end-12,:));
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
    
    % ONLY Matched old keypoints are kept:
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
    segments = flipud(histc(prevPose(end,:),values)')';
    
    % Might do matching in here
    %% FOR EACH SEGMENT, DO RANSAC IF SIZE>k otherwise discard totally
    new_landmarks = [];
    segment_val = 0;
    for i=1:length(segments)
        segment_val_min = segment_val + 1;
        segment_val = segments(i) + segment_val;
        
        % Setting up for RANSAC
        p1 = prevKeypoints(:,segment_val_min:segment_val);
        M1 =  reshape(prevPose(:,segment_val_min),3,4);
        p2 = (matchedCurrKeypoints);
        M2 = currPose;
        p1(3,:)=1;
        p2(3,:)=1;
        
        % Pose to pose displacement for current section
        p2p_dist = sqrt(...
            (prevPose(end-2,segment_val_min)-currPose(end-2))^2+...
            (prevPose(end-1,segment_val_min)-currPose(end-1))^2+...
            (prevPose(end,segment_val_min)-currPose(end))^2);
        % ONLY do ransac if displacement is long enough and segment has
        % more than 8 elements.
        k = 8; % number of datapoints selected(minimum 8)
        if (segments(i) >= k && p2p_dist > pose2pose_threshold)
            disp(['Frame can be triangulated!']);
            pixel_threshold =1;
            num_inliers_history = zeros(1,num_iterations);
            max_num_inliers_history = zeros(1,num_iterations);
            % to fit all candidate points in matrix
            best_guess_history = zeros(3,k*num_iterations,2);
            
            for ii = 1:num_iterations
                
                % choose random data from landmarks
                [~, idx] = datasample(p1(1:3,:),k,2,'Replace',false);
                p1_sample = p1(:, idx);
                p2_sample = p2(:, idx);
                
                F_candidate = fundamentalEightPoint_normalized(p1_sample,p2_sample);
                
                % calculate epipolar line distance
                d = epipolarLineDistance(F_candidate,p1,p2);
                
                % all relevant elements on diagonal
                inlierind = find(d<pixel_threshold);
                inliercount = length(inlierind);
                
                if ii == 1
                    max_num_inliers_history(ii) = inliercount;
                elseif ii > 1
                    
                    if inliercount > max(max_num_inliers_history) && inliercount>=8
                        max_num_inliers_history(ii) = inliercount;
                        
                        % use inliers to compute BEST GUESS
                        
                        p1_inliers = p1(:,inlierind);
                        p2_inliers = p2(:,inlierind);
                        
                        best_F_candidate = fundamentalEightPoint_normalized(p1_inliers,p2_inliers);
                        
                        d_2 = epipolarLineDistance(best_F_candidate,p1,p2);
                        inlierind_2 = find(d_2<pixel_threshold);
                        
                        best_guess_1 = p1(:,inlierind_2);
                        best_guess_2 = p2(:,inlierind_2);
                    elseif inliercount <= max(max_num_inliers_history)
                        % set to previous value
                        max_num_inliers_history(ii) = ...
                            max_num_inliers_history(ii-1);
                    end
                end
            end % END OF RANSAC LOOP
            % Adding NEW landmarks
            P = linearTriangulation(best_guess_1,best_guess_2,K*M1,K*M2);
            if isempty(new_landmarks)
                new_landmarks = [(best_guess_2(1:2,:));P(1:3,:)];
            else
                new_landmarks = [new_landmarks,...
                    [(best_guess_2(1:2,:));P(1:3,:)]];
            end %end of adding landmarks
        elseif segments(i)<8
            % REMOVE KEYPOINTS SINCE NO TRIANGULATION POSSIBLE(for data)
        end % END OF IF RANSAC OR NOT
        
    end % END OF RANSAC FOR SEGMENTS
    %Adding new landmarks to current landmarks data

    % COMMENT/UNCOMMENT to include new landmarks
    currState = [currState,new_landmarks];
    
    % Adding unmatch still vaild landmarks
    tempState = [tempState,...
        [double(unmatchedCurrKeypoints);...
        im2double(unmatchedCurrDescriptors);...
        repmat(currPose(:),1,length(unmatchedCurrKeypoints(1,:)))]];

    currPose = tempPose;
    
end
end
