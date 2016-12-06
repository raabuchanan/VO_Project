function [ currState, currPose,triState ] = processFrame( prevState, prevImage, currImage, K,triState )
% prevState is a 5xk matrix where the columns corespond to 2D points on top
% of the coresponding 3d points. k is the number of keypoints/landmarks
% , 
% 
global harris_patch_size;
global harris_kappa;
global num_keypoints;
global nonmaximum_supression_radius;
global descriptor_radius;
global match_lambda;
global use_p3p;

global num_candidate_keypoints;
global threshold_triangulation;
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

%% Output

%Changed to matches List
currState = [flipud(matchedCurrKeypoints); p_W_landmarks(:, matchesList)];

currPose = [R_C_W, t_C_W];


%% START OF TRIANGULATION PART
%NOTE: triState including info has to be converted to double because of pose

%initializing
if triState == 0
    % Calculate Harris scores
    testHarrisScores = harris(currImage, harris_patch_size, harris_kappa);
    % Select keypoints
    oldKeypoints = selectKeypoints(...
        testHarrisScores, num_candidate_keypoints, nonmaximum_supression_radius);
    %extract descriptors
    olddes = describeKeypoints(currImage, oldKeypoints, descriptor_radius);
    %takes descriptors with matching world points and removes them from set
    matchedCurrDescriptors = currDescriptors(:, matches > 0);
    [olddes,idx] = rmdoubledes(olddes,matchedCurrDescriptors);
    oldKeypoints = oldKeypoints(:,idx);
    oldKeypoints = im2double(oldKeypoints);
    %sets initial triState contraining NaN;descriptor;keypoint;pose
    olddes = im2double(olddes);
    triState = [NaN(1,length(olddes(1,:)));olddes;oldKeypoints;...
        repmat(currPose(:),1,length(olddes(1,:)))];

else
    %% If we have descriptors without 3d worldpoints we calculate new
    % Calculate Harris scores
    testHarrisScores = harris(currImage, harris_patch_size, harris_kappa);
    % Select keypoints
    newKeypoints = selectKeypoints(...
        testHarrisScores, num_candidate_keypoints, nonmaximum_supression_radius);
    %extract descriptors
    newdes = describeKeypoints(currImage, newKeypoints, descriptor_radius);
    %takes descriptors with matching world points and removes them from set
    matchedCurrDescriptors = currDescriptors(:, matches > 0);
    [newdes,idx] = rmdoubledes(newdes,matchedCurrDescriptors);
    newKeypoints = newKeypoints(:,idx);
    newKeypoints = im2double(newKeypoints);
    %sets new triState from current Image
    newdes = im2double(newdes);
    
    
    newtriState = [NaN(1,length(newdes(1,:)));newdes;newKeypoints;...
        repmat(currPose(:),1,length(newdes(1,:)))];

    %     %Discards traces where descriptor no longer exists in newdes
    %     Lia = ismember(triState(4:end-12,:)',newdes(3:end,:)','rows');
    %     triState = triState(:,Lia>0);
    % couldnt be done yet since no matches, the descriptors ofc change and
    % we need to do ransac on what keypoints accually match and what
    % keypoints that are new
    
    %Updating triState with old and new datapoints

    triState = [triState,newtriState];
    
    %     %removing the new descriptors w/o worldpoints if older exists
    %     occurrence='first';
    %     [~,uniqueindex,~] = unique(triState(2:end-14,:)','rows',occurrence);
    %     triState = triState(:,uniqueindex);
    %cant be done yeat since descriptors are not equal ofc
end

%% Take triState and find what points can be triangulated based on poses
%temporary distance calculation for thresholding what pose to choose
distance2pw = mean(currState(5,:));
pose2pose_threshold = distance2pw * threshold_triangulation;

% States that can be triangulated and matched and what index the
% untriangulated/triangulated poses are set
[matching,breakpoint] = triangulatedStates(triState,pose2pose_threshold,match_lambda);


%retrieving matched keypoints that can be triangulated as one set and the
%ones that are new in one set and the old keypoints for triangulation
newstate = triState(:,breakpoint+1:end);
matchednewKeypoints = newstate(end-13:end-12, matching(breakpoint+1:end) > 0);
matchesList = matching(matching(:,breakpoint+1:end) > 0);
matchedoldKeypoints = triState(end-13:end-12, matchesList);

%extracts the different images aka intervals containing the same pose
values = unique(triState(end,matchesList));
segments = histc(triState(end,matchesList),values);
%% RANSAC


%% Need to do one ransac for each pose available for triangulation
accumulated_states = 0;
newkp = matchednewKeypoints;
newkp(3,:) = 1;

%state with all new inliers with keypoints
newcurrState = [];

%performing one RANSAC process for each previous pose
for ii=1:length(segments(1,:))
    %extracting keypoints from the current segment
    minsegment = accumulated_states+1;
    accumulated_states = accumulated_states + segments(ii);
    maxsegment = accumulated_states;
    oldkp = matchedoldKeypoints(:,minsegment:maxsegment-1);
    oldkp(3,:) = 1;
    oldpose = reshape(triState(end-11:end,accumulated_states),3,4);
    
    % Initialize RANSAC.
    inlier_mask = zeros(1, size(matchednewKeypoints, 2));
    newkp = flipud(newkp);
    max_num_inliers_history = zeros(1, num_iterations);
    max_num_inliers = 0;
    k = 10; %number of points taken for datasample, min for 8point=8
    inlier_tol =1; %one pixel
    
    %only does for segments that can be triangulated with the current image
    if accumulated_states<=breakpoint
           
        for i=1:num_iterations
            %randomly selects a set of k keypoints
            [oldkp_sample, idx] = datasample(...
                matchedoldKeypoints(:,minsegment:maxsegment), k, 2, 'Replace', false);
            newkp_sample = matchednewKeypoints(:, idx);
            oldkp_sample;
            oldkp_sample(3,:) = 1;
            newkp_sample(3,:) = 1;
            
            %guess a Fundamental matrix
            F_guess = fundamentalEightPoint_normalized(oldkp_sample,newkp_sample);
                        
            % Check the epipolar constraint, distance from the epipolarline
            curr_inlier =0;
            dist_epi = zeros(1,length(oldkp(1,:)));
            % check each set of keypoints and find how many are inliers
            for iii=1:length(oldkp(1,:))
                dist_epi(iii) = ...
                    distPoint2EpipolarLine(F_guess,oldkp(:,iii),newkp(:,iii));
                if dist_epi(iii)<inlier_tol
                    curr_inlier = curr_inlier + 1;
                end
            end
            if curr_inlier>=max_num_inliers
                max_num_inliers = curr_inlier;
                inlier_mask = dist_epi<inlier_tol;
            end
            max_num_inliers_history(i) = max_num_inliers
        end
        %temporary for debugg
        inlier_mask = inlier_mask(:,1:length(oldkp(1,:)));
        %create a new state containing the points considred as inliers
        oldkp_inlier = oldkp(:,inlier_mask>0);
        newkp_inlier = newkp(:,inlier_mask>0);
        P = linearTriangulation(oldkp_inlier,newkp_inlier,oldpose,currPose);
        %update newcurrState with inliers
        newcurrState = [newcurrState,[newkp_inlier(1:2,:);P(1:3,:)]];
    end
end

%missing the removal of keypoints that exists in two frames, keeping the
%older one, this should be done before the triangulation RANSAC for all
%except the new descriptors

%the current adding doesnt seem to work properly, next image doesnt take
%advantage of them giving us too few images to triangulate and create F

%adds the newly triangulated keypoints 
currState = [currState,newcurrState];
%triState needs to be updated


end

