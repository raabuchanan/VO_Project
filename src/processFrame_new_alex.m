function [ currState, currPose, tempState] = processFrame_new_alex(...
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
inlier_mask = zeros(1, size(matchedCurrKeypoints, 2));
matchedCurrKeypoints = flipud(matchedCurrKeypoints); % [U;V]
max_num_inliers_history = zeros(1, num_iterations);
max_num_inliers = 0;

for i = 1:num_iterations
    [landmark_sample, idx] = datasample(matchedLandmarks, k, 2, 'Replace', true);
    keypoint_sample = matchedCurrKeypoints(:, idx);
 %%%p3p section start:   
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
%%%%p3p section end.

%%%%%DLT (pnp) section start:
    else
        M_C_W_guess = estimatePoseDLT(...
            keypoint_sample', landmark_sample', K);
        R_C_W_guess = M_C_W_guess(:, 1:3);
        t_C_W_guess = M_C_W_guess(:, end);
    end
    
    % Count inliers:(dlt method and p3p method guess 1)
    projected_points = projectPoints(...
        (R_C_W_guess(:,:,1) * matchedLandmarks) + ...
        repmat(t_C_W_guess(:,:,1), ...
        [1 size(matchedLandmarks, 2)]), K);
    difference = matchedCurrKeypoints - projected_points;
    errors = sum(difference.^2, 1);
    is_inlier = errors < pixel_tolerance^2;
    %%end dlt section.
    
    
    %%%p3p inlier section for pose guess 2 
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
    %given found inliers - generate new pose:
    M_C_W = estimatePoseDLT(...
        matchedCurrKeypoints(:, inlier_mask>0)', ...
        matchedLandmarks(:, inlier_mask>0)', K);
    R_C_W = M_C_W(:, 1:3);
    t_C_W = M_C_W(:, end);
end

% back to [V;U]
matchedCurrKeypoints = flipud(matchedCurrKeypoints);

currState = [matchedCurrKeypoints; matchedLandmarks];
%%%%%%%%shows as pose difference(meaning difference from last frame? i
%%%%%%%%think this is just the current pose):
poseDifference = [R_C_W, t_C_W];

%% START OF TRIANGULATION PART

% Retrieves the Descriptors % Keypoints without a Landmark-match

unMatchedIndices = ~ismember(currKeypoints', matchedCurrKeypoints','rows');
% [V;U]
currKeypoints = currKeypoints(:,unMatchedIndices);
currDescriptors = currDescriptors(:,unMatchedIndices);


%First time running processFrame
if isempty(tempState)
    %appends current image information to tempState
    tempState = [double(currKeypoints);...
        im2double(currDescriptors);...
        repmat(poseDifference(:),1,length(currKeypoints(1,:)))];
    
    currPose = poseDifference;
    
else
    %A previous set of keypoints exists, now we need to determine which of
    %these have matches with the new keypoints: If a match is found the
    %keypoint is kept, if no match it is discarded. The new Keypoints that
    %are matched are kept for triangulation, the others are kept for future
    %triangulation
 %commented out alex   
%     currPose = [reshape(tempState(end-11:end,end),3,4);0,0,0,1]... %add last row to make 4X4 for multiply by pose diff then remove last row in next step
%         * [poseDifference;0,0,0,1];
%     currPose = currPose(1:3,1:4);
%%%%%%%
currPose = poseDifference;
    
    
    prevKeypoints = (tempState(1:2,:));%%pull previous keypoints from tempstate [v;u]
    prevDescriptors = im2uint8(tempState(3:end-12,:));%pull descriptor for each prev keypoint
    prevPose = tempState(end-11:end,:);%pull last pose from temp state
    
    %% Find Matches Between NEW Image and OLD Images
    % Match Descriptors
    
%     matches = matchDescriptors( currDescriptors, prevDescriptors, match_lambda);
%     % [V;U]
%     % For Triangulation
%     matchedCurrKeypoints = currKeypoints(:, matches > 0);
%     matchedCurrDescriptors = currDescriptors(:, matches > 0);
%     % [V;U]
%     % For FUTURE Triangulation
%     unmatchedCurrKeypoints = currKeypoints(:, matches == 0);
%     unmatchedCurrDescriptors = currDescriptors(:, matches == 0);


    matches = matchDescriptors( prevDescriptors, currDescriptors, match_lambda);
    % [V;U]
    % For Triangulation
    matchesList = matches(matches > 0);
    matchedCurrKeypoints = currKeypoints(:, matchesList);
    matchedCurrDescriptors = currDescriptors(:, matchesList);
    % [V;U]
    % For FUTURE Triangulation
    unmatchedCurrKeypoints = currKeypoints(:, ~ismember(1:size(currKeypoints,2),matchesList));
    unmatchedCurrDescriptors = currDescriptors(:, ~ismember(1:size(currKeypoints,2),matchesList));


    % [V;U]
    % ONLY Matched old keypoints are kept:
    
    prevKeypoints = prevKeypoints(:, matches > 0);
    prevDescriptors = prevDescriptors(:, matches > 0);
    prevPose = prevPose(:, matches > 0);
    
    % Calculation the Minimum Pose displacement required for triangulation
    distance2pw = mean(currState(5,:)-prevPose(end,1));
    pose2pose_threshold = distance2pw * pose_dist_threshold;
    
    %% Calculate what Keypoints that can be used for triangulation
    
    % Grouping Old Keypoints with same Pose, Segments = number of same pose
    values = unique(prevPose(end,:));
    
    if(length(values)==1)
        segments = length(prevPose(end,:));
    else
        segments = flipud(histcounts(prevPose(end,:),[values inf])')';
    end
    
    % Might do matching in here
    %% FOR EACH SEGMENT, DO RANSAC IF SIZE>k otherwise discard totally
    new_landmarks = [];
    segment_val = 0;
    
    if(length(segments) > 3)
        segmentsStart = length(segments)-3;
        semgentsEnd = length(segments)-1;
    else
        segmentsStart = 1;
        semgentsEnd = length(segments);
    end
    
%     segmentsStart = 1;
%     semgentsEnd = length(segments);
    
    disp(['Sizes of segments: ' num2str(segments)])
    for i=segmentsStart:semgentsEnd
       
            
        segment_val_min = segment_val + 1;
        segment_val = segments(i) + segment_val;
        
        % Setting up for RANSAC [U;V]
        p1 = flipud(prevKeypoints(:,segment_val_min:segment_val));
        M1 =  reshape(prevPose(:,segment_val_min),3,4);
        p2 = flipud((matchedCurrKeypoints(:,segment_val_min:segment_val)));
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
            num_iterations_8pt = 2000;
            disp(['Frame being triangulated from segment ' num2str(i) ' out of ' num2str(length(segments))]);
            %disp(['Size of p1 ' num2str(size(p1),2) ' p2 ' num2str(size(p2),2)])
            pixel_threshold =1;
            num_inliers_history = zeros(1,num_iterations_8pt);
            max_num_inliers_history = zeros(1,num_iterations_8pt);
            % to fit all candidate points in matrix
            best_guess_history = zeros(3,k*num_iterations_8pt,2);
            
            for ii = 1:num_iterations_8pt
                
                % choose random data from landmarks
                [~, idx] = datasample(p1(1:3,:),k,2,'Replace',true);
                p1_sample = p1(:, idx);
                p2_sample = p2(:, idx);
                %[U;V]
                F_candidate = fundamentalEightPoint_normalized(p1_sample,p2_sample);
                
                %%calculate epipolar line distance
%                 d = epipolarLineDistance(F_candidate,p1,p2);
                %[U;V]
                d = zeros(1,length(p1(1,:)));
                for kk=1:length(p1(1,:))
                    d(kk) = distPoint2EpipolarLine(F_candidate,p1(:,kk),p2(:,kk));
                end
                
                % all relevant elements on diagonal
                inlierind = find(d<pixel_threshold);
                inliercount = length(inlierind);
                
                if inliercount > max(max_num_inliers_history) && inliercount>=8
                    max_num_inliers_history(ii) = inliercount;

                    % use inliers to compute BEST GUESS

                    p1_inliers = p1(:,inlierind);
                    p2_inliers = p2(:,inlierind);
                    %[U;V]
                    best_F_candidate = fundamentalEightPoint_normalized(p1_inliers,p2_inliers);

%                         d_2 = epipolarLineDistance(best_F_candidate,p1,p2);

                    d_2 = zeros(1,length(p1(1,:)));
                    for kk=1:length(p1(1,:))
                        d_2(kk) = distPoint2EpipolarLine(best_F_candidate,p1(:,kk),p2(:,kk));
                    end

                    inlierind_2 = find(d_2<pixel_threshold);
                    %[U;V]
                    best_guess_1 = p1(:,inlierind_2);
                    best_guess_2 = p2(:,inlierind_2);

                elseif inliercount < max(max_num_inliers_history)
                    % set to previous value
                    max_num_inliers_history(ii) = ...
                        max_num_inliers_history(ii-1);
                end
            end % END OF RANSAC LOOP
            % Adding NEW landmarks triangulation needs to be [U;V]
            
%             best_guess_1 = [flipud(best_guess_1(1:2,:));best_guess_1(3,:)];
%             best_guess_2 = [flipud(best_guess_2(1:2,:));best_guess_2(3,:)];

%%%%alex
               guess_diff = best_guess_2-best_guess_1;
               guess_dist = sqrt( guess_diff(1,:).^2+ guess_diff(2,:).^2);
               mean_dist = mean(guess_dist);
               dist_thresh = .2;
               best_guess_1 = best_guess_1(:,guess_dist>dist_thresh*mean_dist);
               best_guess_2 = best_guess_2(:,guess_dist>dist_thresh*mean_dist);
               
            try
                
                P = linearTriangulation(K\best_guess_1,K\best_guess_2,K*M1,K*M2);
                
                    %filter new points:
                    R_C_W = currPose(:,1:3);
                    t_C_W = currPose(:,4);
                    world_pose =-R_C_W'*t_C_W;
                    max_dif = [ 16; 2 ; 80];
                    min_dif = [-19; -8; 5];
                    PosZmax = P(3,:) > world_pose(3,1)+min_dif(3);
                    PosYmax = P(2,:) > world_pose(2,1)+min_dif(2);
                    PosXmax = P(1,:) > world_pose(1,1)+min_dif(1);
                    PosZmin = P(3,:) < world_pose(3,1)+max_dif(3);
                    PosYmin = P(2,:) < world_pose(2,1)+max_dif(2);
                    PosXmin = P(1,:) < world_pose(1,1)+max_dif(1);
                    
                    Pos_count = PosZmax +PosYmax+PosXmax+PosZmin+PosYmin+PosXmin;
                    Pok = Pos_count==6;
                    P = P(:,Pok);
                    
            catch
                disp('No inliers');
            end
            if isempty(new_landmarks)
                %[V;U]
                new_landmarks = [flipud(best_guess_2(1:2,Pok));P(1:3,:)];
            else
                %[V;U]
                new_landmarks = [new_landmarks,...
                    [flipud(best_guess_2(1:2,Pok));P(1:3,:)]];
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
    
end
end
