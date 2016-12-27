function [ currState, currPose, dataBase] = processFrame(...
    prevState, prevImage, currImage, K, dataBase)
% prevState is a 5xk matrix where the columns corespond to 2D points on top
% of the coresponding 3d points. k is the number of keypoints/landmarks

addpath(genpath('../../all_solns'))

global harris_patch_size;
global harris_kappa;
global num_keypoints;
global nonmaximum_supression_radius;
global descriptor_radius;
global match_lambda;
global pixel_threshold;

num_iterations = 500;
pixel_tolerance = 3;
k = 3;


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
% [U;V]
matchedCurrKeypoints = flipud(matchedCurrKeypoints); 
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

%% Find unmatched keypoints

% Retrieves the Descriptors and Keypoints without a Landmark-match
unMatchedIndices = ~ismember(currKeypoints', matchedCurrKeypoints','rows');
% [V;U]
currTriKeypoints = currKeypoints(:,unMatchedIndices);
currTriDescriptors = currDescriptors(:,unMatchedIndices);


%% START OF TRIANGULATION PART
disp(['Sizes of segments: ' size(dataBase)])
new_landmarks = [];

emptyColumns = find(cellfun(@isempty,dataBase(1,:)));

if(isempty(emptyColumns))
    dataBaseLength = 5;
else
    dataBaseLength = min(emptyColumns) - 1;
end

%First time running processFrame
if isempty(dataBase{1,1})
    dataBase{1,1} = currTriKeypoints; % 2xM keypoints
    dataBase{2,1} = currTriDescriptors; % NxM descriptors
    dataBase{3,1} = currPose(:); % 12x1 pose
else
    
    % Loop through whole data base
    for i=1:dataBaseLength
        
        disp(['Frame being triangulated from segment ' num2str(i)]);

        %After first time
        prevTriKeypoints = dataBase{1,i};%%pull previous keypoints from tempstate [v;u]
        prevTriDescriptors = dataBase{2,i};%pull descriptor for each prev keypoint
        prevTriPose = dataBase{3,i};%pull all previous poses

        matches = matchDescriptors( currTriDescriptors, prevTriDescriptors, match_lambda);
        
        % [V;U]
        % For Triangulation
        matchesList = matches(matches > 0);
        matchedCurrTriKeypoints = currTriKeypoints(:, matches > 0);
        
        % [V;U]
        % ONLY Matched previous keypoints are kept:
        matchedPrevTriKeypoints = prevTriKeypoints(:, matchesList);

        % [V;U]
        % Unmatched current keypoints are saved for future triangulation
        unmatchedCurrTriKeypoints = currTriKeypoints(:, matches==0);
        unmatchedCurrTriDescriptors = currTriDescriptors(:, matches==0);

        % [U;V]
        % Setting up for triangulation 
        p1 = flipud(matchedPrevTriKeypoints);
        p2 = flipud(matchedCurrTriKeypoints);
        M1 =  K*reshape(prevTriPose,3,4);
        M2 = K*currPose;
        p1_hom = [p1; ones(1,size(p1,2))];
        p2_hom = [p2; ones(1,size(p2,2))];
        
        F_candidate = fundamentalEightPoint_normalized(p1_hom,p2_hom);
        
        d = (epipolarLineDistance(F_candidate,p1_hom,p2_hom));
        inlierIndx = find(d < pixel_threshold);
        
%         showMatchedFeatures(prevImage, currImage, p1(:,inlierIndx)',p2(:,inlierIndx)')
        
%         normalized_p1 = K\p1_hom;%database
%         normalized_p2 = K\p2_hom;%query
%         
%         pose_p2 = R_C_W*normalized_p2;
%         
%         bearing_angles = atan2(norm(cross(normalized_p1,pose_p2)), dot(normalized_p1,pose_p2));
%         bearing_angles_deg = bearing_angles.*180./pi;    
%         ang_thrsh = 62;

        

        P = linearTriangulation(p1_hom(:,inlierIndx),p2_hom(:,inlierIndx),M1,M2); %[U;V]
        triangulated_keypoints = p2(:,inlierIndx);
        
        %filter new points:
        world_pose =-R_C_W'*t_C_W;
        max_dif = [ 16; 2 ; 80];
        min_dif = [-19; -8; 5];
        PosZmax = P(3,:) > world_pose(3)+min_dif(3);
        PosYmax = P(2,:) > world_pose(2)+min_dif(2);
        PosXmax = P(1,:) > world_pose(1)+min_dif(1);
        PosZmin = P(3,:) < world_pose(3)+max_dif(3);
        PosYmin = P(2,:) < world_pose(2)+max_dif(2);
        PosXmin = P(1,:) < world_pose(1)+max_dif(1);
        disp([num2str(size(P,2)) ' New Triangulated points'])
        Pos_count = PosZmax +PosYmax+PosXmax+PosZmin+PosYmin+PosXmin;
        Pok = Pos_count==6;
        P = P(:,Pok);
        triangulated_keypoints = triangulated_keypoints(:,Pok);
        
        
       
        %[V;U]
        new_landmarks = [new_landmarks,...
            [flipud(triangulated_keypoints);P(1:3,:)]];
    end
    


% Add new Landmarks
currState = [currState,new_landmarks];

% Clean up and add to data base
if(isempty(emptyColumns))
    dataBase(:,1) = []; % delete oldest frame
    dataBase{1,5} = unmatchedCurrTriKeypoints; %2xM keypoints
    dataBase{2,5} = unmatchedCurrTriDescriptors; %NxM descriptors
    dataBase{3,5} = currPose(:); %12x1 pose
else
    dataBase{1,min(emptyColumns)} = unmatchedCurrTriKeypoints; %2xM keypoints
    dataBase{2,min(emptyColumns)} = unmatchedCurrTriDescriptors; %NxM descriptors
    dataBase{3,min(emptyColumns)} = currPose(:); %12x1 pose
end
    
end


end
