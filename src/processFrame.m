function [ currState, currPose, dataBase] = processFrame(...
    prevState, prevImage, currImage, K, dataBase)
% prevState is a 5xk matrix where the columns corespond to 2D points on top
% of the coresponding 3d points. k is the number of keypoints/landmarks

global harris_patch_size;
global harris_kappa;
global num_keypoints;
global nonmaximum_supression_radius;
global descriptor_radius;
global match_lambda;
global triangulationTolerance;
global p3pIterations;
global p3pTolerance;
global p3pSample;
global triangulationIterations;
global triangulationSample;
global triangulationRansac;

global max_dif;
global min_dif;

global dataBaseSize;


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

%% P3P RANSAC

% [U;V]
matchedCurrKeypoints = flipud(matchedCurrKeypoints); 
max_num_inliers_history = zeros(1, p3pIterations);
max_num_inliers = 0;

for i = 1:p3pIterations
    [landmark_sample, idx] = datasample(matchedLandmarks, p3pSample, 2, 'Replace', false);
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
    inliers = errors < p3pTolerance^2;
    guess = 1;
    
    % Count inliers for guess 2
    projected_points = projectPoints(...
        (R_C_W_guess(:,:,2) * matchedLandmarks) + repmat(t_C_W_guess(:,:,2), ...
        [1 size(matchedLandmarks, 2)]), K);
    difference = matchedCurrKeypoints - projected_points;
    errors = sum(difference.^2, 1);
    inliers_guess_2 = errors < p3pTolerance^2;
    
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

     currState = [];
     currPose = [];
     return

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
new_landmarks = [];

emptyColumns = find(cellfun(@isempty,dataBase(1,:)));

if(isempty(emptyColumns))
    dataBaseLength = dataBaseSize;
else
    dataBaseLength = min(emptyColumns) - 1;
end

%First time running processFrame
if isempty(dataBase{1,1})
    disp('Can''t triangulate new points this frame.')
    dataBase{1,1} = currTriKeypoints; % 2xM keypoints
    dataBase{2,1} = currTriDescriptors; % NxM descriptors
    dataBase{3,1} = currPose(:); % 12x1 pose
else
    
    % Loop through data base but not last frame
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
        
        if size(p1,2)>=8
            
            p1_hom = [p1; ones(1,size(p1,2))];
            p2_hom = [p2; ones(1,size(p2,2))];

            if (~triangulationRansac)

                F_candidate = fundamentalEightPoint_normalized(p1_hom,p2_hom);

                d = (epipolarLineDistance(F_candidate,p1_hom,p2_hom));
                inlierIndx = find(d<triangulationTolerance);

                % Estimate the essential matrix E using the 8-point algorithm
                E = estimateEssentialMatrix(p1_hom, p2_hom, K, K);
                % Extract the relative camera positions (R,T) from the essential matrix
                % Obtain extrinsic parameters (R,t) from E
                [Rots,u3] = decomposeEssentialMatrix(E);
                % Disambiguate among the four possible configurations
                [R_C2_W,T_C2_W] = disambiguateRelativePose(Rots,u3,p1_hom,p2_hom,K,K);
                % Triangulate a point cloud using the final transformation (R,T)
                M1 = K*reshape(prevTriPose,3,4);
                M2 = [[R_C2_W, T_C2_W];0,0,0,1]*[reshape(prevTriPose,3,4);0,0,0,1];
                M2 = K*M2(1:3,1:4);

                P = linearTriangulation(p1_hom(:,inlierIndx),p2_hom(:,inlierIndx),M1,M2); %[U;V]
                triangulated_keypoints = p2(:,inlierIndx);

            else

                % Dummy initialization of RANSAC variables
                max_num_inliers_history = -1*ones(1,triangulationIterations);

                % Estimate the essential matrix E using the 8-point algorithm
                E = estimateEssentialMatrix(p1_hom, p2_hom, K, K);
                % Extract the relative camera positions (R,T) from the essential matrix
                % Obtain extrinsic parameters (R,t) from E
                [Rots,u3] = decomposeEssentialMatrix(E);
                % Disambiguate among the four possible configurations
                [R_C2_W,T_C2_W] = disambiguateRelativePose(Rots,u3,p1_hom,p2_hom,K,K);
                % Triangulate a point cloud using the final transformation (R,T)
                M1 = K*reshape(prevTriPose,3,4);
                M2 = [[R_C2_W, T_C2_W];0,0,0,1]*[reshape(prevTriPose,3,4);0,0,0,1];
                M2 = K*M2(1:3,1:4);


                P = linearTriangulation(p1_hom,p2_hom,M1,M2); %[U;V]

                for ii = 1:triangulationIterations

                    % choose random data from landmarks
                    [~, idx] = datasample(P(1:3,:),triangulationSample,2,'Replace',false);
                    p1_sample = p1_hom(:,idx);
                    p2_sample = p2_hom(:,idx);

                    F_candidate = fundamentalEightPoint_normalized(p1_sample,p2_sample);

                    % calculate epipolar line distance

                    d = (epipolarLineDistance(F_candidate,p1_hom,p2_hom));

                    % all relevant elements on diagonal
                    inlierind = find(d < triangulationTolerance);
                    inliercount = length(inlierind);

                    if inliercount > max(max_num_inliers_history) && inliercount>=8
                        max_num_inliers_history(ii) = inliercount;
                        F_best = F_candidate;
                    elseif inliercount <= max(max_num_inliers_history)
                        % set to previous value
                        max_num_inliers_history(ii) = ...
                            max_num_inliers_history(ii-1);
                    end
                end

                d = (epipolarLineDistance(F_best,p1_hom,p2_hom));
                inlierIndx = find(d < triangulationTolerance);

                % Estimate the essential matrix E using the 8-point algorithm
                E = estimateEssentialMatrix(p1_hom(:,inlierIndx), p2_hom(:,inlierIndx), K, K);
                % Extract the relative camera positions (R,T) from the essential matrix
                % Obtain extrinsic parameters (R,t) from E
                [Rots,u3] = decomposeEssentialMatrix(E);
                % Disambiguate among the four possible configurations
                [R_C2_W,T_C2_W] = disambiguateRelativePose(Rots,u3,p1_hom(:,inlierIndx),p2_hom(:,inlierIndx),K,K);
                % Triangulate a point cloud using the final transformation (R,T)
                M1 = K*reshape(prevTriPose,3,4);
                M2 = [[R_C2_W, T_C2_W];0,0,0,1]*[reshape(prevTriPose,3,4);0,0,0,1];
                M2 = K*M2(1:3,1:4);

                P = linearTriangulation(p1_hom(:,inlierIndx),p2_hom(:,inlierIndx),M1,M2); %[U;V]
                triangulated_keypoints = p2(:,inlierIndx);

            end

            disp([num2str(size(P,2)) ' New Triangulated points'])

            %[V;U]
            new_landmarks = [new_landmarks,...
                [flipud(triangulated_keypoints);P(1:3,:)]];
        end
    
    end

% Add new Landmarks
currState = [currState,new_landmarks];

%filter new points:
world_pose =-R_C_W'*t_C_W;

%use in R2016b or later
%inFront = R_C_W(3,1:3)*(currState(3:5,:)-world_pose) > 0;

% use in R2016a or earlier
inFront = R_C_W(3,1:3)*(currState(3:5,:)-repmat(world_pose, [1, size(currState,2)])) > 0;

PosZmax = currState(5,:) < world_pose(3)+min_dif(3);
PosYmax = currState(4,:) < world_pose(2)+min_dif(2);
PosXmax = currState(3,:) < world_pose(1)+min_dif(1);

PosZmin = currState(5,:) > world_pose(3)+max_dif(3);
PosYmin = currState(4,:) > world_pose(2)+max_dif(2);
PosXmin = currState(3,:) > world_pose(1)+max_dif(1);
Pos_count = PosZmax+PosYmax+PosXmax+PosZmin+PosYmin+PosXmin+inFront;
Pok = Pos_count==4;
currState = currState(:,Pok);

disp([num2str(size(currState,2)) ' landmarks currently being tracked'])

% Clean up and add to data base
if(isempty(emptyColumns))
    dataBase(:,1) = []; % delete oldest frame
    dataBase{1,dataBaseSize} = unmatchedCurrTriKeypoints; %2xM keypoints
    dataBase{2,dataBaseSize} = unmatchedCurrTriDescriptors; %NxM descriptors
    dataBase{3,dataBaseSize} = currPose(:); %12x1 pose
else
    dataBase{1,min(emptyColumns)} = unmatchedCurrTriKeypoints; %2xM keypoints
    dataBase{2,min(emptyColumns)} = unmatchedCurrTriDescriptors; %NxM descriptors
    dataBase{3,min(emptyColumns)} = currPose(:); %12x1 pose
end
    

end


end
