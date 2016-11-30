%% initialization using two-view geometry for monocular VO

clear all
clc

%% choose method/RANSAC/dataset

% choose method of computation (choose one by commenting the other)
% method = 'self'; 
method = 'toolbox';

% choose whether RANSAC should be used to filter outliers
ransac = 'yes';
%ransac = 'no';

% choose dataset
%dataset = 'parking';
dataset = 'kitti';
%dataset = 'malaga';

%% setting up paths and parameters

% Parameters from exercise 3.
harris_patch_size = 9;
harris_kappa = 0.08;
nonmaximum_supression_radius = 8;
descriptor_radius = 9;
match_lambda = 5;

% Other parameters.
num_keypoints = 300; % with number of keypoints or threshold?

switch(dataset)
    case 'parking'  
        img_1 = rgb2gray(imread('../parking/images/img_00000.png'));
        img_2 = rgb2gray(imread('../parking/images/img_00004.png'));
        K = load('../parking/K.txt');
        % poses = load('../parking/poses.txt');
    case 'kitti' % already grayscale
        img_1 = imread('../kitti/00/image_0/000000.png');
        img_2 = imread('../kitti/00/image_0/000003.png');
        K = [7.188560000000e+02 0 6.071928000000e+02
             0 7.188560000000e+02 1.852157000000e+02
             0 0 1];
         % poses
    case 'malaga'
        img_1 = rgb2gray(imread('../malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_800x600_Images/img_CAMERA1_1261229981.580023_left.jpg'));
        img_2 = rgb2gray(imread('../malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_800x600_Images/img_CAMERA1_1261229981.780037_left.jpg'));
        K = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
end

%% establish keypoint correspondences between these two frames

switch(method)
    case 'self' % according to self-written functions
        
        harris_scores_1 = harris(img_1, harris_patch_size, harris_kappa);
        keypoints_1 = selectKeypoints(...
                harris_scores_1, num_keypoints, nonmaximum_supression_radius);
        descriptors_1 = describeKeypoints(...
                img_1, keypoints_1, descriptor_radius);

        harris_scores_2 = harris(img_2, harris_patch_size, harris_kappa);
        keypoints_2 = selectKeypoints(...
                harris_scores_2, num_keypoints, nonmaximum_supression_radius);
        descriptors_2 = describeKeypoints(img_2, keypoints_2, descriptor_radius);

        % find all matches between the frames
        all_matches = matchDescriptors(descriptors_2, descriptors_1, match_lambda);
        
        keypoint_matches_1 = keypoints_1(:, all_matches > 0);
        keypoint_matches_2 = keypoints_2(:, all_matches > 0);
        
        p1 = [keypoint_matches_1;...
              ones(1,size(keypoint_matches_1,2))];
        p2 = [keypoint_matches_2;...
             ones(1,size(keypoint_matches_2,2))];
     
        %corresponding_matches = all_matches(all_matches > 0);
        %corresponding_landmarks = p_W_landmarks(:, corresponding_matches);
         
    case 'toolbox' % computer vision toolbox
        % Load outlier-free point correspondences

        points1 = detectHarrisFeatures(img_1, 'MinQuality', 0.0001);
        points2 = detectHarrisFeatures(img_2, 'MinQuality', 0.0001);

        %points1 = detectHarrisFeatures(img_1);
        %points2 = detectHarrisFeatures(img_2);
        
        % derivation of descriptors
        [features1,valid_points1] = extractFeatures(img_1,points1);
        [features2,valid_points2] = extractFeatures(img_2,points2);

        % compute indices of the matching features in the two input sets
        indexPairs = matchFeatures(features1,features2);

        matchedPoints1 = valid_points1(indexPairs(:,1),:);
        matchedPoints2 = valid_points2(indexPairs(:,2),:);
        
        % begin debug
        figure(1)
        showMatchedFeatures(img_1,img_2,matchedPoints1,matchedPoints2,...
                                'montage','PlotOptions',{'ro','go','y--'});
        legend('matched points 1','matched points 2');
        % end debug
        
        % make homogeneous coordinates
        p1 = [matchedPoints1.Location ones(matchedPoints1.Count,1)]';
        p2 = [matchedPoints2.Location ones(matchedPoints2.Count,1)]';
     
end

%  % debug
%         
%         subplot(1,2,1)
%         imshow(img_1)
%         hold on
%         scatter(p1(1,:),p1(2,:))
%         subplot(1,2,2)
%         imshow(img_2)
%         hold on
%         scatter(p2(1,:),p2(2,:))

switch(ransac)
    case 'no'
        % Estimate the essential matrix E using the 8-point algorithm
        E = estimateEssentialMatrix(p1, p2, K, K);

        % Extract the relative camera positions (R,T) from the essential matrix
        % Obtain extrinsic parameters (R,t) from E
        [Rots,u3] = decomposeEssentialMatrix(E);

        % Disambiguate among the four possible configurations
        [R_C2_W,T_C2_W] = disambiguateRelativePose(Rots,u3,p1,p2,K,K);
        % R_C2_W = Rots(:,:,2);
        % T_C2_W = u3;

        % Triangulate a point cloud using the final transformation (R,T)
        M1 = K * eye(3,4);
        M2 = K * [R_C2_W, T_C2_W];
        % homogeneous representation of 3D coordinates
        P = linearTriangulation(p1,p2,M1,M2); 
    
       
        % sanity check
       scatter3(P(1,:), P(2,:), P(3,:), 'o')
        
    case 'yes'

        % Dummy initialization of RANSAC variables
        num_iterations = 100;
        pixel_threshold = 1;
        k = 6; % for non-p3p use
        
        % RANSAC implementation
        % use the epipolar line distance to discriminate inliers from
        % outliers; specifically: for a given candidate fundamental matrix
        % F --> a point correspondence should be considered an inlier if
        % the epipolar line distance is less than a threshold. (here 1
        % pixel)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % get first essential/fundamental matrix and calculate landmarks
        
        % Estimate the essential matrix E using the 8-point algorithm
        E = estimateEssentialMatrix(p1, p2, K, K);

        % Extract the relative camera positions (R,T) from the essential matrix
        % Obtain extrinsic parameters (R,t) from E
        [Rots,u3] = decomposeEssentialMatrix(E);

        % Disambiguate among the four possible configurations
        [R_C2_W,T_C2_W] = disambiguateRelativePose(Rots,u3,p1,p2,K,K);
        % R_C2_W = Rots(:,:,2);
        % T_C2_W = u3;

        % Triangulate a point cloud using the final transformation (R,T)
        M1 = K * eye(3,4);
        M2 = K * [R_C2_W, T_C2_W];
        P = linearTriangulation(p1,p2,M1,M2);
        
        figure(2)
        scatter3(P(1,:), P(2,:), P(3,:), '.');
        
        for ii = 1:num_iterations
                    
            % choose random data from landmarks
            [landmark_sample, idx] = datasample(P, k, 2, 'Replace', false);
            p1_sample = p1(:, idx);
        
            F_candidate = fundamentalEightPoint_normalized(p1_sample,p2);
            E_candidate = estimateEssentialMatrix(p1,p2,K,K);
                               
            
                
                
            % calculate epipolar line distance
            % dist1 = epipolarLineDistance(F_candidate,p1(:,jj),p2(:,jj));
            dist2 = distPoint2EpipolarLine(F_candidate,p1,p2); % from exercises
                
            if dist2 < pixel_threshold % candidate is an inlier
                    

            end
        end
        
        % implementing the eight-point algorithm with RANSAC: proper testing...
        % extend exercise 5 to work with RANSAC (test with artificial outliers)
end

%% end of monocular VO initialization





