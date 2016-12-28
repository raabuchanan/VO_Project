function [prevState,firstLandmarks] = monocular_intialization(img0,img1,ransac,K)
%function [firstState,firstLandmarks] = monocular_initialization(img0,img1,ransac,dataset)

% indicate first two images for bootstrapping
% for RANSAC filtering of matched keypoints, specify 0 (no) or 1 (yes)
% for dataset, specify 0 (kitti), 1 (malaga), or 2 (parking)

% Parameters from exercise 3.
global harris_patch_size;
global harris_kappa;
global nonmaximum_supression_radius;
global descriptor_radius;
global match_lambda;
global num_keypoints;


%% establish keypoint correspondences between these two frames

% Find Harris scores / keypoints / descriptors of images
harris0 = harris(img0, harris_patch_size, harris_kappa);
assert(min(size(harris0) == size(img0)));
keypoints0 = selectKeypoints(harris0, num_keypoints, nonmaximum_supression_radius);
descriptors0 = describeKeypoints(img0, keypoints0, descriptor_radius);

harris1 = harris(img1, harris_patch_size, harris_kappa);
assert(min(size(harris1) == size(img1)));
keypoints1 = selectKeypoints(harris1, num_keypoints, nonmaximum_supression_radius);
descriptors1 = describeKeypoints(img1, keypoints1, descriptor_radius);


% find all matches between the frames
all_matches = matchDescriptors(descriptors1, descriptors0, match_lambda);
keypoint_matches1 = flipud(keypoints1(:, all_matches > 0));

matchesList = all_matches(all_matches > 0);
keypoint_matches0 = flipud(keypoints0(:, matchesList));

% for later: detectFASTfeatures
% only find those indices that are non-zero, i.e. that fulfill the
% condition (dists < lambda * min_non_zero_dist)

p0 = [keypoint_matches0; ones(1,size(keypoint_matches0,2))];
p1 = [keypoint_matches1; ones(1,size(keypoint_matches1,2))];

%% RANSAC

    if ransac == 0
        % Estimate the essential matrix E using the 8-point algorithm
        E = estimateEssentialMatrix(p0, p1, K, K);

        % Extract the relative camera positions (R,T) from the essential matrix
        % Obtain extrinsic parameters (R,t) from E
        [Rots,u3] = decomposeEssentialMatrix(E);

        % Disambiguate among the four possible configurations
        [R_C1_W,T_C1_W] = disambiguateRelativePose(Rots,u3,p0,p1,K,K);
        %R_C1_W = Rots(:,:,2);
        %T_C1_W = u3;

        % Triangulate a point cloud using the final transformation (R,T)
        M0 = K * eye(3,4); % transformation from frame 0 to frame 0
        M1 = K * [R_C1_W, T_C1_W]; % transformation from frame 0 to frame 1
        % homogeneous representation of 3D coordinates
        P = linearTriangulation(p0,p1,M0,M1);
        % scatter3(P(1,:),P(2,:),P(3,:))

        figure(5)
        scatter3(P(1,:),P(2,:),P(3,:), 20,'filled');
        hold on
        plot3(T_C1_W(1),T_C1_W(2),T_C1_W(3),'rx');
        axis equal;
        axis vis3d;
        grid off;
        xlabel('X');
        ylabel('Y');
        zlabel('Z');

    elseif ransac == 1

        % Dummy initialization of RANSAC variables
        num_iterations = 2000; % chosen by me
        pixel_threshold = 1; % specified in pipeline
        k = 10; % choose k random landmarks

        % RANSAC implementation
        % use the epipolar line distance to discriminate inliers from
        % outliers; specifically: for a given candidate fundamental matrix
        % F --> a point correspondence should be considered an inlier if
        % the epipolar line distance is less than a threshold. (here 1
        % pixel)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % get first essential/fundamental matrix and calculate landmarks

        % Estimate the essential matrix E using the 8-point algorithm
        E = estimateEssentialMatrix(p0, p1, K, K);

        % Extract the relative camera positions (R,T) from the essential matrix
        % Obtain extrinsic parameters (R,t) from E
        [Rots,u3] = decomposeEssentialMatrix(E);

        % Disambiguate among the four possible configurations
        [R_C2_W,T_C2_W] = disambiguateRelativePose(Rots,u3,p0,p1,K,K);

        % Triangulate a point cloud using the final transformation (R,T)
        M0 = K * eye(3,4);
        M1 = K * [R_C2_W, T_C2_W];
        P = linearTriangulation(p0,p1,M0,M1);

        num_inliers_history = zeros(1,num_iterations);
        max_num_inliers_history = zeros(1,num_iterations);

        % to fit all candidate points in matrix
        best_guess_history = zeros(3,k*num_iterations,2);

        for ii = 1:num_iterations

            % choose random data from landmarks
            [~, idx] = datasample(P(1:3,:),k,2,'Replace',false);
            p1_sample = p0(:,idx);
            p2_sample = p1(:,idx);

            % how many sample points do I need?
            % in exercise: done with landmark indices...

            F_candidate = fundamentalEightPoint_normalized(p1_sample,p2_sample);
            % E_candidate = estimateEssentialMatrix(p1_sample,p2_sample,K,K);

            % calculate epipolar line distance

            d = (epipolarLineDistance(F_candidate,p0,p1));

            % all relevant elements on diagonal
            inlierind = find(d < pixel_threshold);
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
        %% COMPUTE NEW MODEL FROM BEST
        d = (epipolarLineDistance(F_best,p0,p1));
        % all relevant elements on diagonal
        inlierind = find(d < pixel_threshold);
        p0 = p0(:,inlierind);
        p1 = p1(:,inlierind);
        % Estimate the essential matrix E using the 8-point algorithm
        E = estimateEssentialMatrix(p0, p1, K, K);
        % Extract the relative camera positions (R,T) from the essential matrix
        % Obtain extrinsic parameters (R,t) from E
        [Rots,u3] = decomposeEssentialMatrix(E);
        % Disambiguate among the four possible configurations
        [R_C2_W,T_C2_W] = disambiguateRelativePose(Rots,u3,p0,p1,K,K);
        % Triangulate a point cloud using the final transformation (R,T)
        M0 = K * eye(3,4);
        M1 = K * [R_C2_W, T_C2_W];
        firstLandmarks = linearTriangulation(p0,p1,M0,M1);

        prevState = [flipud(p1(1:2,:));firstLandmarks];

    end
end

