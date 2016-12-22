%function [firstState,firstLandmarks] = monocular_initialization(img0,img1,ransac,dataset)

% indicate first two images for bootstrapping
% for RANSAC filtering of matched keypoints, specify 0 (no) or 1 (yes)
% for dataset, specify 0 (kitti), 1 (malaga), or 2 (parking)

%% setting up parameters
close all
% Parameters from exercise 3.
global K_parking;
global K_malaga;
global K_kitti;
global harris_patch_size;
global harris_kappa;
global nonmaximum_supression_radius;
global descriptor_radius;
global match_lambda;
global num_keypoints;

switch(dataset)
    case 0
        K = K_kitti;
    case 1
        K = K_malaga;
    case 2
        K = K_parking;
end

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

figure(1);
subplot(2, 1, 1);
imshow(img0);
subplot(2, 1, 2);
imagesc(harris0);
axis equal;
axis off;

figure(2);
imshow(img0);
hold on;
plot(keypoints0(2, :), keypoints0(1, :), 'rx', 'Linewidth', 2);

figure(3);
for i = 1:16
    subplot(4, 4, i);
    patch_size = 2 * descriptor_radius + 1;
    imagesc(uint8(reshape(descriptors0(:,i), [patch_size patch_size])));
    axis equal;
    axis off;
end

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

figure(4);
subplot(1,2,1)
imshow(img0);
hold on;
plot(keypoint_matches0(1,:), keypoint_matches0(2, :), 'rx', 'Linewidth', 2);
hold on
plotMatches(all_matches,keypoints1,keypoints0);

subplot(1,2,2)
imshow(img1);
hold on;
plot(keypoint_matches1(1,:),keypoint_matches1(2,:),'gx','Linewidth',2);


% %% computer vision toolbox
%     % Load outlier-free point correspondences
% 
%     points1 = detectHarrisFeatures(img0, 'MinQuality', 0.000001);
%     points2 = detectHarrisFeatures(img1, 'MinQuality', 0.000001);
% 
%     %points1 = detectHarrisFeatures(img0);
%     %points2 = detectHarrisFeatures(img1);
% 
%     % derivation of descriptors
%     [features1,valid_points1] = extractFeatures(img0,points1);
%     [features2,valid_points2] = extractFeatures(img1,points2);
% 
%     % compute indices of the matching features in the two input sets
%     indexPairs = matchFeatures(features1,features2);
% 
%     matchedPoints1 = valid_points1(indexPairs(:,1),:);
%     matchedPoints2 = valid_points2(indexPairs(:,2),:);
% 
%     % begin debug
%     figure(1)
%     showMatchedFeatures(img0,img1,matchedPoints1,matchedPoints2,...
%                             'montage','PlotOptions',{'ro','go','y--'});
%     legend('matched points 1','matched points 2');
%     % end debug
% 
%     % make homogeneous coordinates
%     p1 = [matchedPoints1.Location ones(matchedPoints1.Count,1)]';
%     p2 = [matchedPoints2.Location ones(matchedPoints2.Count,1)]';

%  % debug
%         
%         subplot(1,2,1)
%         imshow(img0)
%         hold on
%         scatter(p1(1,:),p1(2,:))
%         subplot(1,2,2)
%         imshow(img1)
%         hold on
%         scatter(p2(1,:),p2(2,:))

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
    num_iterations = 1000; % chosen by me
    pixel_threshold = 1; % specified in pipeline
    k = 50; % choose k random landmarks

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
        [landmark_sample, idx] = datasample(P(1:3,:),k,2,'Replace',false);
        p1_sample = p0(:,idx);
        p2_sample = p1(:,idx);

        % how many sample points do I need? 
        % in exercise: done with landmark indices...

        F_candidate = fundamentalEightPoint_normalized(p1_sample,p2_sample);
        % E_candidate = estimateEssentialMatrix(p1_sample,p2_sample,K,K);

        % calculate epipolar line distance

        d = diag(epipolarLineDistance(F_candidate,p1_sample,p2_sample));

        % all relevant elements on diagonal
        inlierind = find(d < pixel_threshold);
        inliercount = length(inlierind);

        if ii == 1
            max_num_inliers_history(ii) = inliercount;

            counter = 1;
        elseif ii > 1
        % maxcount = maxcount + 1;

            if inliercount > max(max_num_inliers_history)
                max_num_inliers_history(ii) = inliercount;

                % use coefficients from before as BEST GUESS
                % best_guess_history(:,ii) = polydata;

                % use inliers to compute BEST GUESS

                % [row_1,col_1] = ind2sub(size(d),inlierind);
                p1_inliers = p1_sample(:,inlierind);
                p2_inliers = p2_sample(:,inlierind);

                d_2 = diag(epipolarLineDistance(F_candidate,...
                            p1_inliers,p2_inliers));
                inlierind_2 = find(d_2 < pixel_threshold);        
                %[row_2,col_2] = ind2sub(size(d_2),inlierind_2);

                best_guess_1 = p1_inliers(:,inlierind_2);
                best_guess_2 = p2_inliers(:,inlierind_2);  

             

            elseif inliercount <= max(max_num_inliers_history)
                % set to previous value
                max_num_inliers_history(ii) = ...
                                         max_num_inliers_history(ii-1);

            end
            % counter = counter + 1;
        end

    % set inliers back to zero
    % inliers = zeros(,30);


    end        
        
end

      



% figure(2)
% plot(1:num_iterations,max_num_inliers_history)
% title('Convergence of inlier quantity')
% ylim([0 k]); xlim([0 num_iterations]);
% grid on
% 
% figure(3)
% scatter(best_guess_1(1,:)',best_guess_1(2,:)')
% hold on 
% scatter(best_guess_2(1,:)',best_guess_2(2,:)')
% legend('matched points 1','matched points 2');   
% title('Feature match with RANSAC')
            
%% end of monocular VO initialization





