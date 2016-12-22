%% initialization using two-view geometry for monocular VO

clear all
clc

%% choose dataset

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

addpath(genpath('../all_solns'))

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

% Load outlier-free point correspondences

points1 = detectHarrisFeatures(img_1, 'MinQuality', 0.000001);
points2 = detectHarrisFeatures(img_2, 'MinQuality', 0.000001);

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
     

% Dummy initialization of RANSAC variables
num_iterations = 1000; % chosen by me
pixel_threshold = 1; % specified in pipeline
k = 50; % for non-p3p use

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
% Triangulate a point cloud using the final transformation (R,T)
M1 = K * eye(3,4);
M2 = K * [R_C2_W, T_C2_W];
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
    inliercount = length(inlierind);

    if ii == 1
        max_num_inliers_history(ii) = inliercount;
    elseif ii > 1

        if inliercount > max(max_num_inliers_history)
            max_num_inliers_history(ii) = inliercount;

            % use inliers to compute BEST GUESS

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
            
            
            
            
            
%             for jj = 1:length(p1_sample)
%                 
%                 dist2 = distPoint2EpipolarLine(F_candidate,p1_sample(jj),p2_sample(jj));
%                 if dist2 < pixel_threshold % point is an inlier
%                     % point is an inlier % do something
%                 else
%                     % point is outlier
%                 end
%                 % make history of inlier quantity
%                 % find maximum of history
%                 
%                 % 
%             end
% 
%         end
%         
%         % implementing the eight-point algorithm with RANSAC: proper testing...
%         % extend exercise 5 to work with RANSAC (test with artificial outliers)
% end

%% end of monocular VO initialization





