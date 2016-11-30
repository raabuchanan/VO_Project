function [init_keypoints, init_landmarks] = stereo_init(...
    ds, num_keypoints)

%% Setup
%ds = 0; % 0: KITTI, 1: Malaga, 2: parking
%num_keypoints = 500;%choose number of keypoints to initialize

%%Get first sereo pair of images, and camera parameters from selected data set 
if ds == 0
    %  kitti
    
    left_image = imread( 'kitti\00\image_0\000000.png');  %D:\Matlab\Final_Project\kitti\00\image_0
    right_image = imread( 'kitti\00\image_1\000000.png');

     baseline = 0.54;      

      K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
elseif ds == 1
    %  Malaga 7.
    left_image = rgb2gray(imread( 'malaga-urban-dataset-extract-07\malaga-urban-dataset-extract-07_rectified_800x600_Images\img_CAMERA1_1261229981.580023_left.jpg'));  
    right_image = rgb2gray(imread( 'malaga-urban-dataset-extract-07\malaga-urban-dataset-extract-07_rectified_800x600_Images\img_CAMERA1_1261229981.580023_right.jpg'));
    
    baseline = 0.11947;

    K = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
elseif ds == 2
    %Parking
    
    baseline = 0.11947;%using first two images like a stereo pair
     K = load('parking\K.txt');
   
    
    left_image = rgb2gray(imread( 'parking\images\img_00000.png'));
    right_image = rgb2gray(imread( 'parking\images\img_00001.png'));
else
    assert(false);
end

%%%%%%%%%%%%%%%


%%%find harris corners of first image%%%%
% Randomly chosen parameters that seem to work well 

harris_patch_size = 9;
harris_kappa = 0.08;
nonmaximum_supression_radius = 8;
descriptor_radius = 9;
match_lambda = 4;

img = left_image;
%img = imread('../data/000000.png');

%% Calculate Harris scores

harris_scores = harris(img, harris_patch_size, harris_kappa);
assert(min(size(harris_scores) == size(img)));

% % % figure(1);
% % % subplot(2, 1, 1);
% % % imshow(img);
% % % subplot(2, 1, 2);
% % % imagesc(harris_scores);
% % % axis equal;
% % % axis off;

%% Select keypoints
keypoints = selectKeypoints(...
    harris_scores, num_keypoints, nonmaximum_supression_radius);
% % % figure(2);
% % % imshow(img);
% % % hold on;
% % % plot(keypoints(2, :), keypoints(1, :), 'rx', 'Linewidth', 2);


%% Describe keypoints 

descriptors = describeKeypoints(img, keypoints, descriptor_radius);
%figure(3);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%% Stereo matching %%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Given by the KITTI dataset:

% Carefully tuned by the TAs:
patch_radius = 5;
min_disp = 5;
max_disp = 50;
xlims = [7 20];
ylims = [-6 10];
zlims = [-5 5];

%%  Disparity

%tic;
disp_img = getDisparity_init(...
    left_image, right_image, patch_radius, min_disp, max_disp);
%toc;
% % % figure(1);
% % % imagesc(disp_img);
% % % axis equal;
% % % axis off;



%% Create point cloud 


[p_C_points, intensities] = disparityToPointCloud(...
    disp_img, K, baseline, left_image);


%%%%% map 2D keypoints to selected 3d points

disp_bin = zeros(size(disp_img));
disp_bin(disp_img > 0) = 1; %create matrix with 1 indicating positions with valid disparity
keypoints_sorted = sortrows(keypoints',2)'; %Sort keypoints by column

%find which keypooints also have a valid disparity
v = 1;
for s = 1:num_keypoints
    if disp_bin(keypoints_sorted(1,s),keypoints_sorted(2,s)) >0
        disp_bin(keypoints_sorted(1,s),keypoints_sorted(2,s)) = 2; %set disbin to 2 for locations with matching keypoint
        f_keypoints (:, v) = keypoints_sorted(:,s); %Store keypoints with matching valid disparity
        v = v+1;
    end
end
disp_bin = disp_bin(:)';
disp_bin = disp_bin(disp_bin > 0); %create vector with length equal to num of valid disparities
disp_bin = disp_bin > 1; % Set values positions that dont correspond to a matching keypoint to 0
mtch_pts = find(disp_bin); %find vector of keypoint-disparity match indices
real_w_pts = p_C_points(:,mtch_pts); %Get 3d points that correspond to keypoints


init_keypoints = f_keypoints; % final output of selected initialization keypoints
init_landmarks = real_w_pts;%final output of selected initialization landmarks in camera reference frame

%init_landmarks = [0 -1 0; 0 0 -1; 1 0 0]^-1 *real_w_pts; %landmarks in world reference frame

%Plotting
% key_pt_intesnities = intensities(:,mtch_pts);
% figure(6);
% plotpoints = [0 -1 0; 0 0 -1; 1 0 0]^-1 * real_w_pts;
% scatter3(plotpoints(1, :), plotpoints(2, :), plotpoints(3, :), ...
%     20 * ones(1, length(plotpoints)), ...
%     repmat(single(key_pt_intesnities)'/255, [1 3]), 'filled');
%    axis equal;
% axis([0 30 ylims zlims]);
% axis vis3d;
% grid off;
% xlabel('X');
% ylabel('Y');
% zlabel('Z'); 
% 
% figure(2);
% imshow(img);
% hold on;
% plot(init_keypoints(2, :), init_keypoints(1, :), 'rx', 'Linewidth', 2);
    