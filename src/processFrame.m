function [ currState, surrPose ] = processFrame( prevState, prevImage, currImage )

[height, width] = size(currImage);

% Randomly chosen parameters that seem to work well
harris_patch_size = 9;
harris_kappa = 0.08; % Magic number in range (0.04 to 0.15)
num_keypoints = 200;
nonmaximum_supression_radius = 8;
descriptor_radius = 9;
match_lambda = 4;

%% Part 1 - Calculate Harris scores
harris_scores = harris(prevImage, harris_patch_size, harris_kappa);
assert(min(size(harris_scores) == size(prevImage)));

% figure(1);
% subplot(2, 1, 1);
% imshow(img);
% subplot(2, 1, 2);
% imagesc(harris_scores);
% axis equal;
% axis off;

%% Part 2 - Select keypoints
keypoints = selectKeypoints(...
    harris_scores, num_keypoints, nonmaximum_supression_radius);

% figure(2);
% imshow(prevImage);
% hold on;
% plot(keypoints(2, :), keypoints(1, :), 'rx', 'Linewidth', 2);

%% Part 3 - Describe keypoints and show 16 strongest keypoint descriptors
descriptors = describeKeypoints(prevImage, keypoints, descriptor_radius);

% figure(3);
% for i = 1:16
%     subplot(4, 4, i);
%     patch_size = 2 * descriptor_radius + 1;
%     imagesc(uint8(reshape(descriptors(:,i), [patch_size patch_size])));
%     axis equal;
%     axis off;
% end

%% Part 4 - Match descriptors between first two images
harris_scores_2 = harris(currImage, harris_patch_size, harris_kappa);
keypoints_2 = selectKeypoints(...
    harris_scores_2, num_keypoints, nonmaximum_supression_radius);
descriptors_2 = describeKeypoints(currImage, keypoints_2, descriptor_radius);

matches = matchDescriptors(descriptors_2, descriptors, match_lambda);

% figure(4);
% imshow(currImage);
% hold on;
% plot(keypoints_2(2, :), keypoints_2(1, :), 'rx', 'Linewidth', 2);
% plotMatches(matches, keypoints_2, keypoints);

end

