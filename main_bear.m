clear all;
close all;
rng(1);

% Create data for parts 1 and 2
num_inliers = 20;
num_outliers = 10;
noise_ratio = 0.1;
poly = rand(3, 1); % random second-order polynomial
extremum = -poly(2)/(2*poly(1));
xstart = extremum - 0.5;
lowest = polyval(poly, extremum);
highest = polyval(poly, xstart);
xspan = 1;
yspan = highest - lowest;
max_noise = noise_ratio * yspan;
x = rand(1, num_inliers) + xstart;
y = polyval(poly, x);
y = y + (rand(size(y))-.5) * 2 * max_noise;
data = [x (rand(1, num_outliers) + xstart)
    y (rand(1, num_outliers) * yspan + lowest)];

% Data for parts 3 and 4
K = load('data/K.txt');
keypoints = load('data/keypoints.txt')';
p_W_landmarks = load('data/p_W_landmarks.txt')';

% Data for part 4
database_image = imread('data/000000.png');

% Dependencies
addpath('plot');
% Replace the following with the path to your DLT code:
addpath('all_solns\01_pnp');
% Replace the following with the path to your keypoint matcher code:
addpath('all_solns\02_detect_describe_match');

figure(5);
subplot(1, 3, 3);
scatter3(p_W_landmarks(1, :), p_W_landmarks(2, :), p_W_landmarks(3, :), 5);
set(gcf, 'GraphicsSmoothing', 'on');
view(0,0);
axis equal;
axis vis3d;
axis([-15 10 -10 5 -1 40]);
angle = zeros(4,600);
angle_degangle = zeros(4,600);
for i = 0:4
    query_image = imread(sprintf('data/%06d.png',i));
    database_image = imread(sprintf('data/%06d.png',0));
    
    [R_C_W, t_C_W, query_keypoints, all_matches, inlier_mask] = ...
    ransacLocalization(query_image, database_image,  keypoints, ...
    p_W_landmarks, K);

    matched_query_keypoints = query_keypoints(:, all_matches > 0);
    corresponding_matches = all_matches(all_matches > 0);
    
   %%%%%adding  Triangulate
   PW_matches = p_W_landmarks(:, corresponding_matches);
    DB_matches = keypoints(:, corresponding_matches);
    DB_inliers = flipud(DB_matches(:,inlier_mask));
    inlier_qp = flipud(matched_query_keypoints(:,inlier_mask));
       PW_inlier = PW_matches(:,inlier_mask);
       
       qp_hom =[inlier_qp; ones(1,size(inlier_qp,2))];
       db_hom = [DB_inliers; ones(1,size(inlier_qp,2))];
    normalized_qp = K\qp_hom;
    normalized_DB = K\db_hom;
    pose_qp = R_C_W*normalized_qp;
    angle(i+1,1:size(normalized_DB,2)) = atan2(norm(cross(normalized_DB,pose_qp)), dot(normalized_DB,pose_qp));
    angle_degangle = angle(i+1,1:size(normalized_DB,2))*180/pi;    
    ang_thrsh = 0;
    Mold = K*eye(3,4);
    Mnew = K*[R_C_W, t_C_W];
    p1_re = Mold * [PW_inlier;ones(1,size(PW_inlier,2))];
    p1_re_hom = [p1_re(1,:)./p1_re(3,:);p1_re(2,:)./p1_re(3,:);p1_re(3,:)./p1_re(3,:)];
    p2_re = Mnew *[PW_inlier;ones(1,size(PW_inlier,2))];
    p2_re_hom = [p2_re(1,:)./p2_re(3,:);p2_re(2,:)./p2_re(3,:);p2_re(3,:)./p2_re(3,:)];
    
    prepdif = qp_hom - p2_re_hom;
    dif_ok = sqrt(prepdif(1,:).^2+prepdif(2,:).^2)<10;
    sumdif = sum(dif_ok)
     % PW_new = linearTriangulation(p1_re_hom,p2_re_hom,Mold,Mnew);
     
     PW_new = linearTriangulation(db_hom,qp_hom,Mold,Mnew);
   %PW_new = linearTriangulation(db_hom(:,angle_degangle>ang_thrsh),qp_hom(:,angle_degangle>ang_thrsh),Mold,Mnew);
    PW_diff = PW_new(1:3,:) - PW_inlier(:,angle_degangle>ang_thrsh);
    PW_diff1 = sqrt(PW_diff(1,:).^2+PW_diff(2,:).^2+PW_diff(3,:).^2);
    figure(8)
    try
     showMatchedFeatures(database_image,query_image,db_hom(1:2,1:20)',qp_hom(1:2,1:20)',...
                                'montage','PlotOptions',{'ro','go','y--'});
    catch
        lol = 0;
    end
     figure(10)
     hold on
     scatter(angle_degangle(:,angle_degangle>ang_thrsh) ,PW_diff1)
    axis([0,60,0,60])
    hold off
%     figure(11)
%     hold on
%     scatter(PW_inlier(3,angle_degangle>ang_thrsh) ,PW_diff1)
%     axis([0,60,0,60])
%     hold off
%      for jj = 1:size(inlier_qp,2)
%             normalized_qp(:, jj) = normalized_qp(:, jj) / ...
%                 norm(normalized_qp(:, jj), 2);
%             normalized_DB(:, jj) = normalized_DB(:, jj) / ...
%                 norm(normalized_DB(:, jj), 2);
%      end       
     
    % Distinguish success from failure.
    if (numel(R_C_W) > 0)
        subplot(1, 3, 3);
        plotCoordinateFrame(R_C_W', -R_C_W'*t_C_W, 2);
        disp(['Frame ' num2str(i) ' localized with ' ...
            num2str(nnz(inlier_mask)) ' inliers!']);
        view(0,0);
    else
        disp(['Frame ' num2str(i) ' failed to localize!']);
    end
    
    subplot(1, 3, [1 2]);
    figure(24)
    imshow(query_image);
    
    hold on;
    plot(matched_query_keypoints(2, (1-inlier_mask)>0), ...
        matched_query_keypoints(1, (1-inlier_mask)>0), 'rx', 'Linewidth', 2);
    if (nnz(inlier_mask) > 0)
        plot(matched_query_keypoints(2, (inlier_mask)>0), ...
            matched_query_keypoints(1, (inlier_mask)>0), 'gx', 'Linewidth', 2);
    end
    
    plotMatches(corresponding_matches(inlier_mask>0), ...
        matched_query_keypoints(:, inlier_mask>0), ...
        keypoints);
    hold off;
    title('Inlier and outlier matches');
    pause(0.01);
%         keypoints = matched_query_keypoints;
%     p_W_landmarks = corresponding_matches;
end