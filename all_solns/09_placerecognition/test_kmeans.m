%% this script test the kmeans_cluster function with several dataset
clc; close all; clear;

rng default; % For reproducibility

%% sample data from matlab
X = [randn(100,2)*0.75+ones(100,2);
    randn(100,2)*0.5-ones(100,2)];
k = 2;
[idx, C] = kmeans(X, k);
fig1 = vis_cluster_2d(X, idx, C);

%% circle data
N = 200;
k = 2;
rand_theta1 = rand(N, 1) * 2 * pi;
rand_r1 = rand(N, 1); % 0~1
rand_theta2 = rand(N, 1) * 2 * pi;
rand_r2 = rand(N, 1) + 2; % 3~4
X = [rand_r1.*cos(rand_theta1), rand_r1.*sin(rand_theta1);
    rand_r2.*cos(rand_theta2), rand_r2.*sin(rand_theta2)];

[idx, C] = kmeans(X, k);
fig2 = vis_cluster_2d(X, idx, C);

%% unbalanced data
k = 3;
N = 500;
X = [randn(N,2)*0.75+ones(N,2);
    randn(N/5,2)*0.5-2*ones(N/5,2);
    randn(N/25,2)*0.3+[(-2)*ones(N/25,1), ones(N/25,1)]];
[idx, C] = kmeans(X, k);
fig3 = vis_cluster_2d(X, idx, C);