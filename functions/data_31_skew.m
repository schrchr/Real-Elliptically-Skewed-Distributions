function [data_final, labels, r, N, K_true, mu_true, scatter_true, lambda_true] = data_31_skew(N_k, epsilon)
% Data-3.1, no imbalance
%
% Inputs: 
%        N_k: number of data vectors per cluster
%        epsilon: percentage of replacement outlieres
%        p: plot true/false

% Outputs:
%        data - the generated data set
%        r - number of features in the generated data set
%        N - total number of samples in the data set
%        K_true - true number of clusters in the data set
%        mu_true 
%        scatter_true 
%        num_clusters
%
% created by Christian A. Schroth, 29. June 2020
%
% "Real Elliptically Skewed Distributions and Their Application to Robust Cluster Analysis"
% Christian A. Schroth and Michael Muma, Signal Processing Group, Technische Universität Darmstadt
% submitted to IEEE Transactions on Signal Processing


    out_range = [-15 45; -20 30];
    K_true = 3; % number of clusters
    r = 2; % number of features/dimension

    mu_true = zeros(r, K_true);
    mu_true(:,1) = [2; 3.5];
    mu_true(:,2) = [6; 2];
    mu_true(:,3) = [10; 3];

    scatter_true = zeros(r,r, K_true);
    scatter_true(:,:,1) = [0.2 0.1; 0.1 0.75];
    scatter_true(:,:,2) = [0.5 0.25; 0.25 0.5];
    scatter_true(:,:,3) = [1 0.5; 0.5 1];
    
    lambda_true = zeros(r, K_true);
    lambda_true(:,1) = [10; 4];
    lambda_true(:,2) = [1; -2];
    lambda_true(:,3) = [2; 1];

    N_k_factor = [5 4 1];
    N = sum(N_k .*N_k_factor); % total number of data points

    data = []; 
    for k = 1:K_true
        data = [data ;[ones(N_k*N_k_factor(k),1)*k, mvsnrnd(mu_true(:,k), scatter_true(:,:,k), lambda_true(:,k), N_k*N_k_factor(k))]];
    end
    
    % randomly permute data
    data = data(randperm(N), :);

    % replacement outlier
    N_repl = round(N * epsilon);
    index_repl = randperm(N,N_repl).';

    data_rpl = [];
    for ir = 1:r
        data_rpl = [data_rpl, rand(N_repl,1)*(out_range(ir,2) - out_range(ir,1)) + out_range(ir,1)];
    end
    data(index_repl,:) = [ones(N_repl,1)*(K_true+1), data_rpl];
    
    labels = data(:,1);
    data_final = data(:,2:end);
end