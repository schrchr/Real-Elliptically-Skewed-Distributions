function X = mvsnrnd(xi, S, lambda, N)
% Multivariate skew normal random samples
%
% Inputs:
%        xi - (r, 1) mean
%        S - (r, r) covaraince matrix
%        lambda - (r, 1) skewness parameter
%        N - (1, 1) number of samples
% 
% Outputs: 
%        X - (N, r) skew normal random samples
%
% created by Christian A. Schroth, 29. June 2020
%
% "Real Elliptically Skewed Distributions and Their Application to Robust Cluster Analysis"
% Christian A. Schroth and Michael Muma, Signal Processing Group, Technische Universität Darmstadt
% submitted to IEEE Transactions on Signal Processing


% N = 1000;
% xi = [0; 0];
% S = [1 0; 0 1];
% lambda = [-5; -5];

    V = mvnrnd(xi, S, N); % multivariate normal distribution with mu = mu, variance = S
    T = abs(randn(N,1)); % half normal distribution with mu = 0, variance = 1
    X = lambda.' .* T + V;

end