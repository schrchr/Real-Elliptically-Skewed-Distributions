function t = mahalanobisDistance(x, mu, S)
% Computes the squared Mahalanobis distance.
%
% Inputs:
%       x - (N, r) data, r - dimension
%       mu - (r, 1) cluster centroid
%       S - (r, r) cluster scatter matrix
%
% Outputs:
%       t - (N, 1) squared Mahalanobis distances
%
% created by Christian A. Schroth, 29. June 2020
%
% "Real Elliptically Skewed Distributions and Their Application to Robust Cluster Analysis"
% Christian A. Schroth and Michael Muma, Signal Processing Group, Technische Universität Darmstadt
% submitted to IEEE Transactions on Signal Processing

%% Method 1
%     S_inv = inv(S);
%     N = length(x);
%     t = zeros(N,1);
%     for n = 1:N
%         t(n,1) = (x(n,:).' - mu).' * S_inv * (x(n,:).' - mu);
%     end
 
%% Method 2
    %try
        t = dot(((x.' - mu).' / S).',(x.' - mu), 1).';
    %catch
     %   disp("illConditionedMatrix")
    %end

%% Method 3, fast, but not precise
%     S_inv = inv(S);
%     x_centered = (x - mu.').';
%     [L, ~] = chol(S_inv,'lower');
%     t = dot(L*x_centered, L*x_centered, 1).';  

end