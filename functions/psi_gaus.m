function psi = psi_gaus(t)
% Computes psi(t) of the Gaussian distribution
%
% Inputs:
%       t  - (N, 1) squared Mahalanobis distance
%
% Outputs:
%       psi - (N, 1) psi(t) of Gaussian distribution
%
% created by Christian A. Schroth, 29. June 2020
%
% "Real Elliptically Skewed Distributions and Their Application to Robust Cluster Analysis"
% Christian A. Schroth and Michael Muma, Signal Processing Group, Technische Universität Darmstadt
% submitted to IEEE Transactions on Signal Processing
    
    psi = 0.5 * ones(length(t),1);
end