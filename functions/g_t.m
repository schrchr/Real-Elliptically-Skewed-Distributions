function res = g_t(t, r, nu)
% Computes g(t) of the t distribution
%
% Inputs:
%       t  - (N, 1) squared Mahalanobis distance
%       r  - (1, 1) dimension
%       nu - (1, 1) degree of freedom
%
% Outputs:
%       g - (N, 1) g(t) of t distributions
%
% created by Christian A. Schroth, 29. June 2020
%
% "Real Elliptically Skewed Distributions and Their Application to Robust Cluster Analysis"
% Christian A. Schroth and Michael Muma, Signal Processing Group, Technische Universität Darmstadt
% submitted to IEEE Transactions on Signal Processing
    
    res = gamma((nu + r)/2)/(gamma(nu/2)*(pi*nu).^(r/2)) * (1 + t/nu).^(-(nu+r)/2);
end