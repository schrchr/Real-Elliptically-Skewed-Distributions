function psi = psi_t(t, r, nu)
% Computes psi(t) of the t distribution
%
% Inputs:
%       t  - (N, 1) squared Mahalanobis distance
%       r  - (1, 1) dimension
%       nu - (1, 1) degree of freedom
%
% Outputs:
%       psi - (N, 1) psi(t) of t distribution
%
% created by Christian A. Schroth, 29. June 2020
%
% "Real Elliptically Skewed Distributions and Their Application to Robust Cluster Analysis"
% Christian A. Schroth and Michael Muma, Signal Processing Group, Technische Universität Darmstadt
% submitted to IEEE Transactions on Signal Processing

    psi = 0.5 .* (nu + r)./(nu + t);
end