function rho = rho_gaus(t, r)
% Computes rho(t) of the Gaussian distribution
%
% Inputs:
%       t  - (N, 1) squared Mahalanobis distance
%       r  - (1, 1) dimension
%
% Outputs:
%       rho - (N, 1) rho(t) of Gaussian distribution
%
% created by Christian A. Schroth, 29. June 2020
%
% "Real Elliptically Skewed Distributions and Their Application to Robust Cluster Analysis"
% Christian A. Schroth and Michael Muma, Signal Processing Group, Technische Universität Darmstadt
% submitted to IEEE Transactions on Signal Processing

    rho = r./2.*log(2.*pi) + t./2;
    %res = -log((2.*pi).^(-r./2) .* exp(-t./2)); 
end