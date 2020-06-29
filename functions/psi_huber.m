function psi = psi_huber(t, r, varargin)
% Computes psi(t) of the Huber distribution
%
% Possible Input Combinations:
%       t, r
%       t, r, qH
%       t, r, cH, bH - this option is provided, to improve performance, because it allows to avoid the calculation of
%                      the constants cH and bH in every loop iteration
%
% Inputs:
%       t  - (N, 1) squared Mahalanobis distance
%       r  - (1, 1) dimension
%       qH - (1, 1) tuning parameter, standard value 0.8, choose qH > 0.701
%       cH - (1, 1) tuning parameter
%       bH - (1, 1) constant
%
% Outputs:
%       psi - (N, 1) psi(t) of Huber distributions
%
% created by Christian A. Schroth, 29. June 2020
%
% "Real Elliptically Skewed Distributions and Their Application to Robust Cluster Analysis"
% Christian A. Schroth and Michael Muma, Signal Processing Group, Technische Universität Darmstadt
% submitted to IEEE Transactions on Signal Processing

    if(isempty(varargin))
        qH = 0.8;
        cH = sqrt(chi2inv(qH, r));
        bH = chi2cdf(cH^2,r+2) + cH^2/r*(1-chi2cdf(cH^2,r));
    elseif(length(varargin) == 1)
        qH = varargin{1};
        cH = sqrt(chi2inv(qH, r));
        bH = chi2cdf(cH^2,r+2) + cH^2/r*(1-chi2cdf(cH^2,r));
    elseif(length(varargin) == 2)
        cH = varargin{1};
        bH = varargin{2};
    elseif(length(varargin) >= 3)
        error("Check possible input combinations.")
    end
    
    psi = zeros(length(t),1);
    psi(t <= cH^2) = 1./(2.*bH);
    psi(t > cH^2) = cH.^2./(2.*bH.*t(t > cH^2));
end