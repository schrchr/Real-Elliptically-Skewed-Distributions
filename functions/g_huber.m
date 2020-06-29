function g = g_huber(t, r, varargin)
% Computes g(t) of the Huber distribution
%
% Possible Input Combinations:
%       t, r
%       t, r, qH
%       t, r, cH, bH, aH - this option is provided, to improve performance, because it allows to avoid the calculation of
%                          the constants cH, bH and aH in every loop iteration
%
% Inputs:
%       t  - (N, 1) squared Mahalanobis distance
%       r  - (1, 1) dimension
%       qH - (1, 1) tuning parameter, standard value 0.8, choose qH > 0.701
%       cH - (1, 1) tuning parameter
%       bH - (1, 1) constant
%       aH - (1, 1) constant
%
% Outputs:
%       g - (N, 1) g(t) of Huber distributions
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
        aH = gamma(r/2)/pi^(r/2) / ( (2*bH)^(r/2)*(gamma(r/2) - igamma(r/2, cH^2/(2*bH))) + (2*bH*cH^r*exp(-cH^2/(2*bH)))/(cH^2-bH*r) );
    elseif(length(varargin) == 1)
        qH = varargin{1};
        cH = sqrt(chi2inv(qH, r));
        bH = chi2cdf(cH^2,r+2) + cH^2/r*(1-chi2cdf(cH^2,r));
        aH = gamma(r/2)/pi^(r/2) / ( (2*bH)^(r/2)*(gamma(r/2) - igamma(r/2, cH^2/(2*bH))) + (2*bH*cH^r*exp(-cH^2/(2*bH)))/(cH^2-bH*r) );
    elseif(length(varargin) == 3)
        cH = varargin{1};
        bH = varargin{2};
        aH = varargin{3};
    elseif(length(varargin) == 2 || length(varargin) >= 4)
        error("Check possible input combinations.")
    end

    g = zeros(length(t),1);
    g(t <= cH^2) = aH * exp(-t(t <= cH^2)/(2*bH));
    g(t > cH^2) = aH * (exp(1) * t(t > cH^2)./cH^2).^(-cH^2/(2*bH));
end
