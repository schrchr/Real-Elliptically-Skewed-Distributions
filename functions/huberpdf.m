function res = huberpdf(x, varargin)
% pdf of a univariate Huber distribution
%
% Possible Input Combinations:
%       x
%       x, qH -> mu = 0, sigma = 1
%       x, mu, sigma -> qH = 0.8
%       x, mu, sigma, qH
%       x, mu, sigma, cH, bH, aH - this option is provided, to improve performance, because it allows to avoid the calculation of
%                                  the constants cH, bH and aH in every loop iteration
%
% Inputs:
%       x  - (N, 1) Values at which to evaluate the pdf
%       mu  - (1, 1) mean value
%       sigma  - (1, 1) variance
%       qH - (1, 1) tuning parameter, standard value 0.8, choose qH > 0.703
%       cH - (1, 1) tuning parameter
%       bH - (1, 1) constant
%       aH - (1, 1) constant
%
% Outputs:
%       pdf - (N, 1) pdf(x) of Huber distribution
%
% created by Christian A. Schroth, 29. June 2020
%
% "Real Elliptically Skewed Distributions and Their Application to Robust Cluster Analysis"
% Christian A. Schroth and Michael Muma, Signal Processing Group, Technische Universität Darmstadt
% submitted to IEEE Transactions on Signal Processing

    r = 1;
    if(isempty(varargin))
        mu = 0;
        sigma = 1;
        qH = 0.8;
        cH = sqrt(chi2inv(qH, r));
        bH = chi2cdf(cH^2,r+2) + cH^2/r*(1-chi2cdf(cH^2,r));
        aH = gamma(r/2)/pi^(r/2) / ( (2*bH)^(r/2)*(gamma(r/2) - igamma(r/2, cH^2/(2*bH))) + (2*bH*cH^r*exp(-cH^2/(2*bH)))/(cH^2-bH*r) );
    elseif(length(varargin) == 1)
        mu = 0;
        sigma = 1;
        qH = varargin{1};
        cH = sqrt(chi2inv(qH, r));
        bH = chi2cdf(cH^2,r+2) + cH^2/r*(1-chi2cdf(cH^2,r));
        aH = gamma(r/2)/pi^(r/2) / ( (2*bH)^(r/2)*(gamma(r/2) - igamma(r/2, cH^2/(2*bH))) + (2*bH*cH^r*exp(-cH^2/(2*bH)))/(cH^2-bH*r) );
    elseif(length(varargin) == 2)
        mu = varargin{1};
        sigma = varargin{2};
        qH = 0.8;
        cH = sqrt(chi2inv(qH, r));
        bH = chi2cdf(cH^2,r+2) + cH^2/r*(1-chi2cdf(cH^2,r));
        aH = gamma(r/2)/pi^(r/2) / ( (2*bH)^(r/2)*(gamma(r/2) - igamma(r/2, cH^2/(2*bH))) + (2*bH*cH^r*exp(-cH^2/(2*bH)))/(cH^2-bH*r) );
    elseif(length(varargin) == 3)
        mu = varargin{1};
        sigma = varargin{2};
        qH = varargin{3};
        cH = sqrt(chi2inv(qH, r));
        bH = chi2cdf(cH^2,r+2) + cH^2/r*(1-chi2cdf(cH^2,r));
        aH = gamma(r/2)/pi^(r/2) / ( (2*bH)^(r/2)*(gamma(r/2) - igamma(r/2, cH^2/(2*bH))) + (2*bH*cH^r*exp(-cH^2/(2*bH)))/(cH^2-bH*r) );
    elseif(length(varargin) == 5)
        mu = varargin{1};
        sigma = varargin{2};
        cH = varargin{3};
        bH = varargin{4};
        aH = varargin{5};
    elseif(length(varargin) == 4 || length(varargin) > 5)
        error("Check possible input combinations.")
    end
    
    x = (x - mu)/sigma;
    
    res = zeros(length(x),1);
    res(abs(x) <= cH) = aH/sigma * exp(-x(abs(x) <= cH).^2/(2*bH));
    res(abs(x) > cH) = aH/sigma * (exp(1) * x(abs(x) > cH).^2./cH^2).^(-cH^2/(2*bH));
end