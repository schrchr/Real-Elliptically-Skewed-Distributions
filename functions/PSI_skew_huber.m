function PSI = PSI_skew_huber(x, varargin)
% Computes PSI(x) of the Huber distribution
%
% Possible Input Combinations:
%       x
%       x, qH
%       x, cH, bH, aH - this option is provided, to improve performance, because it allows to avoid the calculation of
%                          the constants cH, bH and aH in every loop iteration
%
% Inputs:
%       x  - (N, 1) Values at which to evaluate PSI
%       qH - (1, 1) tuning parameter, standard value 0.8, choose qH > 0.701
%       cH - (1, 1) tuning parameter
%       bH - (1, 1) constant
%       aH - (1, 1) constant
%
% Outputs:
%       PSI - (N, 1) PSI(x) of Huber distribution
%
% created by Christian A. Schroth, 29. June 2020
%
% "Real Elliptically Skewed Distributions and Their Application to Robust Cluster Analysis"
% Christian A. Schroth and Michael Muma, Signal Processing Group, Technische Universität Darmstadt
% submitted to IEEE Transactions on Signal Processing

    PSI = - huberpdf(x, 0, 1, varargin{:}) ./ hubercdf(x, 0, 1, varargin{:});
end