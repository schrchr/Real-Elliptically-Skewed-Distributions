function PSI = PSI_skew_t(x, nu)
% Computes PSI(x) of the t distribution
%
% Inputs:
%       x  - (N, 1) Values at which to evaluate PSI
%       nu - (1, 1) degree of freedom
%
% Outputs:
%       PSI - (N, 1) PSI(x) of t distribution
%
% created by Christian A. Schroth, 29. June 2020
%
% "Real Elliptically Skewed Distributions and Their Application to Robust Cluster Analysis"
% Christian A. Schroth and Michael Muma, Signal Processing Group, Technische Universität Darmstadt
% submitted to IEEE Transactions on Signal Processing

PSI = - tpdf(x, nu) ./ tcdf(x, nu);

end