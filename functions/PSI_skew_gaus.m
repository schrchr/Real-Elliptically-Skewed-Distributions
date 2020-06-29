function PSI = PSI_skew_gaus(x)
% Computes PSI(x) of the Gaussian distribution, with approximation from:
%I. Vrbik, “Non-Elliptical and Fractionally-Supervised Classification,” Ph.D. dissertation,
%The University of Guelph, Guelph, Ontario, Canada, 2014. [Online]. Available: 
%https ://atrium.lib.uoguelph.ca/xmlui/bitstream/handle/10214/8096/Vrbik Irene 201405 Phd.pdf?sequence=1&isAllowed=y.
%
% Inputs:
%       x  - (N, 1) Values at which to evaluate PSI
%
% Outputs:
%       PSI - (N, 1) PSI(x) of Gaussian distribution
%
% created by Christian A. Schroth, 29. June 2020
%
% "Real Elliptically Skewed Distributions and Their Application to Robust Cluster Analysis"
% Christian A. Schroth and Michael Muma, Signal Processing Group, Technische Universität Darmstadt
% submitted to IEEE Transactions on Signal Processing


    PSI = zeros(length(x),1);
    PSI(x >= -37) = - normpdf(x(x >= -37))./normcdf(x(x >= -37));
    PSI(x < -37) = - 1./(1./(-x(x < -37)) - 1./(-x(x < -37)).^3 + 3./(-x(x < -37)).^5 - 15./(-x(x < -37)).^7 + 105./(-x(x < -37)).^9);

end