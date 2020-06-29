function pdf = mvsnpdf(X, xi, S, lambda)
% Multivariate skew normal (gaussian) pdf
%
% Inputs:
%        X  - (N, r) Values at which to evaluate the pdf
%        xi - (r, 1) "mean" vector
%        S - (r, r) covariance matrix
%        lambda - (r, 1) skewness factor
%
% Outputs: 
%        pdf - (N, r) Multivariate skew normal (gaussian) pdf
%
% created by Christian A. Schroth, 29. June 2020
%
% "Real Elliptically Skewed Distributions and Their Application to Robust Cluster Analysis"
% Christian A. Schroth and Michael Muma, Signal Processing Group, Technische Universität Darmstadt
% submitted to IEEE Transactions on Signal Processing

        N = size(X, 1);
        eta = dot((repmat(lambda.', N, 1) / S).',(X.' - xi), 1).' ./ (1 + (lambda.' / S) * lambda);
        tau = sqrt(1./(1 + (lambda.' / S) * lambda));
        
        Omega = S + lambda * lambda.';
        pdf = 2 *  mvnpdf(X, xi.', Omega) .* normcdf(eta./tau);
        
        % slower alternative
        %t = mahalanobisDistance(X, xi, Omega);
        %pdf = 2 * det(Omega)^(-1/2) .* g_gaus(t, r) .* normcdf(eta./tau);

end