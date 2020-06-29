function [bic, like, pen] = BIC_skew_S(data, S_hat, xi_hat, lambda_hat, mem, t, rho, cdf)
% computes the BIC of a Skew-RES distribution with Schwarz Penalty Term
%
% Inputs:
%        data - (N, r+1) (:,1) labels, (:,2:end) data
%        xi_hat - (r, ll) estimated "mean" values
%        S_hat - (r, r, ll) estimated Scatter matrix of cluster m
%        lambda_hat - (r, ll) estimated skewness factors
%        mem - (N, ll) cluster memberships
%        t - (N, 1) squared Mahalanobis distances 
%        rho - rho of density generator g
%        cdf - cdf of density generator g
%
% Outputs: 
%        bic - (1,1)
%        like - (1,1)
%        pen - (1,1)
%
% created by Christian A. Schroth, 29. June 2020
%
% "Real Elliptically Skewed Distributions and Their Application to Robust Cluster Analysis"
% Christian A. Schroth and Michael Muma, Signal Processing Group, Technische Universität Darmstadt
% submitted to IEEE Transactions on Signal Processing


    N_m = sum(mem);
    r = size(S_hat,1);
    ll = size(S_hat,3);
    q = 1/2*r*(r+5);
    N = length(t);
    
    if (N == 0)
       N = 1; 
    end
    
    temp_rho = zeros(1,ll);
    temp_F = zeros(1,ll);
    logdetOmega = zeros(1,ll);

    for m = 1:ll
        x_hat_m = data(mem(:,m), :).' - xi_hat(:,m);
        eta_nm = dot((repmat(lambda_hat(:,m).', N_m(m),1) / S_hat(:,:,m)).',x_hat_m, 1).' ./ (1 + (lambda_hat(:,m).' / S_hat(:,:,m)) * lambda_hat(:,m));     
        tau_m = sqrt(1./(1 + (lambda_hat(:,m).' / S_hat(:,:,m)) * lambda_hat(:,m)));
        kappa_nm = eta_nm./tau_m;
        
        temp_rho(m) = sum(rho(t(mem(:,m), m)));
        temp_F(m) = sum(log(cdf(kappa_nm)));
        logdetOmega(m) = log(det(S_hat(:,:,m) + lambda_hat(:,m) * lambda_hat(:,m).'));
    end

    like = - sum(temp_rho) + sum(N_m(N_m > 0) .* log(N_m(N_m > 0))) - sum(N_m(N_m > 0) .* logdetOmega(N_m > 0))/2 + sum(temp_F);
    pen =  - q*ll/2*log(N);

    bic = like + pen;   
end