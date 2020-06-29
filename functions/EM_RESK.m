function [xi_hat, lambda_hat, S_hat, t, R] = EM_RESK(data, ll, g, psi, eta, PSI, cdf)
% EM algorithm for mixture of RESK distributions defined by g, psi, eta, PSI, cdf
%
% Inputs:
%        data - (N, r) data matrix
%        ll - number of clusters
%        g - density generator
%        psi - psi of density generator g
%        eta - eta of density generator g
%        PSI - PSI of density generator g
%        cdf - cdf of density generator g
%
% Outputs: 
%        xi_hat - (r, ll) estimate of cluster centroids
%        lambda_hat - (r, ll) estimate of skewnes factors
%        S_hat - (r, r, ll) estimate of cluster scatter matrices
%        t - (N, ll) skewed Mahalanobis distances to all clusters
%        R - (N, ll) final estimate of posterior probabilities of cluster
%
% created by Christian A. Schroth, 29. June 2020
%
% "Real Elliptically Skewed Distributions and Their Application to Robust Cluster Analysis"
% Christian A. Schroth and Michael Muma, Signal Processing Group, Technische Universität Darmstadt
% submitted to IEEE Transactions on Signal Processing

limit = 1e-5; % a value that determines when the EM algorithm should terminate
em_max_iter = 150; % maximum number of iterations of the EM algorithm
reg_value = 1e-6; % regularization value used to regularize the covariance matrix in the EM algorithm

r = size(data, 2);
N = size(data, 1);

%% variable initializations
v = zeros(N,ll);
tau = zeros(1,ll);
S_hat = zeros(r,r,ll);
lambda_hat = ones(r,ll);
t = zeros(N,ll);
log_likelihood = zeros(1, em_max_iter);

%% initialization using K-means++ 
warning('off', "stats:kmeans:FailedToConvergeRep")
%[clu_memb_kmeans, mu_Kmeans] = kmeans(data, ll, 'MaxIter', 10, 'Replicates', 5);
[clu_memb_kmeans, mu_Kmeans] = kmeans(data, ll, 'Distance', 'cityblock', 'MaxIter', 10, 'Replicates', 5);
xi_hat = mu_Kmeans.';
for m = 1:ll
    tau(m) = sum(clu_memb_kmeans == m)/N;
end

for m = 1:ll
    x_hat = data(clu_memb_kmeans == m).' - xi_hat(:,m);
    N_m = sum(clu_memb_kmeans == m);
    S_hat(:,:,m) = (x_hat * x_hat.') ./ N_m;
    
    % Check if the sample covariance matrix is positive definite
    [~, indicator] = chol(S_hat(:,:,m));
    if (indicator ~= 0 || cond(S_hat(:,:,m)) > 30)
        S_hat(:,:,m) = 1/(r*N_m)*sum(diag(x_hat'*x_hat))*eye(r); % diagonal covariance matrix whose diagonal entries are identical
        [~,indicator] = chol(S_hat(:,:,m));
        %disp("Initial Covariance not PSD")
        if indicator ~= 0
            S_hat(:,:,m) = eye(r); % if the estimated covariance matrix is still singular, then set it to identity
        end
    end
    
    Omega_hat(:,:,m) = S_hat(:,:,m) + lambda_hat(:,m)*lambda_hat(:,m).';
    t(:,m) = mahalanobisDistance(data, xi_hat(:,m), Omega_hat(:,:,m));
end

%% EM algorithm
for ii = 1:em_max_iter
    % E-step
    
    for m = 1:ll
        eta_nm(:,m) = dot((repmat(lambda_hat(:,m).', N,1) / S_hat(:,:,m)).',(data.' - xi_hat(:,m)), 1).' ...
                    ./ (1 + (lambda_hat(:,m).' / S_hat(:,:,m)) * lambda_hat(:,m));
        tau_m(:,m) = sqrt(1./(1 + (lambda_hat(:,m).' / S_hat(:,:,m)) * lambda_hat(:,m)));

        kappa_nm(:,m) = eta_nm(:,m)./tau_m(:,m) .* sqrt(2*psi(t(:,m)));
        
        e_0(:,m) = 2 * psi(t(:,m)) + 2 * PSI(kappa_nm(:,m)) .* eta_nm(:,m)./tau_m(:,m) .* eta(t(:,m))./sqrt(2*psi(t(:,m)));

        e_1(:,m) = 2 * psi(t(:,m)) .* eta_nm(:,m) - PSI(kappa_nm(:,m)) .* tau_m(:,m) .* sqrt(2*psi(t(:,m)))...
                    + 2 * PSI(kappa_nm(:,m)) .* eta_nm(:,m).^(2)./tau_m(:,m) .* eta(t(:,m))./sqrt(2*psi(t(:,m)));

        e_2(:,m) = tau_m(:,m).^(2) + 2 * psi(t(:,m)) .* eta_nm(:,m).^2 - PSI(kappa_nm(:,m)) .* tau_m(:,m) .*eta_nm(:,m) .* sqrt(2*psi(t(:,m)))...
                    + 2 * PSI(kappa_nm(:,m)) .* eta_nm(:,m).^(3)./tau_m(:,m) .* eta(t(:,m))./sqrt(2*psi(t(:,m)));
    end
    
    
    v_lower = zeros(N,ll);
    for j = 1:ll
        v_lower(:,j) = tau(j) * 2 * det(Omega_hat(:,:,j))^(-1/2) .* g(t(:,j)) .* cdf(kappa_nm(:,j));
    end

    for m = 1:ll
        v(:,m) = tau(m) .* 2 * det(Omega_hat(:,:,m))^(-1/2) .* g(t(:,m)) .* cdf(kappa_nm(:,m)) ./ sum(v_lower,2);
    end


    % M-step
    for m = 1:ll
        xi_hat(:,m) = sum(v(:,m) .* (e_0(:,m) .* data - e_1(:,m) .* lambda_hat(:,m).')) / sum(v(:,m).* e_0(:,m));    
        x_hat = (data.' - xi_hat(:,m));
        lambda_hat(:,m) = sum(v(:,m) .* e_1(:,m) .* x_hat.') ./ sum(v(:,m) .* e_2(:,m));          
        S_hat(:,:,m) = (( v(:,m).' .* e_0(:,m).' .* x_hat * x_hat.' ...
                         - v(:,m).' .* e_1(:,m).' .* x_hat * repmat(lambda_hat(:,m).', N,1)...
                         - v(:,m).' .* e_1(:,m).' .* repmat(lambda_hat(:,m).', N,1).' * x_hat.' ...
                         + v(:,m).' .* e_2(:,m).' .* repmat(lambda_hat(:,m).', N,1).' * repmat(lambda_hat(:,m).', N,1)...
                          ) ./ sum(v(:,m))) + reg_value*eye(r);
        
        % Fallback to symmetric if negative definit
        if(det(S_hat(:,:,m)) <= 0)      
            S_hat(:,:,m) = (2 .* ( v(:,m).' .* psi(t(:,m)).' .* x_hat * x_hat.' ) ./ sum(v(:,m))) + reg_value*eye(r);
            if(det(S_hat(:,:,m)) <= 0)  
                S_hat(:,:,m) = eye(r);
            end
        end              
                                     
        tau(m) = sum(v(:,m))/N;
        Omega_hat(:,:,m) = S_hat(:,:,m) + lambda_hat(:,m)*lambda_hat(:,m).';
        t(:,m) = mahalanobisDistance(data, xi_hat(:,m), Omega_hat(:,:,m));
    end

    % convergence check
    v_conv = zeros(N, ll);
    for m = 1:ll
        v_conv(:,m) = tau(m)* 2 * det(Omega_hat(:,:,m))^(-1/2) * g(t(:,m)) .* cdf(kappa_nm(:,m));
    end
    log_likelihood(ii) = sum(log(sum(v_conv,2)));

    if(ii > 1)
        if(abs(log_likelihood(ii)-log_likelihood(ii-1)) < limit)
            % break if difference between two consecutive steps is small
            break;
        end   
    end
end
% calculate posterior probabilities
R = v_conv./sum(v_conv, 2);

%% diagonal loading
% If the estimated "Matrix is close to singular or badly scaled" it cannot be inverted. 
% https://math.stackexchange.com/questions/261295/to-invert-a-matrix-condition-number-should-be-less-than-what
% If S_hat has a large condition number a small number in comparison to the
% matrix entries is added. This step should be subject for further tweaking.
for m = 1:ll
    cond_S = cond(S_hat(:,:,m));
    if(cond_S > 30)
        %warning("S with large condition number")
        S_hat(:,:,m) = S_hat(:,:,m) + 0.01 * 10^floor(log10(trace(S_hat(:,:,m)))) * log10(cond_S) * eye(r);
    end
end

        
% figure
% plot(log_likelihood(log_likelihood ~= 0))
% grid on
% xlabel("number of iterations")
% ylabel("log-likelihood")