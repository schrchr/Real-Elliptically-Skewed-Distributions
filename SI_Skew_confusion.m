%
% created by Christian A. Schroth, 29. June 2020
%
% "Real Elliptically Skewed Distributions and Their Application to Robust Cluster Analysis"
% Christian A. Schroth and Michael Muma, Signal Processing Group, Technische Universität Darmstadt
% submitted to IEEE Transactions on Signal Processing

close all
clear all

addpath("functions", "result")

%% User Input
% number of threads
%parpool(14); 
% percentage of replacement outliers
epsilon = [0];
% number of data points per cluster
N_k = 200; 
% Monte Carlo iterations
MC = 10;
% Select combinations of EM and BIC to be simulated
% 1: Gaussian, 2: t, 3: Huber
em_bic = [1 1;
          2 2;
          3 3];
           
% design parameter
% t:
nu = 3;
% Huber:
qH = 0.8;


%% data generation     
embic_iter = size(em_bic, 1);
eps_iter = length(epsilon);

for iEpsilon = 1:eps_iter
    for iMC = 1:MC
        [data(:,:,iEpsilon,iMC), labels(:,iEpsilon,iMC), r, N, K_true, mu_true, S_true] = data_31_skew(N_k, epsilon(iEpsilon));
    end
end

L_max = 2*K_true; % search range
 
%% design parameter
cH = sqrt(chi2inv(qH, r));
bH = chi2cdf(cH^2,r+2) + cH^2/r*(1-chi2cdf(cH^2,r));
aH = gamma(r/2)/pi^(r/2) / ( (2*bH)^(r/2)*(gamma(r/2) - igamma(r/2, cH^2/(2*bH))) + (2*bH*cH^r*exp(-cH^2/(2*bH)))/(cH^2-bH*r) );

%% density definitions
g = {@(t)g_gaus(t, r);
     @(t)g_t(t, r, nu);
     @(t)g_huber(t, r, cH, bH, aH)};
 
rho = {@(t)rho_gaus(t, r);
       @(t)rho_t(t, r, nu);
       @(t)rho_huber(t, r, cH, bH, aH)};

psi = {@(t)psi_gaus(t);
       @(t)psi_t(t, r, nu);
       @(t)psi_huber(t, r, cH, bH)};

eta = {@(t)eta_gaus(t);
       @(t)eta_t(t, r, nu); 
       @(t)eta_huber(t, r, cH, bH)};
      
PSI = {@(x)PSI_skew_gaus(x);
       @(x)PSI_skew_t(x, nu + r);
       @(x)PSI_skew_huber(x, cH, bH, aH)}; 
   
cdf = {@(x)normcdf(x);
       @(x)tcdf(x, nu);
       @(x)hubercdf(x, 0, 1, cH, bH, aH)};


   
%% Cluster Enumeration
numMem = 2;
mem_final = zeros(N, numMem, embic_iter, eps_iter, MC);
%mem_final = single(mem_final);
tic
for iEpsilon = 1:eps_iter
    parfor iMC = 1:MC %parfor iMC = 1:MC 
        mem_est = zeros(N, numMem, embic_iter);
        for iEmBic = 1:embic_iter
            for ll = K_true %1:L_max
                %% EM
                [mu_est, S_est, t, R] = EM_RES(data(:,:,iEpsilon,iMC), ll, g{em_bic(iEmBic,1)}, psi{em_bic(iEmBic,1)});
                mem = (R == max(R,[],2));
                
                [xi_est_skew, lambda_est_skew, S_est_skew, t_skew, R_skew] = EM_RES_skew(data(:,:,iEpsilon,iMC), ll, g{em_bic(iEmBic,1)}, psi{em_bic(iEmBic,1)}, eta{em_bic(iEmBic,1)}, PSI{em_bic(iEmBic,1)}, cdf{em_bic(iEmBic,1)});
                mem_skew = (R_skew == max(R_skew,[],2));  
                
                % hard clustering
                if(ll == K_true)
                    mem_est(:,1,iEmBic) = sum(mem.* (1:K_true), 2);
                    mem_est(:,2,iEmBic) = sum(mem_skew.* (1:K_true), 2);
                end
            end
        end
        mem_final(:,:,:,iEpsilon,iMC) = mem_est;
    end
    disp(num2str(epsilon(iEpsilon)))
    toc
end

%% Confusion Matrix
for iEmBic = 1:embic_iter
    for iEpsilon = 1:eps_iter 
        for iMem = 1:numMem
            for iMC = 1:MC 
                % ignore outliers
                labels_true = labels(labels(:,iEpsilon,iMC) ~= K_true+1, iEpsilon,iMC);
                labels_est = mem_final(labels(:,iEpsilon,iMC) ~= K_true+1, iMem, iEmBic, iEpsilon, iMC);

                [C, order] = confusionmat(labels_true, labels_est);
                [I] = allpermsALT(C, "max");
                C_sort = C(:,I);

                for k = 1:K_true
                    C_percent(k,:,iMC) = C_sort(k,:)./sum(labels(:,iEpsilon,iMC) == k);
                end
            end
            C_final(:,:, iMem, iEmBic, iEpsilon) = mean(C_percent, 3).* 100;
            avg_p_det(iMem, iEmBic, iEpsilon) = trace(C_final(:,:, iMem, iEmBic, iEpsilon))/K_true;
        end
    end
end

%(:,:, iMem, iEmBic, iEpsilon)
% iMem: 1 ellip, 2 skew
% iEmBic: 1 Gauß, 2 t, 3, Huber
% iEpsilon: 1 0, 2 0.01, 3 0.02, 4 0.04
save("result/confusion.mat", "C_final")

%% Plot
figure 
plot_scatter([mem_final(:, iMem, iEmBic, iEpsilon, iMC), data(:,:,iEpsilon,iMC)], K_true, r)