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
%parpool(20); 
% number of data points per cluster
N_k = 50; 
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

% range of outliers
pdf_range = [-15 45; -20 30]; %range of outliers
% steps between outliers
step_pdf = 0.1;  

% range of outliers
out_range = [-15 45; -20 30]; %range of outliers
% steps between outliers
step_eps = 5;  

%% data generation     
% grid for single outlier
x = out_range(1,1):step_eps:out_range(1,2); 
y = out_range(2,1):step_eps:out_range(2,2);
[X_eps, Y_eps] = meshgrid(x,y);

embic_iter = length(em_bic);
eps_iter = numel(X_eps);

for iEpsilon = 1:eps_iter
    for iMC = 1:MC
        [data(:,:,iEpsilon,iMC), labels(:,iEpsilon,iMC), r, N, K_true, mu_true, S_true, lambda_true] = data_31_skew(N_k, 0);
        
        % replacement outlier
        N_repl = 1;
        index_repl = randperm(N,N_repl).';
        data(index_repl,:,iEpsilon,iMC) = [X_eps(iEpsilon) Y_eps(iEpsilon)];
        labels(N_repl,iEpsilon,iMC) = K_true+1;
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

   
iKL = 4;
KL_final = zeros(MC, eps_iter, iKL, embic_iter);


x = pdf_range(1,1):step_pdf:pdf_range(1,2); 
y = pdf_range(2,1):step_pdf:pdf_range(2,2);
[X, Y] = meshgrid(x,y);

pdf_true = zeros(size(X));
for ll = 1:K_true
    pdf_true = pdf_true + reshape(mvsnpdf([X(:) Y(:)], mu_true(:,ll), S_true(:,:,ll), lambda_true(:,ll)),size(X));
end
pdf_true = pdf_true./(sum(pdf_true, 'all').*step_pdf.^2);

% figure
% contour(X,Y,pdf_true)
% 
% figure
% surf(X,Y,pdf_true, 'EdgeColor', 'none')

%% Cluster Enumeration
tic
for iEpsilon = 1:eps_iter
    parfor iMC = 1:MC %parfor iMC = 1:MC 
        mu_est = cell(1,L_max);
        S_est = cell(1,L_max);
        
        xi_est_skew = cell(1,L_max);
        lambda_est_skew = cell(1,L_max);
        S_est_skew = cell(1,L_max);
        
        KL = zeros(iKL, embic_iter);
       
        for iEmBic = 1:embic_iter
            bic = zeros(1,L_max);
            bic_skew = zeros(1,L_max);
        	for ll = 1:L_max
                %% EM
                [mu_est{ll}, S_est{ll}, t, R] = EM_RES(data(:,:,iEpsilon,iMC), ll, g{em_bic(iEmBic,1)}, psi{em_bic(iEmBic,1)});
                mem = (R == max(R,[],2));

                [xi_est_skew{ll}, lambda_est_skew{ll}, S_est_skew{ll}, t_skew, R_skew] = EM_RESK(data(:,:,iEpsilon,iMC), ll, g{em_bic(iEmBic,1)}, psi{em_bic(iEmBic,1)}, eta{em_bic(iEmBic,1)}, PSI{em_bic(iEmBic,1)}, cdf{em_bic(iEmBic,1)});
                mem_skew = (R_skew == max(R_skew,[],2));  
                %% BIC
                [bic(ll), ~, ~] = BIC_S(S_est{ll}, t, mem, rho{em_bic(iEmBic,2)});
                [bic_skew(ll), ~, ~] = BIC_skew_S(data(:,:,iEpsilon,iMC), S_est_skew{ll}, xi_est_skew{ll}, lambda_est_skew{ll}, mem_skew, t_skew, rho{em_bic(iEmBic,2)}, cdf{em_bic(iEmBic,2)});                                                                    
            end
            
            %% #Clusters
            K_est = sum((bic == max(bic,[],2)).*[1:L_max]);
            K_est_skew = sum((bic_skew == max(bic_skew,[],2)).*[1:L_max]);
            
            %% KL divergence  
            pdf_est = zeros(size(X));
            for ll = 1:K_est
                pdf_est = pdf_est + reshape(mvnpdf([X(:) Y(:)], mu_est{K_est}(:,ll).', S_est{K_est}(:,:,ll)),size(X));
            end
            pdf_est = pdf_est./(sum(pdf_est, 'all').*step_pdf.^2);
            
            pdf_est_skew = zeros(size(X));
            for ll = 1:K_est_skew
                pdf_est_skew = pdf_est_skew + reshape(mvsnpdf([X(:) Y(:)], xi_est_skew{K_est_skew}(:,ll), S_est_skew{K_est_skew}(:,:,ll), lambda_est_skew{K_est_skew}(:,ll)),size(X));
            end
            pdf_est_skew = pdf_est_skew./(sum(pdf_est_skew, 'all').*step_pdf.^2);
            
            pdf_est_K_true = zeros(size(X));
            pdf_est_skew_K_true = zeros(size(X));
            for ll = 1:K_true
                pdf_est_K_true = pdf_est_K_true + reshape(mvnpdf([X(:) Y(:)], mu_est{K_true}(:,ll).', S_est{K_true}(:,:,ll)),size(X));
                pdf_est_skew_K_true = pdf_est_skew_K_true + reshape(mvsnpdf([X(:) Y(:)], xi_est_skew{K_true}(:,ll), S_est_skew{K_true}(:,:,ll), lambda_est_skew{K_true}(:,ll)),size(X));
            end
            pdf_est_K_true = pdf_est_K_true./(sum(pdf_est_K_true, 'all').*step_pdf.^2);
            pdf_est_skew_K_true = pdf_est_skew_K_true./(sum(pdf_est_skew_K_true, 'all').*step_pdf.^2);
            
%             figure
%             contour(X,Y,pdf_est_skew)
%             figure
%             contour(X,Y,pdf_est)
%             
%             figure
%             surf(X,Y,pdf_est_skew, 'EdgeColor', 'none')
            
            KL(1,iEmBic) = KLdiv(pdf_true, pdf_est);
            KL(2,iEmBic) = KLdiv(pdf_true, pdf_est_skew);
            KL(3,iEmBic) = KLdiv(pdf_true, pdf_est_K_true);
            KL(4,iEmBic) = KLdiv(pdf_true, pdf_est_skew_K_true);
        	
        end
        KL_final(iMC, iEpsilon, :, :) = KL;
    end
    disp(num2str(iEpsilon) + "/" + num2str(eps_iter))
    toc
end


%% Evaluation
KL_plot = mean(permute(KL_final, [2 3 4 1]), 4);

%% Plot & Save
g_names = ["Gaus", "t", "Huber"];
%names = ["Finite", "Asymptotic", "Schwarz", "Skew-Finite", "Skew-Asymptotic", "Skew-Schwarz"];
names = ["ellip", "skew", "ellip-k-known", "skew-k-known"];
[data, labels, r, N, ~, ~, ~, ~] = data_31_skew(N_k, 0);

for ii_embic = 1:embic_iter
    for k_bic = 1:size(KL_final, 3)
        Z = reshape(KL_plot(:,k_bic,ii_embic),size(X_eps));
        fig = figure;
        [M,c] = contour(X_eps,Y_eps,Z);
        c.LineWidth = 1.5;
        hold on
        plot_scatter([labels data], K_true, r)
        title("EM-" + g_names(em_bic(ii_embic,1)) + "-" + names(k_bic))
        colorbar;
        oldcmap = colormap;
        colormap( flipud(oldcmap) );
        %caxis([0 100])
        
        % save to .csv
        T = array2table([contour(X_eps,Y_eps,Z).']); 
        writetable(T,"result/sens_KL_EM_" + g_names(em_bic(ii_embic,1)) + "_" + names(k_bic) + "_Nk_" + num2str(N_k) + "_step_" + num2str(step_eps) + "_MC_" + num2str(MC) + "_1.csv", 'Delimiter','tab')
        T = array2table([data, labels]); 
        T.Properties.VariableNames = ["x", "y", "label"];
        writetable(T,"result/sens_KL_EM_" + g_names(em_bic(ii_embic,1)) + "_" + names(k_bic) + "_Nk_" + num2str(N_k) + "_step_" + num2str(step_eps) + "_MC_" + num2str(MC) + "_2.csv", 'Delimiter','tab')
    end
end