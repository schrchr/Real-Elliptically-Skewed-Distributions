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
epsilon = 0:0.02:0.08;
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


%% data generation     
embic_iter = size(em_bic, 1);
eps_iter = length(epsilon);

for iEpsilon = 1:eps_iter
    for iMC = 1:MC
        [data(:,:,iEpsilon,iMC), labels(:,iEpsilon,iMC), r, N, K_true, mu_true, S_true, lambda_true] = data_31_skew(N_k, epsilon(iEpsilon));
    end
end
for ll = 1:K_true
    S_kew = S_true(:,:,ll);
    S_true_vech(:,ll) = S_kew(triu(true(size(S_kew))));
    theta_true(:,ll) = [mu_true(:,ll); S_true_vech(:,ll)];
    theta_skew_true(:,ll) = [mu_true(:,ll); lambda_true(:,ll); S_true_vech(:,ll)];
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
            pdf_est = pdf_est./(sum(pdf_est, 'all').*step_pdf.^r);
            
            pdf_est_skew = zeros(size(X));
            for ll = 1:K_est_skew
                pdf_est_skew = pdf_est_skew + reshape(mvsnpdf([X(:) Y(:)], xi_est_skew{K_est_skew}(:,ll), S_est_skew{K_est_skew}(:,:,ll), lambda_est_skew{K_est_skew}(:,ll)),size(X));
            end
            pdf_est_skew = pdf_est_skew./(sum(pdf_est_skew, 'all').*step_pdf.^r);
            
            pdf_est_K_true = zeros(size(X));
            pdf_est_skew_K_true = zeros(size(X));
            for ll = 1:K_true
                pdf_est_K_true = pdf_est_K_true + reshape(mvnpdf([X(:) Y(:)], mu_est{K_true}(:,ll).', S_est{K_true}(:,:,ll)),size(X));
                pdf_est_skew_K_true = pdf_est_skew_K_true + reshape(mvsnpdf([X(:) Y(:)], xi_est_skew{K_true}(:,ll), S_est_skew{K_true}(:,:,ll), lambda_est_skew{K_true}(:,ll)),size(X));
            end
            pdf_est_K_true = pdf_est_K_true./(sum(pdf_est_K_true, 'all').*step_pdf.^r);
            pdf_est_skew_K_true = pdf_est_skew_K_true./(sum(pdf_est_skew_K_true, 'all').*step_pdf.^r);
            
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
    disp(num2str(epsilon(iEpsilon)))
    toc
end


%% Evaluation
KL_plot = mean(permute(KL_final, [2 3 4 1]), 4);

%% Plot & Save

marker = {'o','s','d','*','x','^','v','>','<','p','h', '+','o'};
g_names = ["Gaus", "t", "Huber", "Tukey"];
pen_names = ["BIC-Schwarz", "BIC-Skew-Schwarz", "True-K", "True-K-skew"];


for iEmBic = 1:embic_iter
    fig = figure;
    h = plot(epsilon, KL_plot(:,:,iEmBic).', 'LineWidth', 1.5);
    hold on
    grid on
    set(h,{'Marker'}, {marker{1:size(KL_final, 3)}}.')
    xlabel("% of outliers")
    ylabel("KL divergence")
    ylim([0 100])
    legend(pen_names, 'Location', 'northeast')
    title("Nk-" + num2str(N_k) + ", EM-" + g_names(em_bic(iEmBic,1)))

    % save to .csv
     T = array2table([epsilon.', KL_plot(:,:,iEmBic)]);
     T.Properties.VariableNames = ["x", pen_names];
     writetable(T,"result/KL_" + g_names(em_bic(iEmBic,1))+ "_MC_" + num2str(MC) + "_Nk_" + num2str(N_k) + ".csv", 'Delimiter','tab')
end

for iEmBic = 1:embic_iter
    names_3(iEmBic,:) = ["EM: " + g_names(em_bic(iEmBic,1))];
end

for ii_bic = 1:size(KL_plot, 2)
    fig = figure;
    h = plot(epsilon, permute(KL_plot(:,ii_bic,:), [1 3 2]), 'LineWidth', 1.5);
    hold on
    set(h,{'Marker'}, {marker{1:embic_iter}}.')
    grid on
    xlabel("% of outliers")
    ylabel("KL divergence")
    ylim([0 100])
    legend(names_3, 'Location', 'northwest')
    title("Nk-" + num2str(N_k) + ", " + pen_names(ii_bic))

    % save to .csv
    T = array2table([epsilon.', permute(KL_plot(:,ii_bic,:), [1 3 2])]);
    T.Properties.VariableNames = ["x", names_3.'];
    writetable(T,"result/KL_" + pen_names(ii_bic) + "_MC_" + num2str(MC) + "_Nk_" + num2str(N_k) + ".csv", 'Delimiter','tab')
end



% figure
% plot_scatter(data, K_true, r)
% 
% T = array2table([data(:,2:3,3,10), data(:,1,3,10)]); 
% T.Properties.VariableNames = ["x", "y", "label"];
% writetable(T,"result/data_31_skew_" + num2str(N_k) + "_eps_" + replace(num2str(0.02),".", "") + ".csv", 'Delimiter','tab')
