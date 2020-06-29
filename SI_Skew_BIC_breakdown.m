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
N_k = 150; 
% Monte Carlo iterations
MC = 20;
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
        %[data(:,:,iEpsilon,iMC), r, N, K_true, mu_true, S_true] = data_31_skew(N_k, epsilon(iEpsilon));
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

   
iBic = 2;
bic_final = zeros(MC, eps_iter, L_max, iBic, embic_iter);
like_final = zeros(MC, eps_iter, L_max, iBic, embic_iter);
pen_final = zeros(MC, eps_iter, L_max, iBic, embic_iter);
   
%% Cluster Enumeration
tic
for iEpsilon = 1:eps_iter
    parfor iMC = 1:MC %parfor iMC = 1:MC 
        bic = zeros(L_max, iBic, embic_iter);   
        like = zeros(L_max, iBic, embic_iter);   
        pen = zeros(L_max, iBic, embic_iter); 
        for iEmBic = 1:embic_iter
            for ll = 1:L_max
                %% EM
                [mu_est, S_est, t, R] = EM_RES(data(:,:,iEpsilon,iMC), ll, g{em_bic(iEmBic,1)}, psi{em_bic(iEmBic,1)});
                mem = (R == max(R,[],2));
                
                [xi_est_skew, lambda_est_skew, S_est_skew, t_skew, R_skew] = EM_RESK(data(:,:,iEpsilon,iMC), ll, g{em_bic(iEmBic,1)}, psi{em_bic(iEmBic,1)}, eta{em_bic(iEmBic,1)}, PSI{em_bic(iEmBic,1)}, cdf{em_bic(iEmBic,1)});
                mem_skew = (R_skew == max(R_skew,[],2));                                   
                
                %% BIC
                [bic(ll, 1, iEmBic), like(ll, 1, iEmBic), pen(ll, 1, iEmBic)] = BIC_S(S_est, t, mem, rho{em_bic(iEmBic,2)});
                [bic(ll, 2, iEmBic), like(ll, 2, iEmBic), pen(ll, 2, iEmBic)] = BIC_skew_S(data(:,:,iEpsilon,iMC), S_est_skew, xi_est_skew, lambda_est_skew, mem_skew, t_skew, rho{em_bic(iEmBic,2)}, cdf{em_bic(iEmBic,2)});                                                                    
            end
        end
        
        bic_final(iMC, iEpsilon, :, :, :) = bic;
        like_final(iMC, iEpsilon, :, :, :) = like;
        pen_final(iMC, iEpsilon, :, :, :) = pen;
    end
    disp(num2str(epsilon(iEpsilon)))
    toc
end


%% Evaluation
for iEmBic = 1:embic_iter
    for iEpsilon = 1:eps_iter
        for k = 1:size(bic_final, 4)
            BICmax = (permute(bic_final(:,iEpsilon,:,k,iEmBic), [1 3 4 2 5]) == max(permute(bic_final(:,iEpsilon,:,k,iEmBic), [1 3 4 2 5]),[],2));

            K_true_det = repmat([(K_true == 1:max(K_true)) zeros(1, L_max-max(K_true))],MC,1) == 1;
            K_true_under = repmat([~(K_true == 1:max(K_true-1)) zeros(1, L_max-max(K_true-1))],MC,1) == 1;

            p_under(k,iEpsilon,iEmBic) = sum(BICmax(K_true_under), 'all')/MC;
            p_det(k,iEpsilon,iEmBic) = sum(BICmax(K_true_det))/MC;
            p_over(k,iEpsilon,iEmBic) = 1 - p_det(k,iEpsilon,iEmBic) - p_under(k,iEpsilon,iEmBic);
        end
    end
end

%% Plot & Save

marker = {'o','s','d','*','x','^','v','>','<','p','h', '+','o'};
g_names = ["Gaus", "t", "Huber", "Tukey"];
%pen_names = ["Finite", "Asymptotic", "Schwarz", "Skew-Finite", "Skew-Asymptotic", "Skew-Schwarz"];
pen_names = ["Schwarz", "Skew-Schwarz"];

for iEmBic = 1:embic_iter
    fig = figure;
    h = plot(epsilon, p_det(:,:,iEmBic).', 'LineWidth', 1.5);
    hold on
    grid on
    set(h,{'Marker'}, {marker{1:size(bic_final, 4)}}.')
    xlabel("% of outliers")
    ylabel("probability of detection")
    ylim([0 1])
    legend(pen_names, 'Location', 'northeast')
    title("Nk-" + num2str(N_k) + ", EM-" + g_names(em_bic(iEmBic,1)) + ", BIC-" + g_names(em_bic(iEmBic,2)))

    % save to .csv
%      T = array2table([epsilon.', p_det(:,:,iEmBic).']);
%      T.Properties.VariableNames = ["x", pen_names];
%      writetable(T,"result/BIC_breakdown_EM_" + g_names(em_bic(iEmBic,1)) + "_BIC_" + g_names(em_bic(iEmBic,2)) + "_MC_" + num2str(MC) + "_Nk_" + num2str(N_k) + ".csv", 'Delimiter','tab')
end


% for iEmBic = 1:embic_iter
%     names_all(iEmBic,:) = ["EM: " + g_names(em_bic(iEmBic,1)) + ", BIC: " + g_names(em_bic(iEmBic,2)) + "-" + pen_names];
% end
% 
% fig = figure;
% for iEmBic = 1:embic_iter
%     h = plot(epsilon, p_det(:,:,iEmBic).', 'LineWidth', 1.5);
%     hold on
%     set(h,{'Marker'}, {marker{1:size(bic_final, 4)}}.')
% end
% grid on
% xlabel("% of outliers")
% ylabel("probability of detection")
% ylim([0 1])
% legend(names_all.', 'Location', 'southeast')
% title("Nk-" + num2str(N_k))

% save to .csv
% axis = get(gca,'Children');  
% fig_x = axis.XData;
% fig_x = fig_x.';
% fig_y = flip(cell2mat({axis.YData}.').', 2);
% fig_leg = flip(string({axis.DisplayName}));
% T = array2table([fig_x, fig_y]);
% T.Properties.VariableNames = ["x", fig_leg];
% writetable(T,"figures/outliers_all_" + "_MC_" + num2str(mc_iter) + "Nk_" + num2str(N_k) + ".csv", 'Delimiter','tab')


for iEmBic = 1:embic_iter
    names_3(iEmBic,:) = ["EM: " + g_names(em_bic(iEmBic,1)) + ", BIC: " + g_names(em_bic(iEmBic,2))];
end

for ii_bic = 1:size(bic_final, 4)% ii_embic = 1:embic_iter
    fig = figure;
    h = plot(epsilon, permute(p_det(ii_bic,:,:),[2 3 1]), 'LineWidth', 1.5);
    hold on
    set(h,{'Marker'}, {marker{1:embic_iter}}.')
    grid on
    xlabel("% of outliers")
    ylabel("probability of detection")
    ylim([0 1])
    legend(names_3, 'Location', 'southwest')
    title("Nk-" + num2str(N_k) + ", BIC-" + pen_names(ii_bic))

    % save to .csv
%     T = array2table([epsilon.', permute(p_det(ii_bic,:,:), [2 3 1])]);
%     T.Properties.VariableNames = ["x", names_3.'];
%     writetable(T,"result/BIC_breakdown_BIC-" + pen_names(ii_bic) + "_MC_" + num2str(MC) + "_Nk_" + num2str(N_k) + ".csv", 'Delimiter','tab')
end

% figure
% plot_scatter(data, K_true, r)
% 
% T = array2table([data(:,2:3,3,10), data(:,1,3,10)]); 
% T.Properties.VariableNames = ["x", "y", "label"];
% writetable(T,"result/data_31_skew_" + num2str(N_k) + "_eps_" + replace(num2str(0.02),".", "") + ".csv", 'Delimiter','tab')
