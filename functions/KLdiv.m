function [KL] = KLdiv(P, Q)
% calculates Kullback-Leibler divergence between the pdfs P and Q, both
% have to be valid pdfs, thus integrating to 1
%
% Inputs:
%       P - true pdf
%       Q - estimated pdf
%
% Outputs:
%       KL - (1, 1) Kullback-Leibler divergence
%
% created by Christian A. Schroth, 29. June 2020
%
% "Real Elliptically Skewed Distributions and Their Application to Robust Cluster Analysis"
% Christian A. Schroth and Michael Muma, Signal Processing Group, Technische Universität Darmstadt
% submitted to IEEE Transactions on Signal Processing

    KL = sum(P(Q ~= 0 & P ~= 0) .* log(P(Q ~= 0 & P ~= 0)./Q(Q ~= 0 & P ~= 0)),'all');
end