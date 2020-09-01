% =========================================================================
% Part of the Matlab code to accompany the paper
% 'Solving heterogeneous agent models in discrete time with many idiosyncratic states by perturbation methods'
% by Christian Bayer and Ralph Luetticke

% This code solves the Krusell&Smith model by first order perturbation (w/o dimensionality reduction) and MIT shocks.

% Two functional equations describe the incomplete markets model: 1) the Bellman eq and 2) the Law of motion of the distribution
% Eq. 1): EGM_policyupdate.m
% Eq. 2): Gen_BigTransH.m
% These two functions are used for solving i) the steady state, ii) perturbation, and iii) the MIT shock.

%% Initialize workspace and load directories
clear
clc
close all
Computername='HANC' %Het Agent Neo Classical
starttime=clock;

addpath(genpath('functions'))
addpath(genpath('latex'))

%% Switch options
casename='SS_BASELINE_HANC';

%% Solve for Steady state
disp('Solve Steady State by EGM')
tic
% Set parameters
defineSS_pars

mainskript_steadystate
toc

%% Select aggregate shock
aggrshock           = 'TFP';
par.rhoS            = 0.5;     % Persistence
par.sigmaS          = 0.01;    % STD

%% Produce matrices to reduce state-space
disp('Solve TFP shock. First by Perturbation')
disp('Prepare State Space and Compute System for Steady State')
tic

% state vector (no dimensionality reduction)
Xss=[joint_distr(:); 0];
% control vector (no dimensionality reduction)
Yss=[(c_guess(:)); log(par.Q); log(targets.Y);...
    log(par.W); log(par.N); log(par.R); log(grid.K)];

% Create indices
mpar.numstates   = length(Xss) ;
mpar.numcontrols = length(Yss);

mpar.os = length(Xss) - (mpar.nm*mpar.nh); %aggr states
mpar.oc = length(Yss) - (mpar.nm*mpar.nh); %aggr controls

indexes.c = 1:(mpar.nm*mpar.nh);
indexes.Q  = (mpar.nm*mpar.nh)+1;
indexes.Y  = (mpar.nm*mpar.nh)+2;
indexes.W  = (mpar.nm*mpar.nh)+3;
indexes.N  = (mpar.nm*mpar.nh)+4;
indexes.R  = (mpar.nm*mpar.nh)+5;
indexes.K  = (mpar.nm*mpar.nh)+6;

indexes.distr = 1:mpar.nm*mpar.nh;
indexes.S  = (mpar.nm*mpar.nh)+1;


State       = zeros(mpar.numstates,1);
State_m     = State;
Contr       = zeros(mpar.numcontrols,1);
Contr_m     = Contr;

% Init difference equation
F = @(a,b,c,d)Fsys(a,b,c,d,Xss,Yss,par,mpar,grid,meshes,indexes,P_H,aggrshock);

[Fss,LHS,RHS,Distr] = F(State,State_m,Contr,Contr_m);
toc

%% Solve RE via Schmitt-Grohe-Uribe Form
disp('Take Numerical Derivatives and Solve RE via Schmitt-Grohe-Uribe Form')
tic
[hx,gx,F1,F2,F3,F4,par] = SGU_solver(F,mpar,par);
toc

%% Produce IRFs
x0=zeros(mpar.numstates,1);
x0(end)=par.sigmaS;

MX=[eye(length(x0));gx];
IRF_state_sparse=[];
x=x0;
mpar.maxlag=600;

for t=1:mpar.maxlag
    IRF_state_sparse(:,t)=(MX*x)';
    x=hx*x;
end

IRF_distr=IRF_state_sparse(indexes.distr,1:mpar.maxlag);
for t=1:mpar.maxlag
    K_IRF(t)=meshes.m(:)'*IRF_distr(:,t)+grid.K;
end

r_IRF=exp(log(1+par.R)+IRF_state_sparse(indexes.R,:));
Z_IRF=IRF_state_sparse(indexes.S,:);

%% Now solve by MIT Shock
disp('Now solve TFP shock via MIT shock method')

tic
TT=mpar.maxlag;

ZT = [1; exp(Z_IRF(1:end-1))'];
rT = [1+par.R; r_IRF(1:end-1)'];
KT = [targets.K; K_IRF(1:end-1)'];
NT =  (ZT.*par.alpha.*KT.^(1-par.alpha)).^(1/(1-par.alpha+par.gamma));
KNratioT = KT./NT;

%%
diffKNT=10e10;
while diffKNT>1e-5
    
    wT =  ZT.*par.alpha.* (KNratioT).^(1-par.alpha);
    
    KT = ((KNratioT.^-1)./((ZT.*par.alpha).^(1./(1-par.alpha+par.gamma)))).^(-(1-par.alpha+par.gamma)/par.gamma);
    NT =  (ZT.*par.alpha.*KT.^(1-par.alpha)).^(1/(1-par.alpha+par.gamma));
    
    SPT(:,:,1) = m_star;
    CPT(:,:,1) = c_guess;
    SPT(:,:,TT) = m_star;
    CPT(:,:,TT) = c_guess;
    
    % Backwards iteration of policy
    for h = 1:(TT-2)
        t = TT-h;
        
        r = rT(t);
        w = wT(t);
        N = NT(t);
        
        NW=par.gamma/(1+par.gamma).*N.*w;
        WW=NW*ones(mpar.nm,mpar.nh); %Wages
        inc.labor   = WW.*meshes.h;
        inc.money   = r.*meshes.m;
        inc.profits = 0;
        
        mutil_c = 1./(CPT(:,:,t+1).^par.xi); % marginal utility at consumption policy no adjustment
        EVm = reshape(reshape((rT(t+1)).*mutil_c,[mpar.nm mpar.nh])*P_H',[mpar.nm, mpar.nh]);% Expected marginal utility at consumption policy
        
        % Update policy
        [CPT(:,:,t),SPT(:,:,t)]=EGM_policyupdate(EVm,r-1,1,inc,meshes,grid,par,mpar);
        
    end
    
    % Forward Iteration of Distribution
    DIST(:,1)  = joint_distr(:);
    DIST(:,TT) = joint_distr(:);
    
    for t=2:TT
        
        [H]=Gen_BigTransH(SPT(:,:,t-1), P_H, mpar, grid);
        
        DIST(:,t)=DIST(:,t-1)'*H;
        
    end
    
    % Update price vector
    KTUpdate  = DIST'*meshes.m(:);
    NTUpdate =  (ZT.*par.alpha.*KTUpdate.^(1-par.alpha)).^(1/(1-par.alpha+par.gamma));
    
    KNTUpdate=KTUpdate./NTUpdate;
    
    diffKNT = max(abs(KNTUpdate(:)-KNratioT(:)));
    
    KNratioT=0.95*KNratioT+0.05*KNTUpdate;
    
    rT = ZT.* (1-par.alpha).* (KNratioT.^-1).^(par.alpha)- par.delta + 1;
    
end
toc
%% Compare MIT and Perturbation solution
disp('Compare Perturbation and MIT shock results')

figure
plot(KTUpdate)
hold on
plot([targets.K; K_IRF(1:end-1)'])

figure
plot(100*(KTUpdate./[targets.K; K_IRF(1:end-1)']-1))

%% Simulate MIT shock solution

Kdev=log(KTUpdate./targets.K);
KFLip = flip(Kdev,1);

zshocks=randn(10000,1)*par.sigmaS;

approxlength=1; % truncation parameter
for t=2:1000
    Ksim1(:,t-1) = KFLip(TT-approxlength+1:end)'*zshocks(t+1:t+approxlength);
end

approxlength=10;
for t=2:1000
    Ksim10(:,t-1) = KFLip(TT-approxlength+1:end)'*zshocks(t+1:t+approxlength);
end

approxlength=100;
for t=2:1000
    Ksim100(:,t-1) = KFLip(TT-approxlength+1:end)'*zshocks(t+1:t+approxlength);
end

figure
plot(100*Ksim1)
hold on
plot(100*Ksim10)
hold on
plot(100*Ksim100)
