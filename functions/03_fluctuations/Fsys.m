function [Difference,LHS,RHS,JD_new,c_star,m_star,P]  = Fsys(State,Stateminus,...
    Control_sparse,Controlminus_sparse,StateSS, ControlSS,...
    par,mpar,grid,meshes,indexes,P,aggrshock)
% System of equations written in Schmitt-Grohé-Uribe generic form with states and controls
% STATE: Vector of state variables t+1 (only marginal distributions for histogram)
% STATEMINUS: Vector of state variables t (only marginal distributions for histogram)
% CONTROL: Vector of state variables t+1 (only coefficients of sparse polynomial)
% CONTROLMINUS: Vector of state variables t (only coefficients of sparse polynomial)
% STATESS and CONTROLSS: Value of the state and control variables in steady
% state. For the Value functions these are at full grids.
% GAMMA_STATE: Mapping such that perturbationof marginals are still
% distributions (sum to 1).
% PAR, MPAR: Model and numerical parameters (structure)
% GRID: Asset and productivity grid
% COPULA: Interpolant that allows to map marginals back to full-grid
% distribuitions
% P: steady state productivity transition matrix
% aggrshock: sets whether the Aggregate shock is TFP or uncertainty
%
% =========================================================================
% Part of the Matlab code to accompany the paper
% 'Solving heterogeneous agent models in discrete time with
% many idiosyncratic states by perturbation methods', CEPR Discussion Paper
% DP13071
% by Christian Bayer and Ralph Luetticke
% http://www.erc-hump-uni-bonn.net/
% =========================================================================


%% Initializations
mutil = @(c)(1./(c.^par.xi));
invmutil = @(mu)((1./mu).^(1/par.xi));

% Number of states, controls
nx   = mpar.numstates; % Number of states
ny   = mpar.numcontrols; % number of Controls

% Initialize LHS and RHS
LHS  = zeros(ny+nx,1);
RHS  = zeros(ny+nx,1);

%% Control Variables (Change Value functions according to sparse polynomial)
Control       = ControlSS + Control_sparse;
Controlminus  = ControlSS + Controlminus_sparse;

%% State Variables
% read out marginal histogramm in t+1, t
Distribution      = StateSS(indexes.distr) +  State(indexes.distr);
Distributionminus = StateSS(indexes.distr) +  Stateminus(indexes.distr);

% Aggregate Exogenous States
S       = StateSS(indexes.S) + (State(indexes.S));
Sminus  = StateSS(indexes.S) + (Stateminus(indexes.S));

%% Split the Control vector into items with names
% Controls
mutil_c        = mutil((Control(indexes.c)));

% Aggregate Controls (t+1)
K  = exp(Control(indexes.K ));
Q  = exp(Control(indexes.Q ));
R  = exp(Control(indexes.R ));

% Aggregate Controls (t)
Qminus  = exp(Controlminus(indexes.Q ));
Yminus  = exp(Controlminus(indexes.Y ));
Wminus  = exp(Controlminus(indexes.W ));
Nminus  = exp(Controlminus(indexes.N ));

Rminus  = exp(Controlminus(indexes.R ));
Kminus  = exp(Controlminus(indexes.K ));
cminus  = Controlminus(indexes.c);

%% Write LHS values
% Controls
LHS(nx+indexes.c)       =  cminus;
LHS(nx+indexes.Q)       = (Qminus);
LHS(nx+indexes.Y)       = (Yminus);
LHS(nx+indexes.W)       = (Wminus);
LHS(nx+indexes.N)       = (Nminus);
LHS(nx+indexes.R)       = (Rminus);
LHS(nx+indexes.K)       = (Kminus);

% States
% Joint Distribution
LHS(indexes.distr) = Distribution(1:mpar.nm*mpar.nh);

LHS(indexes.S)          = (S);

%% Set of Differences for exogenous process
RHS(indexes.S) = (par.rhoS * (Sminus));

switch(aggrshock)
    case('TFP')
        TFP=exp(Sminus);
end

marginal_mminus = sum(reshape(Distributionminus,[mpar.nm mpar.nh]),2);
marginal_hminus = squeeze(sum(reshape(Distributionminus,[mpar.nm mpar.nh]),1));

RHS(nx+indexes.K)= grid.m(:)'*marginal_mminus(:);

%% Update controls
RHS(nx+indexes.Y) = (TFP*(Nminus).^(par.alpha).*Kminus.^(1-par.alpha));
RHS(nx+indexes.N) = (TFP*par.alpha.*Kminus^(1-par.alpha)).^(1/(1-par.alpha+par.gamma));

% Wage Rate
RHS(nx+indexes.W) = TFP *par.alpha.* (Kminus./(Nminus)).^(1-par.alpha);
% Return on Capital
RHS(nx+indexes.R) = TFP *(1-par.alpha).* ((Nminus)./Kminus).^(par.alpha)  - par.delta;

RHS(nx+indexes.Q)=(par.phi*(K./Kminus-1)+1);

%% Wages net of leisure services
WW=par.gamma/(1+par.gamma)*Nminus.*Wminus*ones(mpar.nm,mpar.nh);

%% Incomes (grids)
inc.labor   = WW.*(meshes.h);
inc.money   = (Rminus+Qminus)*meshes.m;
inc.profits  = 1/2*par.phi*((K-Kminus).^2)./Kminus;

%% Update policies
Raux = (R+Q)/(Qminus); % Expected marginal utility at consumption policy (w &w/o adjustment)
EVm = reshape(reshape(Raux(:).*mutil_c,[mpar.nm mpar.nh])*P',[mpar.nm, mpar.nh]);

[c_star,m_star] = EGM_policyupdate(EVm,Rminus,Qminus,inc,meshes,grid,par,mpar);

RHS(nx+indexes.c) = c_star(:); % Write Marginal Utility to RHS of F

%% Update distribution
[H]=Gen_BigTransH(m_star, P, mpar, grid);

JD_new=Distributionminus(:)'*H;

RHS(indexes.distr) = JD_new(:); %Leave out last state


%% Difference from SS
Difference=((LHS-RHS));


end
