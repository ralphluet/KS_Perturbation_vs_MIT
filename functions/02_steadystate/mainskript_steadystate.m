
%% Setup grids
grid.K  = 60;

% Quadruble Log Grid
m_min = 0;
m_max = 15*grid.K;
grid.m = (exp(exp(exp(exp(linspace(0,log(log(log(log(m_max - m_min+1)+1)+1)+1),mpar.nm))-1)-1)-1)-1+m_min);

%% Use Tauchen method to approximate state space

% Generate transition probabilities and grid
[hgrid,P_H,grid.boundsH] = Tauchen(par.rhoH,mpar.nh,1, 0, 'importance'); % LR variance = 1
% Correct long run variance for *human capital*
hgrid               = hgrid*par.sigmaH/sqrt(1-par.rhoH^2);
grid.h              = exp(hgrid)./mean(exp(hgrid)); % Levels instead of Logs and normalize to 1

%     Layout of matrices:
%       Dimension 1: capital k
%       Dimension 2: stochastic human capital h

[meshes.m,meshes.h] = ndgrid(grid.m,grid.h);

%% Solve for steady state

[c_guess,m_star,joint_distr,R,W,Y,N]=steadystate(P_H,grid,mpar,par,meshes);

%% Calculate steady state capital and further statistics

clear targets
grid.K     = grid.m*(sum(joint_distr,2));
targets.K  = grid.m*(sum(joint_distr,2));
targets.KY = targets.K/Y;
targets.Y  = Y;

par.W=W;
par.N=N;
par.R=R;

