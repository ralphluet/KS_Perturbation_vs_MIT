function [hx,gx,F1,F2,F3,F4,par] = SGU_solver(F, mpar,par)
% This funtion solves for a competitive equilibrium defined as a zero of the
% function F which is written in Schmitt-Grohé Uribe form using the algorithm
% suggested  in Schmit-Grohé and Uribe (2004): "Solving dynamic general equilibrium models
% using a second-order approximation to the policy function"
%
%now do numerical linearization
State       = zeros(mpar.numstates,1);
State_m     = State;
Contr       = zeros(mpar.numcontrols,1);
Contr_m     = Contr;

[Fb,~,~,~]  = F(State,State_m,Contr,Contr_m);


%Use Schmitt Gohe Uribe Algorithm
% E[x' u']' =inv(A)*B*[x u]'
% A = [dF/dx' dF/du'], B =[dF/dx dF/du]
% A = [F1 F2]; B=[F3 F4]
F1 = zeros(mpar.numstates+mpar.numcontrols,mpar.numstates); %Tomorrows states do not affect error on controls and have unit effect on state error
F2 = zeros(mpar.numstates+mpar.numcontrols,mpar.numcontrols); %Jacobian wrt tomorrow's controls (TO BE FILLED)
F3 = zeros(mpar.numstates+mpar.numcontrols,mpar.numstates); % Jacobian wrt today's states (TO BE FILLED)
F4 = zeros(mpar.numstates+mpar.numcontrols,mpar.numcontrols); % Jacobian wrt today's controls (TO BE FILLED)

% Absolute deviations
par.scaleval1 = 1e-6; %vector of numerical differentiation step sizes

cc = zeros(mpar.numcontrols,1);
ss = zeros(mpar.numstates,1);

disp('Computing Jacobian F1=DF/DXprime F3 =DF/DX')
for i=1:mpar.numstates
    
    X = zeros(mpar.numstates,1);
    h = par.scaleval1;
    X(i) = h;
    Fx = F(ss,X,cc,cc);
    F3(:,i)= (Fx-Fb)/h;
    Fx = F(X,ss,cc,cc);
    F1(:,i)= (Fx-Fb)/h;
end

disp('Computing Jacobian F2=DF/DYprime')
for i=1:mpar.numcontrols
    
    Y = zeros(mpar.numcontrols,1);
    h = par.scaleval1;
    Y(i) = h;
    Fy = F(ss,ss,Y,cc);
    F2(:,i)= (Fy-Fb)/h;
end

disp('Computing Jacobian F4=DF/DY')
for i=1:mpar.numcontrols
    
    Y = zeros(mpar.numcontrols,1);
    h = par.scaleval1;
    Y(i) = h;
    Fy = F(ss,ss,cc,Y);
    F4(:,i)= (Fy-Fb)/h;
end

%% Use linear time iteration to solve RE eq

A = [F3,zeros(mpar.numstates+mpar.numcontrols,mpar.numcontrols)];
B = [F1,F4];
C = [zeros(mpar.numstates+mpar.numcontrols,mpar.numstates),F2];

F0 = zeros(size(C));
diff1=1000;
i=1;

while abs(diff1)>1e-6 && i<1000
    F_1 = (B+C*F0)\(-A);
    diff1=max(max(abs(F_1-F0)));
    F0=F_1;
    i=i+1;
end

gx = F_1(mpar.numstates+1:end,1:mpar.numstates);
hx = F_1(1:mpar.numstates,1:mpar.numstates);


end
