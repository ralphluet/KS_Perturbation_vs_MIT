function [c_guess,k_star,joint_distr,R,W,Y,N]=steadystate(P_H,grid,mpar,par,meshes)

KL=0.5*grid.K;
KH=2*grid.K;

[excessL]=excessK(KL, grid,P_H,mpar,par,meshes);

[excessH]=excessK(KH, grid,P_H,mpar,par,meshes);

if sign(excessL)==sign(excessH)
    disp('ERROR! Sign not diff')
end

%% Brent Method To Find Eq Capital Stock
fa=excessL; fb=excessH;
a=KL;
b=KH;

if fa*fb>0
    error('f(a) und f(b) sollten unterschiedliche Vorzeichen haben');
end

c=a; fc=fa;   %Zu Beginn ist c = a

c=a; fc=fa; d=b-a; e=d;

iter=0;
maxiter=1000;

while iter<maxiter
    iter=iter+1;
    
    if fb*fc>0
        c=a; fc=fa; d=b-a; e=d;
    end
    
    if abs(fc)<abs(fb)
        a=b; b=c; c=a;
        fa=fb; fb=fc; fc=fa;
    end
    
    tol=2*eps*abs(b)+mpar.crit; m=(c-b)/2; %Toleranz
    
    if (abs(m)>tol) && (abs(fb)>0) %Verfahren muss noch durchgeführt werden
        
        if (abs(e)<tol) || (abs(fa)<=abs(fb))
            d=m; e=m;
        else
            s=fb/fa;
            if a==c
                p=2*m*s; q=1-s;
            else
                q=fa/fc; r=fb/fc;
                p=s*(2*m*q*(q-r)-(b-a)*(r-1));
                q=(q-1)*(r-1)*(s-1);
            end
            if p>0
                q=-q;
            else
                p=-p;
            end
            s=e; e=d;
            if ( 2*p<3*m*q-abs(tol*q) ) && (p<abs(s*q/2))
                d=p/q;
            else
                d=m; e=m;
            end
        end
        
        a=b; fa=fb;
        
        if abs(d)>tol
            b=b+d;
        else
            if m>0
                b=b+tol;
            else
                b=b-tol;
            end
        end
    else
        break;
    end
    
    [fb,c_guess]=excessK(b, grid,P_H,mpar,par,meshes);
    
end

Kcand=b;

%% Update

[excess,c_guess,k_star,joint_distr,R,W,Y,N]=excessK(Kcand, grid,P_H,mpar,par,meshes);

end
function [excess,c_guess,k_star,joint_distr,R,W,Y,N]=excessK(K, grid,P_H,mpar,par,meshes)

%% Returns
N =  (par.alpha.*K^(1-par.alpha)).^(1/(1-par.alpha+par.gamma));
W =  par.alpha.* (K./N).^(1-par.alpha);
R = (1-par.alpha).* (N./K).^(par.alpha)- par.delta;

Y = N.^(par.alpha).*K.^(1-par.alpha);

%% Policy guess
NW=par.gamma/(1+par.gamma).*N.*W; % GHH adjustment

inc.labor   = NW.*meshes.h;
inc.money   = R.*meshes.m;
inc.profits = 0;

% Consumption guess
c_guess = inc.labor + max(inc.money,0);

%% Solve Policies
distC=9999;
while max([distC])>mpar.crit
    
    mutil_c = 1./(c_guess.^par.xi); % marginal utility at consumption policy no adjustment
    EVm = reshape(reshape((1+R).*mutil_c,[mpar.nm mpar.nh])*P_H',[mpar.nm, mpar.nh]);% Expected marginal utility at consumption policy
    
    % Update policy
    [c_new,k_star]=EGM_policyupdate(EVm,R,1,inc,meshes,grid,par,mpar);
    k_star(k_star>grid.m(end)) = grid.m(end); % No extrapolation
    
    % Check convergence of policy
    distC = max((abs(c_guess(:)-c_new(:))));
    
    % Update c policy guesses
    c_guess=c_new;
    
end

%% Solve Joint Distribution

[H]=Gen_BigTransH(k_star, P_H, mpar, grid);

distJD=9999;
countJD=1;
[joint_distr,~]=eigs(H',1);
joint_distr=joint_distr'./sum(joint_distr);

while (distJD>1e-14 || countJD<50) && countJD<10000
    
    joint_distr_next=joint_distr*H;
    joint_distr_next=full(joint_distr_next);
    joint_distr_next=joint_distr_next./sum(joint_distr_next);
    
    distJD=max((abs(joint_distr_next(:)-joint_distr(:))));
    
    countJD=countJD+1;
    joint_distr=joint_distr_next;
    
end

joint_distr=reshape(joint_distr,[mpar.nm mpar.nh]);

%% Update K
AggregateSavings=k_star(:)'*joint_distr(:);
excess=AggregateSavings-K;
end
