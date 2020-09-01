function [H]=Gen_BigTransH(m_star, P_H, mpar, grid)

weight11  = zeros(mpar.nm, mpar.nh,mpar.nh);
weight12  = zeros(mpar.nm, mpar.nh,mpar.nh);

% Adjustment case
[Dist_m,idm] = genweight(m_star,grid.m);

idm=repmat(idm(:),[1 mpar.nh]);
idh=kron(1:mpar.nh,ones(1,mpar.nm*mpar.nh));

index11 = sub2ind([mpar.nm mpar.nh],idm(:),idh(:));
index12 = sub2ind([mpar.nm mpar.nh],idm(:)+1,idh(:));

for hh=1:mpar.nh
    
    %Corresponding weights
    weight11_aux = (1-Dist_m(:,hh));
    weight12_aux =  (Dist_m(:,hh));
    
    % Dimensions (mxk,h',h)
    weight11(:,:,hh)=weight11_aux(:)*P_H(hh,:);
    weight12(:,:,hh)=weight12_aux(:)*P_H(hh,:);
end

weight11=permute(weight11,[1 3 2]);
weight12=permute(weight12,[1 3 2]);

rowindex=repmat(1:mpar.nm*mpar.nh,[1 2*mpar.nh]);

H=sparse(rowindex,[index11(:); index12(:)],...
    [weight11(:); weight12(:)],mpar.nm*mpar.nh,mpar.nm*mpar.nh); % mu'(h',k'), a without interest

end
function [ weight,index ] = genweight( x,xgrid )
% function: GENWEIGHT generates weights and indexes used for linear interpolation
%
% X: Points at which function is to be interpolated.
% xgrid: grid points at which function is measured
% no extrapolation allowed
[~,index] = histc(x,xgrid);
index(x<=xgrid(1))=1;
index(x>=xgrid(end))=length(xgrid)-1;

weight = (x-xgrid(index))./(xgrid(index+1)-xgrid(index)); % weight xm of higher gridpoint
weight(weight<=0) = 1.e-16; % no extrapolation
weight(weight>=1) = 1-1.e-16; % no extrapolation

end  % function