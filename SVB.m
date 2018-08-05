function [Result]=SVB(Phi,y,grouping,group_size,status,marg)

% SVB recovers block sparse signal using gaussian scale mixture
% parameterized by some scalar random parameters. 
% Reference: Babacan S.D., et al.,'Bayesian Group-Sparse Modeling and Variational 
% Inference',IEEE Trans. on  Sig. Proc., vol. 62, no. 11, pp. 2906-2921,
% 2014.

% ==========INPUTS=============
% Phi: M X N Gaussian matrix
% y: Observation vector
% grouping: grouping information
% block_size: size of each block (Here block size is assumed to be same for all the blocks)
% status: (1) If status = 0, we use noiseless setting where noise variance parameter beta is not learned.
%         (2) If status = 1, we use mildly noisy setting where SNR of signal is 25dB.
% marg:   (1) If marg = 1, update rules corresponding to marginal density Jeffery prior.
%         (2) If marg = 2, update rules corresponding to marginal density Laplace distribution.
%         (3) If marg = 3, update rules corresponding to marginal density Student's t distribution.

scl = std(y);
if (scl < 0.5) || (scl > 1)
    y = y/scl*0.5;
end

if status==0 %noiseless case
    beta=1e+12;
    prune_z=1e+3;
elseif status==1 % SNR=25dB
    prune_z=1e+3;
    beta=1e+3;
end

[M,N]=size(Phi);
G=N/group_size;
x=Phi\y;
z=ones(G,1);
list=1:G; usable_list=G; count=0;
k_beta=1e-6; theta_beta=1e-6;
k_a=1e-6; theta_a=1e-6; a_0=1;a=a_0*ones(N,1);
lambda=1;
z_inv=ones(G,1);

% Iteration
while (1)
    count=count+1;
    if (max(z) > prune_z)
        index=find(z < prune_z);
        usable_list=length(index);
        % prune gamma, and associated components in Phi and Sigma_0
        z=z(index);
        if marg==2
            z_inv=z_inv(index);
            a=a(index);
        elseif marg==3
            a=a(index);
        end
        Phi=Phi(:,[grouping{index}]);
        % post processing
        list=list(index);
    end
    
    x_old=x;
    Phi_t_Phi=Phi'*Phi;
    Phi_t_y=Phi'*y;
    Z = diag(z);
    
    % Soution Vector and Covariance Matrix Estimate
    x = (beta*Phi_t_Phi + kron(Z,eye(group_size)) )\(beta*Phi_t_y);
    Sigma_x = diag(1./diag(beta*Phi_t_Phi+ kron(Z,eye(group_size))));
    
    % Random Scalar Parameter Updates
    d_Sigma_x=diag(Sigma_x);
    z=zeros(usable_list,1);
    inv_z=zeros(usable_list,1);
    for g=1:usable_list
        xg_2=sum(x(grouping{g}).^2 + d_Sigma_x(grouping{g}) );
        if marg==1
            z(g)=group_size/(xg_2+eps);
        elseif marg==2
            z(g)=sqrt(a(g))/sqrt(xg_2+eps);
            inv_z(g) = 1/z(g) + 1/a(g);
        else 
            z(g)=(2*lambda + group_size )/(a(g)+xg_2+eps);
        end
    end
        
    % Hyper-parameter Updates
    if marg==2
        a=(k_a+(group_size+1)/2)./(theta_a + inv_z./2);
    elseif marg==3
        a=(k_a+lambda)./(theta_a+z/2);
    end
    
    % Learn beta for SNR=25dB
    if status==1 
        PZP=Phi*kron(diag(1/z),eye(group_size))*Phi';
        beta=(M + 2*k_beta)/(2*theta_beta + norm(y-Phi*x,2)+(1/beta)*group_size*trace(PZP/((1/beta)*eye(M)+PZP)));
    end
    
    % Convergence Criteria |x-x_old|<=10^{-8}
    if (size(x)==size(x_old))
        d_mu=max(max(abs(x-x_old)));
        %disp(['difference in mu is : ', num2str(d_mu)])
        if (d_mu<=1e-8)
            break;
        end
    end
    
    % Maximum Iteration Value to terminate the program 
    if (count >= 800)
       % disp('Reached max iteration. Stop \n \n');
        break;
    end
    
end

x_est=zeros(N,1);
z_ind=[grouping{list}];
x_est(z_ind)=x;
if (scl < 0.5) || (scl > 1)
    x_est = x_est * scl/0.5;
end

% =============OUTPUT===========
Result.x=x_est;
Result.count=count;
Result.z=z_ind;
end