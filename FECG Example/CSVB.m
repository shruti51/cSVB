function [Result]=CSVB(Phi,y,grouping,block_size,status,marg)

% CSVB recovers block sparse signal using gaussian scale mixture
% parameterized by some scalar random parameters and deterministic matrices
% to model correlation within the block.

% ==========INPUTS=============
% Phi: M X N Gaussian matrix
% y: Observation vector
% grouping: grouping information
% block_size: size of each block (Here block size is assumed to be same fo all the blocks)
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
    prune_z=1e+12;
elseif status==1 % SNR=25dB
    prune_z=1e+3;
    beta=1e+3;
end

[M,N]=size(Phi);
G=N/block_size;
x=Phi\y; count=0;
z=ones(G,1);
list=1:G; usable_list=G;
Sigma0_inv=repmat(eye(block_size),[1,1,G]);
k_beta=1e-6; theta_beta=1e-6;
k_a=1e-6; theta_a=1e-6; a_0=1;a=a_0*ones(N,1);
lambda=1;

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
        Sigma0_inv=Sigma0_inv(:,:,index);
        % post processing
        list=list(index);
    end
    
    PSP=0;
    for i=1:usable_list
        PSP=PSP+Phi(:,grouping{i})*Sigma0_inv(:,:,i)*Phi(:,grouping{i})';
    end
    
    PtH=Phi'/(1/beta * eye(M) + PSP);
    PtHy=PtH*y;
    PtHP=PtH*Phi;
    x_old=x;
    x=zeros(usable_list*block_size,1);
    Sigma_x=repmat(zeros(block_size),[1,1,usable_list]);
    B_inv=0;
    Cov_x=Sigma_x;
    
    % Solution Vector and Covariance Matrix Update
    for i=1:usable_list
        x(grouping{i})= Sigma0_inv(:,:,i)*PtHy(grouping{i});
        Sigma_x(:,:,i)=Sigma0_inv(:,:,i)-Sigma0_inv(:,:,i)* PtHP(grouping{i},grouping{i})* Sigma0_inv(:,:,i);
        Cov_x(:,:,i)=Sigma_x(:,:,i)+x(grouping{i})*x(grouping{i})';
        B_inv=B_inv+(Cov_x(:,:,i)*z(i));
    end
   
    % Deterministic Parameter Update
    b = (mean(diag(B_inv,1))/mean(diag(B_inv)));
    if abs(b) >= 0.99
        b = 0.99*sign(b);
    end
    bs = [];
    for j = 1 : block_size
        bs(j) = (b)^(j-1);
    end
    B_inv=toeplitz(bs);
    B = inv(B_inv);
    
    % Random Scalar Parameter Updates 
    for i=1:usable_list
        if marg==1
            z(i)=block_size/(trace(B*Cov_x(:,:,i)));
        elseif marg==2
            z(i)=sqrt(a(i))/(sqrt(trace(B*Cov_x(:,:,i)))+eps);
            z_inv(i)=1/z(i) + 1/a(i);
        elseif marg==3
            z(i)=(2*lambda + block_size )/(a(i)+(trace(B*Cov_x(:,:,i)))+eps);
        end
        Sigma0_inv(:,:,i)= 1/z(i) * B_inv;
    end
    
    % Hyper-parameter Updates
    if marg==2
        a=(k_a+(block_size+1)/2)./(theta_a+z_inv./2);
    elseif marg==3
        a=(k_a+lambda)./(theta_a+z/2);
    end
    
    % Learn beta for SNR=25dB
    if status==1 
        PZP=Phi*kron(diag(1/z),eye(block_size))*Phi';
        beta=(M + 2*k_beta)/(2*theta_beta + norm(y-Phi*x,2)+(1/beta)*block_size*trace(PZP/((1/beta)*eye(M)+PZP)));
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
    if (count >= 40) 
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

% Outputs
Result.x=x_est;
Result.count=count;
Result.z=z_ind;
end