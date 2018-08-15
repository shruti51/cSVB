function [x,index]=correlated_input(beta,N,no_of_nonzero_coeff,group_size)




x=zeros(N,1);

% No of non zero groups
K = no_of_nonzero_coeff / group_size; 
no_of_groups=N/group_size;

groups = reshape( ([1:no_of_groups]'*ones(1,group_size))' ,[N,1]);
ar_coeff=ones(K,1)*(beta); 
coeff=[];
coeff(:,1)=randn(K,1);

for i=2:group_size*100
    % AR(1) model: y(k)=-ay(k-1)+w(k)
    coeff(:,i)= ar_coeff.*coeff(:,i-1)+sqrt(1-ar_coeff.^2).*randn(K,1);      
end
coeff=coeff(:,end-group_size+1:end);

% Normalize each row
coeff=coeff./(sqrt(sum(coeff.^2,2))*ones(1,group_size));

% Rescale each row such that the squared row-norm distributes in [1,scalefactor]
mag = rand(1,K);        % uniformly distributed r.v. [0,1]

% (max'-min')(v-min)/(max-min) +min' to make r.v. in the range [1,3] [min',max']
mag=2*(mag-min(mag))/(max(mag)-min(mag)) + 1 ;

% l2 norm of each source is scaled to be in the range [1,3]
coeff = diag(sqrt(mag)) * coeff;

ind = randperm(no_of_groups)';
index = ind(1:K);
ind = ismember(groups, index);
x(ind) = coeff';

end