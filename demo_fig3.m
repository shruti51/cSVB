% This code aims to reproduce the results corresponding to Fig.3
% of the below mentioned paper which proposes the correlated sparse
% variational bayes framework for block sparse signal recovery. 
%
% Author: Shruti Sharma
% Paper: Shruti Sharma, Santanu Chaudhury, Jayadeva, ' Variational
% Bayes Block Sparse Modeling with Correlated Entries', ICPR-2018.

% To reproduce Fig.3, we set status=1 which corresponds to signal with SNR=25dB.

% Value of \rho can be varied from 0 to 1. In the paper, we have used \rho=0,0.7 and 0.99.
% We have used 1000 independent trials and Comparisons are made w.r.t. BSBL-BO, BSBL-EM 
% Reference: Zhang Z., Rao B.D., 'Extension of SBL Algorithms for the Recovery of Block Sparse
% Signals With Intra-Block Correlation', IEEE Trans. on Sig. Proc., vol. 1, no. 8, pp. 2009-2015. 2013)
% and the variants of sparse variational bayes SVB 
% Reference: Babacan S.D., et al.,'Bayesian Group-Sparse Modeling and Variational 
% Inference',IEEE Trans. on  Sig. Proc., vol. 62, no. 11, pp. 2906-2921, 2014).
%
% Codes for  BSBL-BO and BSBL-EM can be downloaded from
% https://sites.google.com/site/researchbyzhang/publications.


clear
close all
rng('default')
N=480; no_of_nonzero_coeff=24;
status=1;
if status==0
    threshold=1e-5;
else
    threshold=1e-1;
end
M=50;

for rho=[0 0.7 0.99]
    rho
    succ_cJSVB=zeros(4,1); succ_cLSVB=zeros(4,1); succ_cStSVB=zeros(4,1);
    succ_JSVB=zeros(4,1); succ_LSVB=zeros(4,1); succ_StSVB=zeros(4,1);
    % succ_BSBL_BO=zeros(4,1); succ_BSBL_EM=zeros(4,1);
    p=0;
    
    for block_size=[2 3 4 6]
        block_size
        p=p+1;
        % Number of non-zero blocks
        K = no_of_nonzero_coeff / block_size;
        
        % 100 independent trials loop
        for k=1:100
            k
            % Correlated Input Generation Block using AR(1) process with
            % correlation coefficient \rho
            [x_gen,actual_supp]=correlated_input(rho,N,no_of_nonzero_coeff,block_size);
            
            group_ind=reshape(ones(block_size,1)*(1:N),N*block_size,1);
            grouping=cell(1,125);
            for i=1:N
                grouping{i}=find(ismember(group_ind,i));
            end
            supp_actual=[grouping{actual_supp}];
            
            Phi=randn(M,N);
            Phi = Phi./(ones(M,1)*sqrt(sum(Phi.^2)));
            Signal=Phi*x_gen;
            
            if status==0
                y=Signal;
            else
                SNR=25;
                std_noise=std(Signal*10^(-SNR/20));
                v=randn(M,1)*std_noise;
                y=Signal+ v;
            end
            
%             % BSBL-EM
%             groupStartLoc = 1:group_size:N;
%             Result_BSBL_EM = BSBL_EM(Phi,y,groupStartLoc,2);
%             a = norm( Result_BSBL_EM.x - x_gen , 'fro') / norm( x_gen , 'fro');
%             supp_1=setdiff(Result_BSBL_EM.gamma_used,actual_supp);
%             supp_2=setdiff(actual_supp,Result_BSBL_EM.gamma_used);
%             if isempty(supp_1)==1 && isempty(supp_2)==1
%                 succ_BSBL_EM(p)=succ_BSBL_EM(p)+1;
%             else
%                 t=intersect(Result_BSBL_EM.gamma_used,actual_supp);
%                 if length(t)==K
%                     if a < threshold
%                         succ_BSBL_EM(p)=succ_BSBL_EM(p)+1;
%                     end
%                 end
%             end
%             
%             % BSBL-BO
%             Result_BSBL_BO = BSBL_BO(Phi,y,groupStartLoc,2);
%             a = norm( Result_BSBL_BO.x - x_gen , 'fro') / norm( x_gen , 'fro');
%             supp_1=setdiff(Result_BSBL_BO.gamma_used,actual_supp);
%             supp_2=setdiff(actual_supp,Result_BSBL_BO.gamma_used);
%             if isempty(supp_1)==1 && isempty(supp_2)==1
%                 succ_BSBL_BO(p)=succ_BSBL_BO(p)+1;
%             else
%                 t=intersect(Result_BSBL_BO.gamma_used,actual_supp);
%                 if length(t)==K
%                     if a < threshold
%                         succ_BSBL_BO(p)=succ_BSBL_BO(p)+1;
%                     end
%                 end
%             end
          

            %% Jeffrey cSVB 
            [Result_cJ]=CSVB(Phi,y,grouping,block_size,status,1);
            a = norm(Result_cJ.x - x_gen , 'fro') / norm( x_gen , 'fro');
            supp_1=setdiff(Result_cJ.z,supp_actual);
            supp_2=setdiff(supp_actual,Result_cJ.z);
            if isempty(supp_1)==1 && isempty(supp_2)==1
                succ_cJSVB(p)=succ_cJSVB(p)+1;
            else
                t=intersect(Result_cJ.z,supp_actual);
                if length(t)==block_size*K
                    if a <threshold
                        succ_cJSVB(p)=succ_cJSVB(p)+1;
                    end
                end
            end
            
            %Jeffrey SVB
            [Result_J]=SVB(Phi,y,grouping,block_size,status,1);
            a = norm(Result_J.x - x_gen , 'fro') / norm( x_gen , 'fro');
            supp_1=setdiff(Result_J.z,supp_actual);
            supp_2=setdiff(supp_actual,Result_J.z);   
            if isempty(supp_1)==1 && isempty(supp_2)==1
                succ_JSVB(p)=succ_JSVB(p)+1;
            else
                t=intersect(Result_J.z,supp_actual);
                if length(t)==block_size*K
                    if a< threshold
                        succ_JSVB(p)=succ_JSVB(p)+1;
                    end
                end
            end
                        
            %% Laplace cSVB
            [Result_cL]=CSVB(Phi,y,grouping,block_size,status,2);
            a = norm(Result_cL.x - x_gen , 'fro') / norm( x_gen , 'fro');
            supp_1=setdiff(Result_cL.z,supp_actual);
            supp_2=setdiff(supp_actual,Result_cL.z);
            if isempty(supp_1)==1 && isempty(supp_2)==1
                succ_cLSVB(p)=succ_cLSVB(p)+1;
            else
                t=intersect(Result_cL.z,supp_actual);
                if length(t)==block_size*K
                    if a < threshold
                        succ_cLSVB(p)=succ_cLSVB(p)+1;
                    end
                end
            end
            
            % Laplace SVB
            [Result_L]=SVB(Phi,y,grouping,block_size,status,2);
            a = norm(Result_L.x - x_gen , 'fro') / norm( x_gen , 'fro');
            supp_1=setdiff(Result_L.z,supp_actual);
            supp_2=setdiff(supp_actual,Result_L.z);
            if isempty(supp_1)==1 && isempty(supp_2)==1
                succ_LSVB(p)=succ_LSVB(p)+1;
            else
                t=intersect(Result_L.z,supp_actual);
                if length(t)==block_size*K
                    if a < threshold
                        succ_LSVB(p)=succ_LSVB(p)+1;
                    end
                end
            end
            
            %%  Student cSVB
            [Result_cSt]=CSVB(Phi,y,grouping,block_size,status,3);
            a= norm( Result_cSt.x - x_gen , 'fro') / norm( x_gen , 'fro');
            supp_1=setdiff(Result_cSt.z,supp_actual);
            supp_2=setdiff(supp_actual,Result_cSt.z);
            if isempty(supp_1)==1 && isempty(supp_2)==1
                succ_cStSVB(p)=succ_cStSVB(p)+1;
            else
                t=intersect(Result_cSt.z,supp_actual);
                if length(t)==block_size*K
                    if a < threshold
                        succ_cStSVB(p)=succ_cStSVB(p)+1;
                    end
                end
            end            
            
            % Student SVB
            [Result_St]=SVB(Phi,y,grouping,block_size,status,3);
            a = norm(Result_St.x - x_gen , 'fro') / norm( x_gen , 'fro');
            supp_1=setdiff(Result_St.z,supp_actual);
            supp_2=setdiff(supp_actual,Result_St.z);
            
            if isempty(supp_1)==1 && isempty(supp_2)==1
                succ_StSVB(p)=succ_StSVB(p)+1;
            else
                t=intersect(Result_St.z,supp_actual);
                if length(t)==block_size*K
                    if a < threshold
                        succ_StSVB(p)=succ_StSVB(p)+1;
                    end
                end
            end
            
        end
        %succ=[succ_BSBL_EM succ_BSBL_BO succ_cJSVB succ_JSVB succ_cLSVB succ_LSVB succ_cStSVB succ_StSVB];
        succ=[succ_cJSVB succ_JSVB succ_cLSVB succ_LSVB succ_cStSVB succ_StSVB];
    end
    
    %succ=[zeros(1,8); succ_BSBL_EM succ_BSBL_BO succ_cJSVB succ_JSVB succ_cLSVB succ_LSVB succ_cStSVB succ_StSVB];
    
    % ============ RESULTS =============
    succ=[zeros(1,6); succ_cJSVB succ_JSVB succ_cLSVB succ_LSVB succ_cStSVB succ_StSVB]
    
    d=[1 2 3 4 6];
    
    FigHandle = figure;
    set(FigHandle, 'Position', [100, 100, 520, 380]);
    plot(d,(1-(succ(:,1)/k)),'-.b*');
    hold on;
    plot(d,(1-(succ(:,2)/k)),'b--o');
    hold on;
    plot(d,(1-(succ(:,3)/k)),'-.g*');
    hold on;
    plot(d,(1-(succ(:,4)/k)),'g--o');
    hold on;
    plot(d,(1-(succ(:,5)/k)),'-.k*');
    hold on;
    plot(d,(1-(succ(:,6)/k)),'k--o');
    
    a_t=title(sprintf('rho=%d',rho));
    set(a_t,'FontSize',12,'FontWeight','bold')
    a_l=legend({'cJSVB';'JSVB' ; 'cLSVB'; 'LSVB'; 'cStSVB'; 'StSVB' });
    set(a_l,'FontSize',10,'FontWeight','bold');
    a_y=ylabel(' Failure Rate');
    set(a_y,'FontSize',12,'FontWeight','bold');
    xlabel('Group Size');
    set(gca,'XTick', [1,2,3,4,6], 'FontSize',12,'FontWeight','bold');
    saveas(FigHandle,strcat('fig3_rho=',num2str(rho),'.png'));
end