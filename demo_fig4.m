% This code aims to reproduce the results corresponding to Fig.4
% of the below mentioned paper which proposes the correlated sparse
% variational bayes framework for block sparse signal recovery.
%
% Author: Shruti Sharma
% Paper: Shruti Sharma, Santanu Chaudhury, Jayadeva, ' Variational
% Bayes Block Sparse Modeling with Correlated Entries', ICPR-2018.
%
% To reproduce Fig.4, we set status=1 which corresponds to signal with
% SNR=25dB and block_size=3. Value of \rho can be varied from 0 to 1.
% In the paper, we have used \rho=0,0.7 and 0.99  and used 1000 independent trials
% to validate the results. Comparisons in the paper are made w.r.t. BSBL-BO, BSBL-EM
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
N=480; block_size=3; no_of_nonzero_coeff=24; p=1;
K = no_of_nonzero_coeff / block_size;
status=1;
if status==0
    threshold=1e-5;
else
    threshold=1e-1;
end

for rho=[0 0.7 0.99]
    rho
    succ=[]; err=[];
    
    for ratio=0.1:0.05:0.5
        ratio
        M=floor(N*ratio);
        succ_JTSVB=0; succ_LTSVB=0; succ_StTSVB=0;
        succ_JSVB=0; succ_LSVB=0; succ_StSVB=0;
        % succ_BSBL_BO=0; succ_BSBL_EM=0;
        err_Jeffrey_tsvb=0; err_Laplace_tsvb=0; err_St_tsvb=0;
        err_Jeffrey_svb=0; err_Laplace_svb=0; err_St_svb=0;
        % err_BSBL_BO=0; err_BSBL_EM=0;
        for k=1:100
            k
            % Correlated Input Generation Block using AR(1) process with
            % correlation coefficient
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
            
            %             groupStartLoc = 1:block_size:N;
            %
            %             Result1 = BSBL_BO(Phi,y,groupStartLoc,2);
            %             a = norm( Result1.x - x_gen , 'fro') / norm( x_gen , 'fro');
            %             supp_1=setdiff(Result1.gamma_used,actual_supp);
            %             supp_2=setdiff(actual_supp,Result1.gamma_used);
            %             if isempty(supp_1)==1 && isempty(supp_2)==1
            %                 succ_BSBL_BO=succ_BSBL_BO+1;
            %             else
            %                 t=intersect(Result1.gamma_used,actual_supp);
            %                 if length(t)==K
            %                     if a < threshold
            %                         succ_BSBL_BO=succ_BSBL_BO+1;
            %                     end
            %                 end
            %             end
            %             err_BSBL_BO =err_BSBL_BO+ a;
            %
            %             Result2 = BSBL_EM(Phi,y,groupStartLoc,2);
            %             a = norm( Result2.x - x_gen , 'fro') / norm( x_gen , 'fro');
            %             supp_1=setdiff(Result2.gamma_used,actual_supp);
            %             supp_2=setdiff(actual_supp,Result2.gamma_used);
            %             c_B_E=[c_B_E Result2.count];
            %             if isempty(supp_1)==1 && isempty(supp_2)==1
            %                 succ_BSBL_EM=succ_BSBL_EM+1;
            %             else
            %                 t=intersect(Result2.gamma_used,actual_supp);
            %                 if length(t)==K
            %                     if a < threshold
            %                         succ_BSBL_EM=succ_BSBL_EM+1;
            %                     end
            %                 end
            %             end
            %             err_BSBL_EM =err_BSBL_EM+ a;
            
            %% TSVB
            % Jeffrey cSVB
            [Result_cJ]=CSVB(Phi,y,grouping,block_size,status,1);
            a = norm(Result_cJ.x - x_gen , 'fro') / norm( x_gen , 'fro');
            supp_1=setdiff(Result_cJ.z,supp_actual);
            supp_2=setdiff(supp_actual,Result_cJ.z);
            if isempty(supp_1)==1 && isempty(supp_2)==1
                succ_JTSVB=succ_JTSVB+1;
            else
                t=intersect(Result_cJ.z,supp_actual);
                if length(t)==block_size*K
                    if a <threshold
                        succ_JTSVB=succ_JTSVB+1;
                    end
                end
            end
            err_Jeffrey_tsvb =err_Jeffrey_tsvb+ a;
            
            
            % Laplace cSVB
            [Result_cL]=CSVB(Phi,y,grouping,block_size,status,2);
            a = norm(Result_cL.x - x_gen , 'fro') / norm( x_gen , 'fro');
            supp_1=setdiff(Result_cL.z,supp_actual);
            supp_2=setdiff(supp_actual,Result_cL.z);
            if isempty(supp_1)==1 && isempty(supp_2)==1
                succ_LTSVB=succ_LTSVB+1;
            else
                t=intersect(Result_cL.z,supp_actual);
                if length(t)==block_size*K
                    if a < threshold
                        succ_LTSVB=succ_LTSVB+1;
                    end
                end
            end
            err_Laplace_tsvb =err_Laplace_tsvb+ a;
            
            %  Student cSVB
            [Result_cSt]=CSVB(Phi,y,grouping,block_size,status,3);
            a= norm( Result_cSt.x - x_gen , 'fro') / norm( x_gen , 'fro');
            supp_1=setdiff(Result_cSt.z,supp_actual);
            supp_2=setdiff(supp_actual,Result_cSt.z);
            if isempty(supp_1)==1 && isempty(supp_2)==1
                succ_StTSVB=succ_StTSVB+1;
            else
                t=intersect(Result_cSt.z,supp_actual);
                if length(t)==block_size*K
                    if a < threshold
                        succ_StTSVB=succ_StTSVB+1;
                    end
                end
            end
            err_St_tsvb =err_St_tsvb+ a;
            
            %% SVB
            %Jeffrey SVB
            [Result_J]=SVB(Phi,y,grouping,block_size,status,1);
            a = norm(Result_J.x - x_gen , 'fro') / norm( x_gen , 'fro');
            supp_1=setdiff(Result_J.z,supp_actual);
            supp_2=setdiff(supp_actual,Result_J.z);
            if isempty(supp_1)==1 && isempty(supp_2)==1
                succ_JSVB=succ_JSVB+1;
            else
                t=intersect(Result_J.z,supp_actual);
                if length(t)==block_size*K
                    if a< threshold
                        succ_JSVB=succ_JSVB+1;
                    end
                end
            end
            err_Jeffrey_svb=err_Jeffrey_svb+ a;
            
            % Laplace SVB
            [Result_L]=SVB(Phi,y,grouping,block_size,status,2);
            a = norm(Result_L.x - x_gen , 'fro') / norm( x_gen , 'fro');
            supp_1=setdiff(Result_L.z,supp_actual);
            supp_2=setdiff(supp_actual,Result_L.z);
            
            if isempty(supp_1)==1 && isempty(supp_2)==1
                succ_LSVB=succ_LSVB+1;
            else
                t=intersect(Result_L.z,supp_actual);
                if length(t)==block_size*K
                    if a < threshold
                        succ_LSVB=succ_LSVB+1;
                    end
                end
            end
            err_Laplace_svb=err_Laplace_svb+ a;
            
            % Student SVB
            [Result_St]=SVB(Phi,y,grouping,block_size,status,3);
            a = norm(Result_St.x - x_gen , 'fro') / norm( x_gen , 'fro');
            supp_1=setdiff(Result_St.z,supp_actual);
            supp_2=setdiff(supp_actual,Result_St.z);
            if isempty(supp_1)==1 && isempty(supp_2)==1
                succ_StSVB=succ_StSVB+1;
            else
                t=intersect(Result_St.z,supp_actual);
                if length(t)==block_size*K
                    if a < threshold
                        succ_StSVB=succ_StSVB+1;
                    end
                end
            end
            err_St_svb =err_St_svb+ a;
            %           succ=[succ; [succ_BSBL_EM succ_BSBL_BO succ_JTSVB succ_JSVB succ_LTSVB succ_LSVB succ_StTSVB succ_StSVB]];
        end
        %       succ=[succ; [succ_BSBL_EM succ_BSBL_BO succ_JTSVB succ_JSVB succ_LTSVB succ_LSVB succ_StTSVB succ_StSVB]]
        succ=[succ; [succ_JTSVB succ_JSVB succ_LTSVB succ_LSVB succ_StTSVB succ_StSVB]]
        
        %         err_BSBL_EM=err_BSBL_EM/k;
        %         err_BSBL_BO=err_BSBL_BO/k;
        err_Jeffrey_tsvb=err_Jeffrey_tsvb/k;
        err_Laplace_tsvb=err_Laplace_tsvb/k;
        err_St_tsvb=err_St_tsvb/k;
        err_Jeffrey_svb=err_Jeffrey_svb/k;
        err_Laplace_svb=err_Laplace_svb/k;
        err_St_svb=err_St_svb/k;
        %         err=[err; [err_BSBL_EM err_BSBL_BO err_Jeffrey_tsvb err_Jeffrey_svb err_Laplace_tsvb err_Laplace_svb err_St_tsvb err_St_svb]]
        err=[err; [err_Jeffrey_tsvb err_Jeffrey_svb err_Laplace_tsvb err_Laplace_svb err_St_tsvb err_St_svb]]
    end
    FigHandle = figure;
    set(FigHandle, 'Position', [100, 100, 520, 380]);
    % semilogy(0.1:0.05:0.5,err(:,1),'-.r*')
    % hold on
    % semilogy(0.1:0.05:0.5,err(:,2),'r--o')
    % hold on
    semilogy(0.1:0.05:0.5,err(:,1),'-.b*')
    hold on
    semilogy(0.1:0.05:0.5,err(:,2),'b--o')
    hold on
    semilogy(0.1:0.05:0.5,err(:,3),'-.g*')
    hold on
    semilogy(0.1:0.05:0.5,err(:,4),'g--o')
    hold on
    semilogy(0.1:0.05:0.5,err(:,5),'-.k*')
    hold on
    semilogy(0.1:0.05:0.5,err(:,6),'k--o')
    a_t=title(sprintf('rho=%g, Block Size=%g, SNR=%g dB',rho,block_size,SNR));
    set(a_t,'FontSize',12,'FontWeight','bold')
    a_l=legend({'cJSVB';'JSVB'; 'cLSVB';'LSVB';'cStSVB'; 'StSVB'});
    set(a_l,'FontSize',10,'FontWeight','bold');
    a_y=ylabel(' Relative Reconstruction Error');
    set(a_y,'FontSize',12,'FontWeight','bold');
    xlabel('M/N');
    set(gca,'XTick', [0.1:0.05:0.5], 'FontSize',12,'FontWeight','bold');
    saveas(FigHandle,strcat('Fig4_rho=',num2str(rho),'_BlockSize=',num2str(block_size),'RelativeReconstructionError','.png'));
    
    FigHandle=figure;
    plot(0.1:0.05:0.5,(1-(succ(:,1)/k)),'-.b*')
    hold on
    plot(0.1:0.05:0.5,(1-(succ(:,2)/k)),'b--o')
    hold on
    plot(0.1:0.05:0.5,(1-(succ(:,3)/k)),'-.g*')
    hold on
    plot(0.1:0.05:0.5,(1-(succ(:,4)/k)),'g--o')
    hold on
    plot(0.1:0.05:0.5,(1-(succ(:,5)/k)),'-.k*')
    hold on
    plot(0.1:0.05:0.5,(1-(succ(:,6)/k)),'k--o')
    a_t=title(sprintf('rho=%g, Block Size=%g, SNR=%g dB',rho,block_size,SNR));
    set(a_t,'FontSize',12,'FontWeight','bold')
    a_l=legend({'cJSVB';'JSVB'; 'cLSVB';'LSVB';'cStSVB'; 'StSVB'});
    set(a_l,'FontSize',10,'FontWeight','bold');
    a_y=ylabel('Failure rate');
    set(a_y,'FontSize',12,'FontWeight','bold');
    xlabel('M/N');
    set(gca,'XTick', [0.1:0.05:0.5], 'FontSize',12,'FontWeight','bold');
    
    saveas(FigHandle,strcat('Fig4_rho=',num2str(rho),'_BlockSize=',num2str(block_size),'FailureRate','.png'));
end