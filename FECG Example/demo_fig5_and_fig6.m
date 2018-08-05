% This code aims to reproduce the results corresponding to Fig.5 and Fig.6 
% of the below mentioned paper which proposes the correlated sparse
% variational bayes framework for block sparse signal recovery. 
%
% Author: Shruti Sharma
% Paper: Shruti Sharma, Santanu Chaudhury, Jayadeva, ' Variational
% Bayes Block Sparse Modeling with Correlated Entries', ICPR-2018.
%
% To reproduce Fig.5 and Fig.6, we set status=0 which corresponds to signal recovery
% in noiseless setting. That is, we aim to reconstruct the original signal
% with less number of measurements and try to replicate the experiment setting
% as proposed in 
% Reference: Zhang Z., et al., ' CS of Energy Efficient Wireless Telemonitoring of Noninvasive
% Fetal ECG Via Block Sparse Learning', IEEE Trans. on Biomed. Engg., Vol. 60, No. 2,
% pp. 300-309, 2013. 
% Relevant MATLAB codes for BSBL-BO and dataset can also be downloaded from
% https://sites.google.com/site/researchbyzhang/publications.

% Also in the paper, we have shown only results corresponding to cLSVB and
% LSVB for the sake of brevity. From this code, we can get results
% corresponding to all the variants of cSVB and SVB.

% Comparisons in the paper are made w.r.t. BSBL-BO, 
% Reference: Zhang Z., Rao B.D., 'Extension of SBL Algorithms for the Recovery of Block Sparse
% Signals With Intra-Block Correlation', IEEE Trans. on Sig. Proc., vol. 1, no. 8, pp. 2009-2015. 2013)
% and the variants of sparse variational bayes SVB 
% Reference: Babacan S.D., et al.,'Bayesian Group-Sparse Modeling and Variational 
% Inference',IEEE Trans. on  Sig. Proc., vol. 62, no. 11, pp. 2906-2921, 2014).
%


clear;  %close all;
load signal_01.mat;
s = s(:,1:4:51200); 
 
% the size of the sensing matrix Phi
M=200;
N=512;

% Random Matrix Phi generation
while(1)
    [Phi,flag]=genPhi(M,N, 10);
    if flag==1
       break; 
    end
end

% variables for recovered dataset
X_hat_cJSVB=zeros(size(s));
X_hat_cLSVB=zeros(size(s));
X_hat_cStSVB=zeros(size(s));
X_hat_JSVB=zeros(size(s));
X_hat_LSVB=zeros(size(s));
X_hat_StSVB=zeros(size(s));

k=0;

% Noiseless Signal Recovery channel-wise

for i=1:8
    fprintf('\nChannel:%d\n',i);
    for j = 1 : size(s,2)/N
        k = k + 1;
        
        fprintf('  Segment %d: ',j);
         
        % Observed and compressed Data
        y=Phi*s(i,(j-1)*N+1:j*N)';   
        Y(i,(j-1)*M+1:j*M)=y';         
        
        block_size=16;
        block_ind=reshape(ones(block_size,1)*(1:N),N*block_size,1);
        for g=1:32
            grouping{g}=find(ismember(block_ind,g));
        end
        
        status=0;
        [Result_cJ]=CSVB(Phi,y,grouping,block_size,status,1);
        [Result_J]=SVB(Phi,y,grouping,block_size,status,1);
        [Result_cL]=CSVB(Phi,y,grouping,block_size,status,2);
        [Result_L]=SVB(Phi,y,grouping,block_size,status,2);
        [Result_cSt]=CSVB(Phi,y,grouping,block_size,status,3);
        [Result_St]=SVB(Phi,y,grouping,block_size,status,3);
         
        X_hat_cJSVB(i,(j-1)*N+1:j*N) = (Result_cJ.x)';
        X_hat_JSVB(i,(j-1)*N+1:j*N) = (Result_J.x)';
        X_hat_cLSVB(i,(j-1)*N+1:j*N) = (Result_cL.x)';
        X_hat_LSVB(i,(j-1)*N+1:j*N) = (Result_L.x)';
        X_hat_cStSVB(i,(j-1)*N+1:j*N) = (Result_cSt.x)';
        X_hat_StSVB(i,(j-1)*N+1:j*N) = (Result_St.x)';
        
        mse_cJSVB(k)=(norm(s(i,(j-1)*N+1:j*N)-X_hat_cJSVB(i,(j-1)*N+1:j*N),'fro')/norm(s(i,(j-1)*N+1:j*N),'fro'))^2;
        mse_JSVB(k)=(norm(s(i,(j-1)*N+1:j*N)-X_hat_JSVB(i,(j-1)*N+1:j*N),'fro')/norm(s(i,(j-1)*N+1:j*N),'fro'))^2;
        fprintf(' MSE of cJSVB= %g, count = %d\n',mse_cJSVB(k),Result_cJ.count);
        fprintf(' MSE of JSVB= %g, count = %d\n',mse_JSVB(k),Result_J.count);
        mse_cLSVB(k)=(norm(s(i,(j-1)*N+1:j*N) - X_hat_cLSVB(i,(j-1)*N+1:j*N),'fro')/norm(s(i,(j-1)*N+1:j*N),'fro'))^2;
        mse_LSVB(k)=(norm(s(i,(j-1)*N+1:j*N) - X_hat_LSVB(i,(j-1)*N+1:j*N),'fro')/norm(s(i,(j-1)*N+1:j*N),'fro'))^2;
        fprintf(' MSE of cLSVB= %g, count = %d\n',mse_cLSVB(k),Result_cL.count);
        fprintf(' MSE of LSVB= %g, count = %d\n',mse_LSVB(k),Result_L.count);
        mse_cStSVB(k)=(norm(s(i,(j-1)*N+1:j*N) - X_hat_cStSVB(i,(j-1)*N+1:j*N),'fro')/norm(s(i,(j-1)*N+1:j*N),'fro'))^2;
        mse_StSVB(k)=(norm(s(i,(j-1)*N+1:j*N) - X_hat_StSVB(i,(j-1)*N+1:j*N),'fro')/norm(s(i,(j-1)*N+1:j*N),'fro'))^2;
        fprintf(' MSE of cStSVB= %g, count = %d\n',mse_cStSVB(k),Result_cSt.count);
        fprintf(' MSE of StSVB= %g, count = %d\n',mse_StSVB(k),Result_St.count);

    end
    
end
fprintf('Total MSE: %g\n',mean(mse_cJSVB)); 
fprintf('Total MSE: %g\n',mean(mse_JSVB)); 
fprintf('Total MSE: %g\n',mean(mse_cLSVB)); 
fprintf('Total MSE: %g\n',mean(mse_LSVB)); 
fprintf('Total MSE: %g\n',mean(mse_cStSVB)); 
fprintf('Total MSE: %g\n',mean(mse_StSVB)); 

% display the original dataset (Fig.5 of the paper)
set(0, 'DefaultFigurePosition', [100 70 500 600]);
ICAshow(s,'title','Original Recordings');

% display the reconstructed dataset (Fig.5 of the paper)
set(0, 'DefaultFigurePosition', [650 70 500 600]);
ICAshow(X_hat_cJSVB,'title','Reconstructed Recordings by cJSVB');

set(0, 'DefaultFigurePosition', [650 70 500 600]);
ICAshow(X_hat_JSVB,'title','Reconstructed Recordings by JSVB');

set(0, 'DefaultFigurePosition', [650 70 500 600]);
ICAshow(X_hat_cLSVB,'title','Reconstructed Recordings by cLSVB');

set(0, 'DefaultFigurePosition', [650 70 500 600]);
ICAshow(X_hat_LSVB,'title','Reconstructed Recordings by LSVB');

set(0, 'DefaultFigurePosition', [650 70 500 600]);
ICAshow(X_hat_cStSVB,'title','Reconstructed Recordings by cStSVB');

set(0, 'DefaultFigurePosition', [650 70 500 600]);
ICAshow(X_hat_StSVB,'title','Reconstructed Recordings by StSVB');

%====================================================
%   ICA decomposition of the original recordings (Fig.6 of the paper)
%====================================================

% band-pass filtering 
lo=0.008;  % normalized low frequency cut-off corresponding to 1.75 Hz
hi=0.4;    % normalized high frequency cut-off corresponding to 1000 Hz
s_flt=BPFilter(s,lo,hi);

% remove mean and normalized to unit variance
Z=standarize(s_flt);

% whitening
[WhitenedSig,WhitenMatrix]=whiten(Z); 

% perform ICA decomposition 
s1 = FastICA(WhitenedSig,'method','defl','ICNum',5);
set(0, 'DefaultFigurePosition', [50 20 500 650]);

% display the ICA decomposition
seg = 1:1000;
ICAshow(s1(:,seg),'title','ICA of Original Recordings');


%====================================================
%   ICA decomposition of the recovered recordings (Fig. 6 of the paper)
%====================================================

% perform the same band-pass filtering
X_hat_flt_cJSVB = BPFilter(X_hat_cJSVB,lo,hi);
X_hat_flt_JSVB = BPFilter(X_hat_JSVB,lo,hi);
X_hat_flt_cLSVB = BPFilter(X_hat_cLSVB,lo,hi);
X_hat_flt_LSVB = BPFilter(X_hat_LSVB,lo,hi);
X_hat_flt_cStSVB = BPFilter(X_hat_cStSVB,lo,hi);
X_hat_flt_StSVB = BPFilter(X_hat_StSVB,lo,hi);

% perform the same mean-removing and normalized to unit variance
Z_hat_cJSVB = standarize(X_hat_flt_cJSVB);
Z_hat_JSVB = standarize(X_hat_flt_JSVB);
Z_hat_cLSVB = standarize(X_hat_flt_cLSVB);
Z_hat_LSVB = standarize(X_hat_flt_LSVB);
Z_hat_cStSVB = standarize(X_hat_flt_cStSVB);
Z_hat_StSVB = standarize(X_hat_flt_StSVB);

% perform the same whitening
[WhitenedSig2_cJSVB,WhitenMatrix2_cJSVB]=whiten(Z_hat_cJSVB); 
[WhitenedSig2_JSVB,WhitenMatrix2_JSVB]=whiten(Z_hat_JSVB); 
[WhitenedSig2_cLSVB,WhitenMatrix2_cLSVB]=whiten(Z_hat_cLSVB); 
[WhitenedSig2_LSVB,WhitenMatrix2_LSVB]=whiten(Z_hat_LSVB); 
[WhitenedSig2_cStSVB,WhitenMatrix2_cStSVB]=whiten(Z_hat_cStSVB); 
[WhitenedSig2_StSVB,WhitenMatrix2_StSVB]=whiten(Z_hat_StSVB); 

% perform the same ICA algorithm
[s2_cJSVB] = FastICA(WhitenedSig2_cJSVB,'method','defl','ICNum',5);
[s2_JSVB] = FastICA(WhitenedSig2_JSVB,'method','defl','ICNum',5);
[s2_cLSVB] = FastICA(WhitenedSig2_cLSVB,'method','defl','ICNum',5);
[s2_LSVB] = FastICA(WhitenedSig2_LSVB,'method','defl','ICNum',5);
[s2_cStSVB] = FastICA(WhitenedSig2_cStSVB,'method','defl','ICNum',5);
[s2_StSVB] = FastICA(WhitenedSig2_StSVB,'method','defl','ICNum',5);

% sorting of the independent components
s2_sort_cJSVB = zeros(size(s2_cJSVB));
for i = 1: size(s2_cJSVB,1)
    co = s1(i,:)*s2_cJSVB'/length(s1(i,:));
    [~,ind]=max(abs(co));
    s2_sort_cJSVB(i,:) = s2_cJSVB(ind,:)*sign(co(ind));
end
s2_sort_JSVB = zeros(size(s2_JSVB));
for i = 1: size(s2_JSVB,1)
    co = s1(i,:)*s2_JSVB'/length(s1(i,:));
    [~,ind]=max(abs(co));
    s2_sort_JSVB(i,:) = s2_JSVB(ind,:)*sign(co(ind));
end
s2_sort_cLSVB = zeros(size(s2_cLSVB));
for i = 1: size(s2_cLSVB,1)
    co = s1(i,:)*s2_cLSVB'/length(s1(i,:));
    [~,ind]=max(abs(co));
    s2_sort_cLSVB(i,:) = s2_cLSVB(ind,:)*sign(co(ind));
end
s2_sort_LSVB = zeros(size(s2_LSVB));
for i = 1: size(s2_LSVB,1)
    co = s1(i,:)*s2_LSVB'/length(s1(i,:));
    [~,ind]=max(abs(co));
    s2_sort_LSVB(i,:) = s2_LSVB(ind,:)*sign(co(ind));
end
s2_sort_cStSVB = zeros(size(s2_cStSVB));
for i = 1: size(s2_cStSVB,1)
    co = s1(i,:)*s2_cStSVB'/length(s1(i,:));
    [~,ind]=max(abs(co));
    s2_sort_cStSVB(i,:) = s2_cStSVB(ind,:)*sign(co(ind));
end
s2_sort_StSVB = zeros(size(s2_StSVB));
for i = 1: size(s2_StSVB,1)
    co = s1(i,:)*s2_StSVB'/length(s1(i,:));
    [~,ind]=max(abs(co));
    s2_sort_StSVB(i,:) = s2_StSVB(ind,:)*sign(co(ind));
end

% Display the ICA decomposition from the recovered dataset.
% *** The fetal ECG could be the 3rd component, or the 4th component **** $
ICAshow(s2_sort_cJSVB(:,seg),'title','ICA of Recovered Recordings cJSVB');
ICAshow(s2_sort_JSVB(:,seg),'title','ICA of Recovered Recordings JSVB');
ICAshow(s2_sort_cLSVB(:,seg),'title','ICA of Recovered Recordings cLSVB');
ICAshow(s2_sort_LSVB(:,seg),'title','ICA of Recovered Recordings LSVB');
ICAshow(s2_sort_cStSVB(:,seg),'title','ICA of Recovered Recordings cStSVB');
ICAshow(s2_sort_StSVB(:,seg),'title','ICA of Recovered Recordings StSVB');