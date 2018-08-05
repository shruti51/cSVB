Demo Codes for Variational Bayes Block Sparse Modeling with Correlated Entries
Author: Shruti Sharma (shruti_sml@yahoo.com)
Date: August 05, 2018

This directory contains all the codes for the paper:
Shruti Sharma, Santanu Chaudhury, Jayadeva, "Variational Bayes Block Sparse Modeling with Correlated Entries", ICPR-2018.

Codes were tested under MATLAB R2017(a) on Windows 10 platform. 

In the directory, the demo codes titled as demo_figX.m generate the X figure number in the paper. Demo codes titled as demo_figY_and_figZ.m generate the X and Y figure number in the paper as per the instructions given in the comment section of the corresponding code. 

CSVB.m is implementation of the proposed framework. The command CSVB is fed with measurement matrix $\Phi$, observation vector $y$, block size of each block, setting in which we want to work i.e. noiseless or noisy, and induced marginal information i.e. Laplace, Student's t or Jeffrey. It gives the result in the form of structure consisting of solution vector $x$, number of iterations to converge the algorithm and $\alpha_i$ parameter values corresponding to non-zero blocks.

[Result]=CSVB(Phi,y,grouping,block_size,status,marg)

SVB command can be used in the similar fashion. For more details on SVB variants, kindly refer the following paper: 
[1] Babacan S.D., et al.,'Bayesian Group-Sparse Modeling and Variational Inference',IEEE Trans. on  Sig. Proc., vol. 62, no. 11, pp. 2906-2921, 2014. 
