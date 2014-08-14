RBM Toolbox for Torch
===============

RBM toolbox is a Torch7 toolbox for online training of RBM's. A MATLAB version exists at 
[LINK](https://github.com/skaae/rbm_toolbox).

The following is supported:
 * Support for training RBM's with class labels including:
    * Generative training objective [2,7]
    * Discriminative training objective [2,7]
    * Hybrid training objective [2,7]
    * Semi-supervised learning [2,7]
 * CD - k (contrastive divergence k) [5]    (cd1, cdk=TODO)
 * PCD (persistent contrastive divergence) [6] (TODO)
 * RBM sampling functions (pictures / movies) (TODO)
 * RBM Classification support [2,7]
 * Regularization: L1, L2, sparsity, early-stopping, dropout [1],dropconnect[10], momentum [3] 

# Installation

Requires Torch7 to be intalled. Follow [these](https://github.com/torch/torch7/wiki/Cheatsheet#installing-and-running-torch) instructions

# Examples

Reproducing results from [7], specifically the results from the table reproduced below:

| Model  |Objective                                         | Errror (%)    | Example  |
|---     |---                                               |---            |---       |
|        | Generative(lr = 0.005, H = 6000)                 |   3.39        |    1     |
|ClassRBM| Discriminative(lr = 0.05, H = 500)               |   1.81        |    2     |
|        | Hybrid(alpha = 0.01, lr = 0.05, H = 1500)        |   1.28        |    3     |
|        | Sparse Hybrid( idem + H = 3000, sparsity=10^-4)  |   1.16        |    4     |
lr = learning rate
H = hidden layer size

August 2014: Documentation and Examples will be added in the coming weeks.


Example 1 - Discriminative Training

Trains a discriminative RBM 

```LUA
require('torch')
require('rbm')
require('dataset-mnist')
require('ProFi')
require 'paths'

torch.manualSeed(101)
torch.setdefaulttensortype('torch.FloatTensor')

-- LOAD DATA
mnist_folder = '../mnist-th7'
rescale = 1
x_train, y_train, x_val, y_val, x_test, y_test = mnist.createdatasets(mnist_folder,rescale) 
   

-- SETUP RBM
local opts = {}
local tempfile = 'discriminative_temp.asc'
local tempfolder = '../rbmtemp'
os.execute('mkdir -p ' .. tempfolder)              -- create tempfolder if it does not exist
local finalfile = 'discriminative_final.asc'             -- Name of final RBM file
os.execute('mkdir -p ' .. tempfolder)              -- Create save folder if it does not exists
opts.tempfile = paths.concat(tempfolder,tempfile)  -- current best is saved to this folder
opts.n_hidden     = 500
opts.numepochs    = 200
opts.patience     = 15                             -- early stopping is always enabled, to disble set this to inf = 1/0   
opts.learningrate = 0.05
opts.alpha = 0
opts.beta = 0
opts.isgpu = 0

-- DO TRAINING
local rbm = rbmsetup(opts,x_train, y_train)
rbm = rbmtrain(rbm,x_train,y_train,x_val,y_val)
saverbm(paths.concat(tempfolder,tempfile),rbm)
local acc_train = accuracy(rbm,x_train,y_train)
local acc_val = accuracy(rbm,x_val,y_val)
local acc_test = accuracy(rbm,x_test,y_test)
print('Train error      : ', 1-acc_train)
print('Validation error : ', 1-acc_val)
print('Test error       : ', 1-acc_test)

```

Train error      : 	2.000000000002e-05	
Validation error : 	0.0183	
Test error       : 	0.0188	

<img src="/uploads/discriminative_ex1.png" height="400" width="400"> 

# References

[1] Srivastava Nitish, G. Hinton, A. Krizhevsky, I. Sutskever, and R. R. Salakhutdinov, “Dropout: A Simple Way to Prevent Neural Networks from Overfitting,” J. Mach. Learn. Res., vol. 5(Jun), no. 2, p. 1929−1958, 2014.    
[2] H. Larochelle and Y. Bengio, “Classification using discriminative restricted Boltzmann machines,” in Proceedings of the 25th international conference on Machine learning. ACM,, 2008.     
[3] G. Hinton, “A practical guide to training restricted Boltzmann machines,” Momentum, vol. 9, no. 1, p. 926, 2010.    
[4] G. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and R. R. Salakhutdinov, “Improving neural networks by preventing co-adaptation of feature detectors,” arXiv Prepr., vol. 1207.0580, no. Hinton, Geoffrey E., et al. "Improving neural networks by preventing co-adaptation of feature detectors." arXiv preprint arXiv:1207.0580 (2012)., Jul. 2012.    
[5] G. Hinton, “Training products of experts by minimizing contrastive divergence,” Neural Comput., vol. 14, no. 8, pp. 1771–1800, 2002.     
[6] T. Tieleman, “Training restricted Boltzmann machines using approximations to the likelihood gradient,” in Proceedings of the 25th international conference on Machine learning. ACM, 2008.    
[7] H. Larochelle, M. Mandel, R. Pascanu, and Y. Bengio, “Learning algorithms for the classification restricted boltzmann machine,” J. Mach. Learn. Res., vol. 13, no. 1, pp. 643–669, 2012.    
[8] R. Salakhutdinov and I. Murray, “On the quantitative analysis of deep belief networks,” in Proceedings of the 25th international conference on Machine learning. ACM,, 2008.    
[9] Y. Tang and I. Sutskever, “Data normalization in the learning of restricted Boltzmann machines,” Dep. Comput. Sci. Toronto Univ., vol. UTML-TR-11, 2011.     
[10] L. Wan, M. Zeiler, S. Zhang, Y. Le Cun, and R. Fergus, “Regularization of Neural Networks using DropConnect,” in Proceedings of The 30th International Conference on Machine Learning, 2013, pp. 1058–1066. 

Copyright (c) 2014, Søren Kaae Sønderby (skaaesonderby@gmail.com) All rights reserved.
