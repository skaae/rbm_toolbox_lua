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
 * CD - k (contrastive divergence k) [5]
 * PCD (persistent contrastive divergence) [6]
 * RBM sampling functions (pictures / movies) (TODO)
 * RBM Classification support [2,7]
 * Regularization: L1, L2, sparsity, early-stopping, dropout [1],dropconnect[10], momentum [3] 

# Installation

 1. Install torch7: Follow [these](https://github.com/torch/torch7/wiki/Cheatsheet#installing-and-running-torch) instructions
 2. download this repository: `git clone https://github.com/skaae/rbm_toolbox_lua.git`
 3. Execute your scripts from the repository folder

# Settings

## Setting up the RBM and training
A RBM with standard settings can be trained and with:

```LUA
opts = {}
rbm = rbmsetup(opts,x_train, y_train)
rbm = rbmtrain(rbm,x_train,y_train,x_val,y_val)
saverbm('rbm.asc',rbm)
```

You may resume training of an already trained RBM. E.g if you trained a RBM for 10 epochs but want to train it for 
10 more epochs, then:

```LUA
rbm = loadrbm('rbm.asc')
rbm.numepochs = 20
rbm = rbmtrain(rbm,x_train,y_train,x_val,y_val) -- resumes training from current epoch
```

Settings are controlled with the opts table. 
Passing in an empty opts table to rbmsetup will use the default value. The following code shows valid opts fields 
and their default value:

```LUA
 -- Max epochs, lr and Momentum
rbm.numepochs       = opts.numepochs or 5
rbm.learningrate    = opts.learningrate or 0.05
rbm.momentum        = opts.momentum or 0
rbm.traintype       = opts.traintype or 'CD'   -- CD or PCD
rbm.cdn             = opts.cdn or 1
rbm.npcdchains      = opts.npcdchains or 100

-- OBJECTIVE
rbm.alpha           = opts.alpha or 1
rbm.beta            = opts.beta or 0

-- REGULARIZATION
rbm.dropout         = opts.dropout or 0
rbm.dropconnect     = opts.dropconnect or 0
rbm.L1              = opts.L1 or 0
rbm.L2              = opts.L2 or 0
rbm.sparsity        = opts.sparsity or 0
rbm.patience        = opts.patience or 15

-- -
rbm.tempfile        = opts.tempfile or "temp_rbm.asc"
rbm.isgpu           = opts.isgpu or 0
```

## Passing in data
Training, valdiation and semisupervised data are arguments to `rbmtrain`:

```LUA
rbmtrain(rbm,x_train,y_train,x_val,y_val,x_semisup)
```

validation data and semisupervised data may be ommitted.

## Training objective

The training behavior is controlled with `opts.alpha`

 * opts.alpha =     0 : Discriminative training
 * opts.alpha =     1 : Generative training
 * 0 < opts.alpha < 1 : Hybrid training

 Amount of semisupervised training is controlled with `opts.beta`

## Regularization

 * `opts.L1`: specify the regularization weight
 * `opts.L2`: specify the regularization weight
 * `opts.sparsity`: implemented as in [7]. Specify the sparsity being subtracted from biases after each weight update.
 * `opts.dropout`: dropout on hidden units. Specify the 1-probability of being dropped. see [1]
 * `opts.dropconnect`: dropout on connections, specify 1-probability of connection being zeroed, see [10]
 * Early-stopping: Always enabled. Set the patience with opts.patience. To disable
 early stopping set patience to infinity.

### DropOut
Dropout is implemented by clamping random hidden units to zero during training. The probability of each hidden units 
being clamped is `1-opts.dropout`, i.e `opts.dropout = 0` will clamp all hidden units to zero and `opts.dropout =1` will clamp 
no units to zero. `opts.dropout = 0` will disable dropout (It does not make sense to clamp all units anyway).

### DropConnect
Drop connect clamps Weights and Biases of the hidden layer to zero with probability `1-opts.dropconnect`.
DropConnect is applied by randomly clamping weights of *W*, *U* and *c* to zero. DropConnect is slower than 
dropout because we need to clone the weight matrices before each weight update. 

# Examples

Reproducing results from [7], specifically the results from the table reproduced below:

| Model  |Objective                                         | Errror (%)    | Example  |
|---     |---                                               |---            |---       |
|ClassRBM| Discriminative(lr = 0.05, H = 500)               |   1.81        |    1     |
|        | Generative(lr = 0.005, H = 6000)                 |   3.39        |    2     |
|        | Hybrid(alpha = 0.01, lr = 0.05, H = 1500)        |   1.28        |    3     |
|        | Sparse Hybrid( idem + H = 3000, sparsity=10^-4)  |   1.16        |    4     |
lr = learning rate
H = hidden layer size

## Example 1 - Discriminative Training

Trains a discriminative RBM with 500 hidden units. The toolbox supports discriminative, generative,
hybrid training as described in [7].

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

The results are:
 * Train error      : 	2.000000000002e-05	
 * Validation error : 	0.0183	
 * Test error       : 	0.0188	
Which is comparable to the results reported in [7].


The results can be improved significantly by enabling dropout:

```LUA
opts.dropout = 0.5
```
The effective number of units, opts.dropout * #hidden, vas kept at 500, i.e the number of hidden units was increased 
to 1000. All other settings where kept at their previous values.

Using dropout the results are:
Train error      : 	2.000000000002e-05	
Validation error : 	0.0135	
Test error       : 	0.0141

These results are significantly better than non-dropout and similar to the performance of an SVM.

The figures below show differeces between training without dropout (left) and with dropout (right).
First the validation error and training error is plotted. Training with dropout requires many more epochs before convergence.
The second figure vizualizes the learned filters. 

<img src="/uploads/ex1_trainval.png" height="400" width="600">    


<img src="/uploads/ex1_weights.png" height="700" width="550">   


DO DROPOUT DISCRIMINATIVE WITH SPARSITY??



The graphs are created in MATLAB. I created a simple script to pass RBM's from Torch to MATLAB

In Torch do:

```LUA
-- Load RBM or use one you have trained
rbm = loadrbm('discriminative_final.asc')

-- Save the RBM to CSV files.
writerbmtocsv(rbm)  -- optinally specify save folder as 2. arg
```

Launch MATLAB browse to the folder where the CSV's are saved, then 

```MATLAB
figure;
rbm = loadrbm();  % optonally spcify another path to CSV's
plotx = 1:numel(rbm.err_val);
plot(plotx,rbm.err_val,plotx,rbm.err_train);

prettyfig(gca,title('Discriminative training'),...
          xlabel('Epochs'),ylabel('Error(%)'),legend({'Validation','Training'}))
grid on

figure;
[~,idx] = sort( sum(rbm.W.^2,2), 1, 'descend');  % sort by norm
idx = idx(1:100);
visualize(rbm.W(idx,:)')
axis off;
```

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
