RBM Toolbox for Torch
===============

RBM toolbox is a Torch7 toolbox for online training of RBM's. A MATLAB version exists at 
[LINK](https://github.com/skaae/rbm_toolbox).

The following is supported:
 * Support for training RBM's with class labels including:
    * Generative training objective [2,7]
    * Discriminative training objective [2,7]
    * Hybrid training objective [2,7]
    * Semi-supervised learning [2,7]   untested
 * CD - k (contrastive divergence k) [5]
 * PCD (persistent contrastive divergence) [6]
 * RBM Classification support [2,7]
 * Regularization: L1, L2, sparsity, early-stopping, dropout [1], momentum [3] 

# Installation

 1. Install torch7: Follow [these](https://github.com/torch/torch7/wiki/Cheatsheet#installing-and-running-torch) instructions
 2. download this repository: `git clone https://github.com/skaae/rbm_toolbox_lua.git`
 4. To run the examples install wget with homebrew
 3. Run example rbms with examples/runrbm.lua 

# Examples
Run from /example folder


  1) th runrbm.lua -eta 0.05 -alpha 0 -nhidden 500 -folder test_discriminative
  
  2) th runrbm.lua -eta 0.05 -alpha 0 -nhidden 500 -folder test_discriminative_dropout -dropout 0.5
  
   3) th runrb,.lua -eta 0.05 -alpha 1 -nhidden 500 -folder test_generative_pcd -traintype PCD
   4) th runrbm.lua -eta 0.05 -alpha 0.01 -nhidden 1500 -folder test_hybrid

   5) th runrbm.lua -eta 0.05 -alpha 0.01 -nhidden 1500 -folder test_hybrid_dropout -dropout 0.5

   6) th runrbm.lua -eta 0.05 -alpha 0.01 -nhidden 3000 -folder test_hybrid_sparsity -sparsity 0.0001

   7) th runrbm.lua -eta 0.05 -alpha 0.01 -nhidden 3000 -folder test_hybrid_sparsity_dropout -sparsity 0.0001 -dropout 0.5

   8) th runrbm.lua -eta 0.05 -alpha 1 -nhidden 1000 -folder test_generative

   9) th runrbm.lua -eta 0.05 -alpha 1 -nhidden 2000 -folder test_generative -dropout -0.5

# Using your own data
You can create our own datasets with the functions in
code/dataset-from-tensor.lua

```LUA
codeFolder = '../code/'
require('torch')
require(codeFolder..'rbm')
require(codeFolder..'dataset-from-tensor')
require 'paths'
geometry = {1,100}   -- dimensions of your training data
nclasses = 3
nSamples = 5 
trainTensor = torch.rand(nSamples,geometry[1],geometry[2])
trainLabels = torch.Tensor({1,2,3,1,2})
classes = {'ClassA','ClassB','ClassC'}
trainData = datatensor.createDataset(trainTensor,
                                     oneOfK(nclasses,trainLabels),
                                     classes,
                                     geometry)
print(trainData:next())
print(trainData[2])
print(trainData:classnames())

```
# TODO

 1. DO DROPOUT DISCRIMINATIVE WITH SPARSITY?   
 2. Use momentum to smooth gradients? + Decrease learning rate    
 3. Generative training example + samples drawn from model   
 4. Hybrid training exampe
 5. Semisupervised example
 6. Implement stacking of RBM's

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
