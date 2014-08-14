require('rbm-util')
rbm = loadrbm('../rbmtemp/discriminative_final.asc')


--require('nn')
require('torch')
-- load rbm functions
require('rbm')
require('dataset-mnist')
require 'paths'

torch.manualSeed(101)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)

--------------------------
--       LOAD DATA      --
--------------------------
mnist_folder = '../mnist-th7'
rescale = 1
x_train, y_train, x_val, y_val, x_test, y_test = mnist.createdatasets(mnist_folder,rescale) 


----------------------------
----       SETUP RBM      --
----------------------------

--opts.n_hidden     = 500
--opts.numepochs    = 200
--opts.patience     = 15                             -- early stopping is always enabled, to disble set this to inf = 1/0   
--opts.learningrate = 0.05
--opts.alpha = 0
--opts.beta = 0
--opts.isgpu = 0

acc_train = accuracy(rbm,x_train,y_train)
acc_val = accuracy(rbm,x_val,y_val)
acc_test = accuracy(rbm,x_test,y_test)
print('Train error      : ', 1-acc_train)
print('Validation error : ', 1-acc_val)
print('Test error       : ', 1-acc_test)


