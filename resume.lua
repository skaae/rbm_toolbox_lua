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
   
   
local tempfile = 'discriminative_dropout_temp.asc'
local tempfolder = '../rbmtemp'            -- create tempfolder if it does not exist
local finalfile = 'discriminative_dropout_final.asc'             -- Name of final RBM file   
   

-- DO TRAINING
local rbm = loadrbm('../rbmtemp/discriminative_dropout_temp.asc')
rbm.tempfile = paths.concat(tempfolder,tempfile)
rbm.numepochs = 200
--rbm = rbmtrain(rbm,x_train,y_train,x_val,y_val)
--saverbm(paths.concat(tempfolder,finalfile), rbm)
local acc_train = accuracy(rbm,x_train,y_train)
local acc_val = accuracy(rbm,x_val,y_val)
local acc_test = accuracy(rbm,x_test,y_test)
print('Train error      : ', 1-acc_train)
print('Validation error : ', 1-acc_val)
print('Test error       : ', 1-acc_test)

