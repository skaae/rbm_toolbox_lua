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

