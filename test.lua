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
torch.setnumthreads(1)

--------------------------
--       LOAD DATA      --
--------------------------
mnist_folder = '../mnist-th7'
rescale = 1
x_train, y_train, x_val, y_val, x_test, y_test = mnist.createdatasets(mnist_folder,rescale) 


_,y_trainidx = torch.max(y_train,2)

trainSet = {
    data = x_train,
    labels = y_trainidx,
    size = 60000
}

nInputs = 784
nOutputs = 10
nHidden = nInputs / 2
-- Container = Sequential
model = nn.Sequential()
model:add(nn.Reshape(nInputs))
model:add(nn.Linear(nInputs, nHidden))
model:add(nn.Tanh())
model:add(nn.Linear(nHidden, nOutputs))
model:forward(trainSet.data[200])


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



