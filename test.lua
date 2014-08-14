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
rescale = 1   -- Recales dataset, for testing 

-- Download mnist if not present
mnist_folder = '../mnist-th7'
mnist.unpack(mnist_folder)
data_test = mnist.loadFlatDataset(paths.concat(mnist_folder,'test.th7') )
data_train = mnist.loadFlatDataset(paths.concat(mnist_folder,'train.th7') )

--sc = function(s) return math.floor(s*rescale) end


----GPU?
----require 'cutorch'
----print(  cutorch.getDeviceProperties(cutorch.getDevice()) )

--local x_train = data_train.x[{{1,sc(50000)},{}}]
--local y_train = data_train.y_vec[{{1,sc(50000)},{}}]
--local x_val = data_train.x[{{sc(50001),sc(60000)},{}}]
--local y_val= data_train.y_vec[{{sc(50001),sc(60000)},{}}]
--local x_test = data_test.x
--local y_test = data_test.y_vec
--local opts = {}


----------------------------
----       SETUP DATA      --
----------------------------

--opts.n_hidden     = 500
--opts.numepochs    = 200
--opts.patience     = 15                             -- early stopping is always enabled, to disble set this to inf = 1/0   
--opts.learningrate = 0.05
--opts.alpha = 0
--opts.beta = 0
--opts.isgpu = 0

--acc_train = accuracy(rbm,x_train,y_train)
--acc_val = accuracy(rbm,x_val,y_val)
--acc_test = accuracy(rbm,x_test,y_test)
--print('Train error      : ', 1-acc_train)
--print('Validation error : ', 1-acc_val)
--print('Test error       : ', 1-acc_test)


