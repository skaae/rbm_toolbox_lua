--require('nn')
require('torch')
-- load rbm functions
require('rbm')
require('dataset-mnist')
require('ProFi')
require 'paths'

torch.manualSeed(101)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)

--------------------------
--       LOAD DATA      --
--------------------------
rescale = 1   -- Recales dataset, for testing 

-- Download mnist if not present
mnist_folder = '../MNISTDATA'
mnist.download( mnist_folder )
data_test = mnist.loadFlatDataset(paths.concat(mnist_folder,'test.th7') )
data_train = mnist.loadFlatDataset(paths.concat(mnist_folder,'train.th7') )

sc = function(s) return math.floor(s*rescale) end


-- torch.setdefaulttensortype('torch.FloatTensor')-- GPU
local x_train = data_train.x[{{1,sc(50000)},{}}]
local y_train = data_train.y_vec[{{1,sc(50000)},{}}]
local x_val = data_train.x[{{sc(50001),sc(60000)},{}}]
local y_val= data_train.y_vec[{{sc(50001),sc(60000)},{}}]
local x_test = data_test.x
local y_test = data_test.y_vec
local opts = {}


--------------------------
--       SETUP DATA      --
--------------------------
tempfile = 'discriminative_temp.asc'
tempfolder = '../rbmtemp'
finalfile = 'discriminative_final.asc'
os.execute('mkdir -p ' .. tempfolder)   -- Create save folder if it does not exists
finalfile = 

opts.tempfile = paths.concat(tempfolder,tempfile)  -- current best is saved to this folder
opts.n_hidden     = 500
opts.numepochs    = 200
opts.patience     = 15                             -- early stopping is always enabled, to disble set this to inf = 1/0   
opts.learningrate = 0.05
opts.alpha = 0
opts.beta = 0

-- DO Training
local rbm = rbmsetup(opts,x_train, y_train)
rbm = rbmtrain(rbm,x_train,y_train,x_val,y_val)

saverbm(paths.concat(tempfolder,tempfile),rbm)
