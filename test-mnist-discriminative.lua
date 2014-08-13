--require('nn')
require('torch')
-- load rbm functions
require('rbm')
require('dataset-mnist')
require('ProFi')

torch.manualSeed(101)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)

mnist.download()
data_test = mnist.loadFlatDataset('mnist-th7/test.th7')
data_train = mnist.loadFlatDataset('mnist-th7/train.th7')
rescale = 1
sc = function(s) return math.floor(s*rescale) end

local x_train = data_train.x[{{1,sc(50000)},{}}]
local y_train = data_train.y_vec[{{1,sc(50000)},{}}]
local x_val = data_train.x[{{sc(50001),sc(60000)},{}}]
local y_val= data_train.y_vec[{{sc(50001),sc(60000)},{}}]
local x_test = data_test.x
local y_test = data_test.y_vec
local opts = {}
opts.n_hidden     = 500
opts.numepochs    = 200
opts.learningrate = 0.05
opts.alpha = 0
opts.beta = 0
local rbm = rbmsetup(opts,x_train, y_train)
rbm = rbmtrain(rbm,x_train,y_train,x_val,y_val)


