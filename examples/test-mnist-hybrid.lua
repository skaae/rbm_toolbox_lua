codeFolder = '../code/'

require('torch')
require(codeFolder..'rbm')
require(codeFolder..'dataset-mnist')
require(codeFolder..'ProFi')
require 'paths'

torch.manualSeed(101)
torch.setdefaulttensortype('torch.FloatTensor')

-- LOAD DATA
mnist_folder = '../../mnist-th7'
rescale = 1
train, val, test = mnist.createdatasets(mnist_folder,rescale) 
   
num_threads = 1
torch.setnumthreads(num_threads)
if torch.getnumthreads() < num_threads then
     print("Setting number of threads had no effect. Maybe install with gcc 4.9 for openMP?")
end

-- SETUP RBM
local opts = {}
local tempfile = 'hybrid_temp.asc'
local tempfolder = '../rbmtemp'
os.execute('mkdir -p ' .. tempfolder)              -- create tempfolder if it does not exist
local finalfile = 'hybrid_final.asc'             -- Name of final RBM file
os.execute('mkdir -p ' .. tempfolder)              -- Create save folder if it does not exists
opts.tempfile = paths.concat(tempfolder,tempfile)  -- current best is saved to this folder
opts.traintype = 'CD'
opts.cdn = 1
opts.n_hidden     = 1500
opts.dropout      = 0
opts.numepochs    = 1500
opts.patience     = 15                             -- early stopping is always enabled, to disble set this to inf = 1/0   
opts.learningrate = 0.05
opts.alpha = 0.01
opts.beta = 0
opts.isgpu = 0

-- DO TRAINING
rbm = trainAndPrint(opts,train,val,test,tempfolder,finalfile)

