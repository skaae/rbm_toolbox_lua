codeFolder = '../code/'

require('torch')
require(codeFolder..'rbm')
require(codeFolder..'dataset-mnist')
require(codeFolder..'ProFi')
require 'paths'


if not opts then
 print '==> processing options'
 cmd = torch.CmdLine()
 cmd:text()
 cmd:text('MNIST/Optimization')
 cmd:text()
 cmd:text('Options:')
 cmd:option('-eta', 0.05, 'LearningRate')
 cmd:option('-save', 'logs', 'subdirectory to save/log experiments in')
 cmd:option('-datasetsize', 'full', 'small|full size of dataset')
 cmd:option('-dataset', 'MNIST', 'Select dataset')
 cmd:option('-seed', 101, 'random seed')
 cmd:option('-folder', '../rbmtest', 'folder where models are saved')
 cmd:option('-traintype', 'CD', 'CD|PCD')
 cmd:option('-ngibbs', 1, 'Number of gibbs steps, e.g CD-5')
 cmd:option('-numepochs', 500, 'Number of epochs')
 cmd:option('-patience', 15, 'Early stopping patience')
 cmd:option('-alpha', 0.5, '0=dicriminative, 1=generative, ]0-1[ = hybrid')
 cmd:option('-beta', 0, 'semisupervised training (NOT IMPLEMENTED)')
 cmd:option('-dropout', 0, 'dropout probability')
 cmd:option('-progress', 1, 'display progressbar')
 cmd:option('-L2', 0, 'weight decay')
 cmd:option('-L1', 0, 'weight decay')
 cmd:option('-momentum', 0, 'momentum')
 cmd:option('-sparsity', 0, 'sparsity')
 cmd:option('-inittype', 'crbm', 'crbm|gauss Gaussian or uniformly drawn initial weights')
 cmd:option('-nhidden', 500, 'number of hidden units')
 cmd:option('-toprbm', true, 'non-toprbms are trained generatively,used for stacking RBMs')
 cmd:option('-batchsize', 1, 'Minibatch size')
 cmd:option('-errfunc', 'acc', 'acc|classacc|spec|sens|mcc|ppv|npv|fpr|fdr|F1| Error measure')
 cmd:option('-pretrain', 'none', 'none|top|bottom specify if rbm will be used in DBM as top or bottom (untested)')
 cmd:text()
 opts = cmd:parse(arg or {})
end


torch.manualSeed(opts.seed)
torch.setdefaulttensortype('torch.FloatTensor')

   -- geometry: width and height of input images
if opts.dataset == "MNIST" then
    geometry = {32,32}
    if opts.datasetsize == 'full' then
      trainData,valData = mnist.loadTrainAndValSet(geometry,'none')
      testData = mnist.loadTestSet(nbTestingPatches, geometry)
    elseif  opts.datasetsize == 'small' then
      print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
      trainData = mnist.loadTrainSet(2000, geometry,'none')
      testData = mnist.loadTestSet(1000, geometry)
      valData = mnist.loadTestSet(1000, geometry)
    else
      print('Unknown datasize')
      error()
    end
    trainData:toProbability()
    valData:toProbability()
    testData:toProbability()

    local errfunc
    local class_to_optimize = 1

    if opts.errfunc == "acc" then
      print('Using 1-accuracy error')
      errfunc = function(conf) return  1-conf:accuracy() end
    elseif opts.errfunc == "spec" then
      print('Using 1-specicity error for class',class_to_optimize)
      errfunc = function(conf) return 1-conf:specificity()[{1,class_to_optimize}] end
    elseif opts.errfunc == "sens" then 
      print('Using 1-sensitivity error for class',class_to_optimize)
      errfunc = function(conf) return 1-conf:sensitivity()[{1,class_to_optimize}] end
    elseif opts.errfunc == "mcc" then 
      print('Using 1-matthew correlation error for class',class_to_optimize)
      errfunc = function(conf) return 1-conf:matthewsCorrelation()[{1,class_to_optimize}] end
    elseif opts.errfunc == "ppv" then 
      print('Using 1-positive predictive value error for class',class_to_optimize)
      errfunc = function(conf) return 1-conf:positivePredictiveValue()[{1,class_to_optimize}] end
    elseif opts.errfunc == "npv" then 
      print('Using 1-negative predictive value error for class',class_to_optimize)
      errfunc = function(conf) return 1-conf:negativePredictiveValue()[{1,class_to_optimize}] end
    elseif opts.errfunc == "fpr" then 
      print('Using 1-false positive rate error for class',class_to_optimize)
      errfunc = function(conf) return 1-conf:falsePositiveRate()[{1,class_to_optimize}] end
    elseif opts.errfunc == "fdr" then 
      print('Using 1-false discovery rate error for class',class_to_optimize)
      errfunc = function(conf) return 1-conf:falseDiscoveryRate()[{1,class_to_optimize}] end
    elseif opts.errfunc == "F1" then  
      print('Using 1-F1 error for class',class_to_optimize)
      errfunc = function(conf) return 1-conf:F1()[{1,class_to_optimize}] end
    elseif opts.errfunc == "classacc" then 
      print('Using 1-class Accuracy error for class',class_to_optimize)
      errfunc = function(conf) return 1-conf:classAccuracy()[{1,class_to_optimize}] end
    else
      print('unknown errorfunction')
      error()
    end

else
    opts.errorfunction = errfunc
    print('unknown dataset')
    error()
end

num_threads = 1
torch.setnumthreads(num_threads)
if torch.getnumthreads() < num_threads then
 print("Setting number of threads had no effect. Maybe install with gcc 4.9 for openMP?")
end

-- SETUP RBM

os.execute('mkdir -p ' .. opts.folder)              -- create tempfolder if it does not exist
opts.finalfile = paths.concat(opts.folder,'final.asc')
opts.tempfile = paths.concat(opts.folder,'temp.asc')  -- current best is saved to this folder
opts.learningrate = opts.eta
opts.n_hidden     = opts.nhidden
opts.cdn = opts.ngibbs

-- DO TRAINING
rbm = rbmsetup(opts,trainData)
rbm = rbmtrain(rbm,trainData,valData)
local err_train = geterror(rbm,trainData)
local err_val = geterror(rbm,valData)
local err_test = geterror(rbm,testData)
print('Train error      : ', err_train)
print('Validation error : ', err_val)
print('Test error       : ', err_test)

