codeFolder = '../code/'

require('torch')
require(codeFolder..'rbm')
require(codeFolder..'dataset-mnist')
require(codeFolder..'ProFi')
require(codeFolder..'dataset-from-tensor.lua')
require 'paths'
local opts
if not opts then
 print '==> processing options'
 cmd = torch.CmdLine()
 cmd:text()
 cmd:text('MNIST/Optimization')
 cmd:text()
 cmd:text('Options:')
 cmd:option('-eta', 0.05, 'LearningRate')
 cmd:option('-save', 'logs', 'subdirectory to save/log experiments in')
 cmd:option('-datasetsize', 'small', 'small|full  size of dataset')
 cmd:option('-dataset', 'MNIST', 'MNIST|SIGP select dataset')
 cmd:option('-seed', 101, 'random seed')
 cmd:option('-folder', '../rbmtest', 'folder where models are saved')
 cmd:option('-traintype', 'CD', 'CD|PCD')
 cmd:option('-ngibbs', 1, 'Number of gibbs steps, e.g CD-5')
 cmd:option('-numepochs1', 1, 'Number of epochs rbm1')
 cmd:option('-numepochs2', 1, 'Number of epochs rbm2')
 cmd:option('-numepochs3', 1, 'Number of epochs rbm3')
 cmd:option('-numepochs4', 1, 'Number of epochs rbm4')
 cmd:option('-patience', 3, 'Early stopping patience')
 cmd:option('-alpha', 0.0, '0=dicriminative, 1=generative, ]0-1[ = hybrid')
 cmd:option('-beta', 0, 'semisupervised training (NOT IMPLEMENTED)')
 cmd:option('-dropout', 0, 'dropout probability')
 cmd:option('-progress', 1, 'display progressbar')
 cmd:option('-L2', 0, 'weight decay')
 cmd:option('-L1', 0, 'weight decay')
 cmd:option('-momentum', 0, 'momentum')
 cmd:option('-sparsity', 0, 'sparsity')
 cmd:option('-inittype', 'crbm', 'crbm|gauss Gaussian or uniformly drawn initial weights')
 cmd:option('-nhidden1', 10, 'number of hidden units in RBM1')
 cmd:option('-nhidden2', 10, 'number of hidden units in RBM2')
 cmd:option('-nhidden3', 10, 'number of hidden units in RBM3')
 cmd:option('-nhidden4', 10, 'number of hidden units in RBM4')
 cmd:option('-batchsize', 1, 'Minibatch size')
 cmd:option('-errfunc', 'acc', 'acc|classacc|spec|sens|mcc|ppv|npv|fpr|fdr|F1| Error measure does not apply to sigp data')
  cmd:option('-boost', 'none', 'none|diff enable disable boosting')
 cmd:option('-flip', 0, 'flip data probability SIGP dataset only')
 cmd:text()
 opts = cmd:parse(arg or {})
end

torch.manualSeed(opts.seed)
torch.setdefaulttensortype('torch.FloatTensor')

if opts.dataset == "MNIST" then
    geometry = {32,32}
    if opts.datasetsize == 'full' then
      trainData,valData = mnist.loadTrainAndValSet(geometry,opts.boost)
      testData = mnist.loadTestSet(nbTestingPatches, geometry)
    elseif  opts.datasetsize == 'small' then
      print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
      trainData = mnist.loadTrainSet(2000, geometry,opts.boost)
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


    opts.errorfunction = errfunc
    --print(opts.errorfunction,errfunc)
    --error()
elseif opts.dataset == 'SIGP' then
  --th runRBM.lua -eta 0.05 -alpha 1 -nhidden 500 -folder sigp_discriminative -dataset SIGP -progress 0
   require(codeFolder..'dataset-sigp')
   geometry = {1,882}
   classes = {'SP','CS','TM','OTHER'}
   if opts.datasetsize == "full" then
      print("Loading full SIGP dataset...")
      trainFiles = {'../sigp-data/Eukar1.dat',
                     '../sigp-data/Eukar2.dat',
                     '../sigp-data/Eukar3.dat'}
      trainData = sigp.loadsigp(trainFiles,opts.boost,opts.flip)
   elseif opts.datasetsize == "small" then
      print("Loading small SIGP dataset...")
      trainFiles = {'../sigp-data/Eukar1.dat'}
      trainData = sigp.loadsigp(trainFiles,opts.boost,opts.flip)
   elseif opts.datasetsize == "schmidhuber" then
    --th runRBM.lua -dataset SIGP -datasetsize schmidhuber -nhidden 100 -numepochs 5000 -alpha 0 -folder schmidhuber_probs
    require(codeFolder..'dataset-from-tensor.lua')
        local trainTensor = torch.load('../sigp-data/schmidhuber_sigp_inputs.dat'):view(-1,1,882)
        local trainLabels = torch.load('../sigp-data/schmidhuber_sigp_targets.dat'):view(-1,1,4)
        trainData = datatensor.createDataset(trainTensor,trainLabels,classes,geometry)
        -- for i = 1,10000 do
        --   if trainData:next()[2]:sum() > 1 then
        --     print(i)
        --   end
        -- end
        -- error()
   elseif opts.datasetsize == "schmidhuberweighted" then   
   --th runRBM.lua -dataset SIGP -datasetsize schmidhuberweighted -nhidden 100 -numepochs 5000 -alpha 0 -folder schmidhuber_weighted
        require(codeFolder..'dataset-from-tensor.lua')
        local trainTensor = torch.load('../sigp-data/schmidhuber_sigp_inputs_weighted.dat'):view(-1,1,882)
        local trainLabels = torch.load('../sigp-data/schmidhuber_sigp_targets_weighted.dat'):view(-1,1,4)
        trainData = datatensor.createDataset(trainTensor,trainLabels,classes,geometry)  
   else
    print('unknown datasize')
    error()
   end
   valFiles =   {'../sigp-data/Eukar4.dat'}
   testFiles =   {'../sigp-data/Eukar5.dat'}
   
   valData = sigp.loadsigp(valFiles,'none',0)
   testData = sigp.loadsigp(testFiles,'none',0)
   

   if opts.datasetsize ~= "schmidhuber" then
     errorfunc = function(conf) 
                 conf:updateValids()
                 conf:printscore('mcc')
                 local mcc = conf:matthewsCorrelation()
                 return 1-mcc[{1,2}]
              end
    else
      errorfunc = function(conf) 
                 conf:updateValids()
                 conf:printscore('acc')
                 local mcc = conf:matthewsCorrelation()
                 return 1-conf:accuracy()
              end
    end
    opts.errorfunction =errorfunc
elseif opts.dataset == 'NEWS' then
  print("Loading NEWS data")
  require(codeFolder..'dataset-from-tensor.lua')
  trainTensor = torch.load('../20news-data/trainData.dat'):view(-1,1,5000)
  valTensor  = torch.load('../20news-data/valData.dat'):view(-1,1,5000)
  testTensor = torch.load('../20news-data/testData.dat'):view(-1,1,5000)
  trainLabels = torch.load('../20news-data/trainLabels.dat'):view(-1)
  valLabels = torch.load('../20news-data/valLabels.dat'):view(-1)
  testLabels = torch.load('../20news-data/testLabels.dat'):view(-1)
  classes = {} 
  for i = 1,20 do classes[i] = tostring(i) end
  geometry = {1,5000}
  trainData = datatensor.createDataset(trainTensor,oneOfK(20,trainLabels),classes,geometry)
  valData = datatensor.createDataset(valTensor,oneOfK(20,valLabels),classes,geometry)
  testData = datatensor.createDataset(testTensor,oneOfK(20,testLabels),classes,geometry)


else 
    
    print('unknown dataset')
    error()
 end

--print(test)
num_threads = 1
torch.setnumthreads(num_threads)
if torch.getnumthreads() < num_threads then
 print("Setting number of threads had no effect. Maybe install with gcc 4.9 for openMP?")
end


topAlpha = opts.alpha
-- SETUP RBM

os.execute('mkdir -p ' .. opts.folder)              -- create tempfolder if it does not exist
opts.learningrate = opts.eta


geometry = {1,opts.nhidden1}
classes = trainData:classnames()
function train(layer,nhidden,numepochs,toprbm,alpha,dropout,traindata,valdata,testdata)
  print("################# TRAINING RBM "..layer .." #################")
  returnLabels = true
  opts.finalfile = paths.concat(opts.folder,'final_rbm'..layer..'.asc')
  opts.tempfile = paths.concat(opts.folder,'temp_rbm'..layer..'.asc')  -- current best is saved to this folder
  opts.n_hidden     = nhidden
  opts.numepochs = numepochs
  opts.toprbm = toprbm
  opts.isgpu = 0
  opts.alpha = alpha -- generative for non top rbm
  opts.dropout = dropout

  local rbm = rbmsetup(opts,traindata)
  rbm = rbmtrain(rbm,traindata,valdata)


  trainTensor2,trainLabels2 = rbmuppass(rbm,traindata,returnLabels)
  valTensor2,valLabels2 = rbmuppass(rbm,valdata,returnLabels)
  testTensor2,testLabels2 = rbmuppass(rbm,testdata,returnLabels)


  traindata2 = datatensor.createDataset(trainTensor2,trainLabels2,classes,geometry)
  valdata2 = datatensor.createDataset(valTensor2,valLabels2,classes,geometry)
  testdata2 = datatensor.createDataset(testTensor2,testLabels2,classes,geometry)
  return rbm,traindata2,valdata2,testdata2
end




-- Train layer one
print(trainData,valData,testData)
rbm1,trainData2,valData2,testData2 = train(1,opts.nhidden1,opts.numepochs1,false,1,opts.dropout,trainData,valData,testData)
collectgarbage()
rbm2,trainData3,valData3,testData3 = train(2,opts.nhidden2,opts.numepochs2,false,1,opts.dropout,trainData2,valData2,testData2)
collectgarbage()
rbm3,trainData4,valData4,testData4 = train(3,opts.nhidden3,opts.numepochs3,false,1,opts.dropout,trainData3,valData3,testData3)
collectgarbage()
rbm4,trainData5,valData5,testData5 = train(4,opts.nhidden4,opts.numepochs4,true,opts.alpha,opts.dropout,trainData4,valData4,testData4)






-- DO TRAINING
--rbm = trainAndPrint(opts,train,val,test,tempfolder,finalfile)

local err_train = geterror(rbm4,trainData4)
local err_val = geterror(rbm4,valData4)
local err_test = geterror(rbm4,testData4)
print('Train error      : ', err_train)
print('Validation error : ', err_val)
print('Test error       : ', err_test)



