require 'torch'
require 'paths'

mnist = {}

mnist.path_remote = 'http://data.neuflow.org/data/mnist-th7.tgz'
mnist.path_dataset = 'mnist-th7'
mnist.path_trainset = paths.concat(mnist.path_dataset, 'train.th7')
mnist.path_testset = paths.concat(mnist.path_dataset, 'test.th7')

function mnist.download()
   if not paths.filep(mnist.path_trainset) or not paths.filep(mnist.path_testset) then
      local remote = mnist.path_remote
      local tar = paths.basename(remote)
      os.execute('wget ' .. remote .. '; ' .. 'tar xvf ' .. tar .. '; rm ' .. tar)
   end
end

function mnist.loadTrainSet(maxLoad, geometry,boost)
   boost = boost or 'none'
   return mnist.loadConvDataset(mnist.path_trainset, maxLoad, geometry,nil,boost,'train')
end

function mnist.loadTrainAndValSet(geometry,boost)
   boost = boost or 'none'
   local train =  mnist.loadConvDataset(mnist.path_trainset, 60000, geometry,'train',boost)
   local val   =  mnist.loadConvDataset(mnist.path_trainset, 60000, geometry,'val')
   return train,val
end

function mnist.loadTestSet(maxLoad, geometry)
   return mnist.loadConvDataset(mnist.path_testset, maxLoad, geometry,nil,nil,'test')
end

function mnist.loadFlatDataset(fileName, maxLoad,trainOrVal)
   mnist.download()

   local f = torch.DiskFile(fileName, 'r')
   f:binary()

   local nExample = f:readInt()
   local dim = f:readInt()
   if maxLoad and maxLoad > 0 and maxLoad < nExample then
      nExample = maxLoad
      print('<mnist> loading only ' .. nExample .. ' examples')
   end
   print('<mnist> reading ' .. nExample .. ' examples with ' .. dim-1 .. '+1 dimensions...')
   local tensor = torch.Tensor(nExample, dim)
   tensor:storage():copy(f:readFloat(nExample*dim))
   print('<mnist> done')

   if trainOrVal then
      if trainOrVal == 'train' then
         tensor = tensor[{{1,50000},{} }]
      elseif trainOrVal == 'val' then
         tensor = tensor[{{50001,60000},{} }]
      else
         print('trainOrVal must be train|val')
         error()
      end
   end
   local dataset = {}
   dataset.tensor = tensor



   function dataset:normalize(mean_, std_)
      local data = tensor:narrow(2, 1, dim-1)
      local std = std_ or torch.std(data, 1, true)
      local mean = mean_ or torch.mean(data, 1)
      for i=1,dim-1 do
         tensor:select(2, i):add(-mean[1][i])
         if std[1][i] > 0 then
            tensor:select(2, i):mul(1/std[1][i])
         end
      end
      return mean, std
   end

   function dataset:toProbability()
      local data = tensor:narrow(2, 1, dim-1)
      data:mul(1/255)
   end

   function dataset:resize(nsamples,start)
      start = start or 1
      tensor = tensor[{{start,nsamples},{} }]
   end

   function dataset:normalizeGlobal(mean_, std_)
      local data = tensor:narrow(2, 1, dim-1)
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std
   end

   dataset.dim = dim-1

   function dataset:size()
      return tensor:size(1)
   end

   local labelvector = torch.zeros(10)

   setmetatable(dataset, {__index = function(self, index)
                                       local input = tensor[index]:narrow(1, 1, dim-1)
                                       local class = tensor[index][dim]+1
                                       local label = labelvector:zero()
                                       label[class] = 1
                                       local example = {input, label}
                                       return example
                                    end})

   return dataset
end

function mnist.loadConvDataset(fileName, maxLoad, geometry,trainOrVal,boost,name)
   local dataset = mnist.loadFlatDataset(fileName, maxLoad,trainOrVal)
   local cdataset = {}
   
   function cdataset:normalize(m,s)
      return dataset:normalize(m,s)
   end
   function cdataset:normalizeGlobal(m,s)
      return dataset:normalizeGlobal(m,s)
   end
   function cdataset:size()
      return dataset:size()
   end
   function cdataset:resize(nsamples,start)
      dataset:resize(nsamples,start)
   end

   function cdataset:toProbability()
      dataset:toProbability()
   end

   function cdataset:classnames()
      return {'1','2','3','4','5','6','7','8','9','10'}
   end


   local currentSample = 1
   local nSamples = dataset:size()
   local currentPerm   = torch.randperm(nSamples)
   local skipped = 0

   --iterator over dataset 
   function cdataset:nextboost()
      if boost == 'none' then
            print('boost is not enabled for this dataset')
            error()
      end 
            --if boost == 'diff' then
      -- print('add some function that samples based on yprop')
      -- print('+add function to get number of indeces check')
      -- print('+in sigp set up some heuristic as in schmidhuber - here use som general 1-error prob or whatever')
      local sample
      if myprob then-- check if yprobs is set, which happens after the first epoch
         local choosen = false
         while not choosen do
            sample = self:next()
            if self:getcurrentsample() > self:size() then
               print(name,'NUMBER of skipped this pass: ',skipped)
               skipped = 0
            end
            --print(ex[2])
            local _,idx = torch.max(sample[2],1)
            
            local pred_prob = myprob[{currentSample-1,idx[1]}]
            --print(currentSample,correct_prob)
            local sampling_prob = 1-pred_prob
            if torch.rand(1)[1] < sampling_prob then
               choosen = true
            else
               skipped = skipped +1
            end
         end
      else

         -- myprop or boost is not defined
         sample = self:next()
      end

      return sample,skipped

   end


   function cdataset:next()
      if currentSample > nSamples then
         currentSample = 1
         currentPerm   = torch.randperm(nSamples)

      end
      
      currentSample = currentSample + 1
      return self[ currentPerm[currentSample-1] ]      
   end

   function cdataset:getcurrentsample()
      return currentSample
   end

   function cdataset:geometry()
      return geometry
   end

   function cdataset:setyprobs(yprops)
      myprob = yprops
      --print(myprob)
   end


   local iheight = geometry[2]
   local iwidth = geometry[1]
   local inputpatch = torch.zeros(1, iheight, iwidth)

   setmetatable(cdataset, {__index = function(self,index)
                                       local ex = dataset[index]
                                       local input = ex[1]
                                       local label = ex[2]
                                       local w = math.sqrt(input:nElement())
                                       local uinput = input:unfold(1,input:nElement(),input:nElement())
                                       local cinput = uinput:unfold(2,w,w)
                                       local h = cinput:size(2)
                                       local w = cinput:size(3)
                                       local x = math.floor((iwidth-w)/2)+1
                                       local y = math.floor((iheight-h)/2)+1
                                       inputpatch:narrow(3,x,w):narrow(2,y,h):copy(cinput)
                                       local example = {inputpatch, label}
                                       return example
                                    end})
   return cdataset
end
