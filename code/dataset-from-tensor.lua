require 'torch'
require 'paths'

datatensor = {}
function datatensor.createDataset(tensor,labels,classes,geometry)
   local cdataset = {}

   if tensor:dim() ~= 3 then
      print("Tensor dim must be batches X datadim1 X datadim2")
      error()
   end


   local tensor = tensor:clone()
   local labels = labels:clone()
   local dim = tensor:size(2)
   --cdataset.classes = classes
   
   function cdataset:normalize(mean_, std_)
      print('Not implemented - see dataset-mnist.lua')
      error()
   end
   function cdataset:normalizeGlobal(mean_, std_)
      local std = std_ or tensor:std()
      local mean = mean_ or tensor:mean()
      tensor:add(-mean)
      tensor:mul(1/std)
      return mean, std
   end
   function cdataset:size()
      return tensor:size(1)
   end
   function cdataset:resize(nsamples,start)
      start = start or 1
      tensor = tensor[{{start,nsamples},{},{} }]
      labels = labels[{{start,nsamples},{}}]
   end

   function cdataset:toProbability()
      tensor:add( tensor:min() ) -- minimum to 0 
      tensor:mul(1/tensor:max()) -- maximimum to 1
   end

   function cdataset:classnames()
      return classes
   end

   function cdataset:getTensor()
      return tensor
   end

   function cdataset:getLabels()
      return labels
   end


   local currentSample = 1
   local nSamples = tensor:size(1)
   local currentPerm   = torch.randperm(nSamples)

   --iterator over dataset 
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

   setmetatable(cdataset, {__index = function(self,index)
                                       --print(dataset)
                                       local x = tensor[{ index,{},{}  }]
                                       local x = x:view(1,geometry[1],geometry[2])
                                       local y = labels[{index,{}}]:view(-1)
                                       local example = {x, y}
                                       return example
                                    end})
   return cdataset
end
