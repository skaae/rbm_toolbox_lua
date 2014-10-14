require 'torch'
require 'paths'

sigp = {}

-- 

function sigp.loadsigp(files,boost,flip)
   print("boostsetting:", boost)
   print("flipsetting:", flip)
   local datsize,selected_perm,oversample_data,stop,start
   local geometry = {1,882}
   local skipped = 0

    

   local fileData = {} 
   start = 1
   local datasetSize = 0
   for i = 1,#files do
      local file = files[i]
      local dat = torch.load(file)
      
      datsize = dat:size()
      stop = start+datsize
      fileData[i] = {size = datsize,
      start = start,
      stop  = stop-1,
      filename = file}

      start = stop
      datasetSize = datasetSize + datsize
     -- print(dat)
     collectgarbage()
   end
   dat = nil
   collectgarbage()
   --print(fileData[3].size)

   -- local dataset = mnist.loadFlatDataset(fileName, maxLoad)
   local cdataset = {}
   
   -- function cdataset:normalize(m,s)
   --    return dataset:normalize(m,s)
   -- end
   -- function cdataset:normalizeGlobal(m,s)
   --    return dataset:normalizeGlobal(m,s)
   -- end
   function cdataset:setyprobs(yprops)
      myprob = yprops
   end

   function cdataset:size()
       return datasetSize
   end
   function cdataset:resize(nsamples,start)
      dataset:resize(nsamples,start)
   end

   function cdataset:classnames()
      return {'SP', 'CS','TM','OTHER'}
   end

   function cdataset:getfilenames()
      return fileData
   end

   local currentSample = 1
   local nSamples = datasetSize
   local currentPerm   = torch.randperm(nSamples)

   --iterator over dataset 
   function cdataset:next()
      if currentSample > nSamples then
         currentSample = 1
         currentPerm   = torch.randperm(nSamples)
      end
      
      currentSample = currentSample + 1
      return self[currentSample-1]      
   end

   function cdataset:getcurrentsample()
      return currentSample
   end

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
            if sample[2][{2}] == 1 or sample[2][{3}] == 1 then
               -- always take CS and TM classes
               break 
            end


            if self:getcurrentsample() > self:size() then
               print('NUMBER of skipped this pass: ',skipped)
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


   local aasize = 21
   local naa_org = 41
   local aalength = naa_org*aasize      --- change to 861


   function flipAA(seq)
      --print('flipping!')
      --print(seq)
      revseq = seq:clone()
      --local aacount = naa
      naa = naa_org
      for i = 1,aalength,aasize do
         aa = seq[{ {},{i,i+aasize-1}  }]
         --print(naa,aasize,(naa-1)*aasize+1,naa*aasize)
         revseq[{{},{(naa-1)*aasize+1,naa*aasize}}] = aa
         naa = naa - 1
      end
      return revseq
   end


   function cdataset:geometry()
      return geometry
   end

   local currentFile = fileData[1]
   local dataset = torch.load(currentFile.filename)
   local inputdim = 882
   local nclasses = 4
   --dataset.data = dataset.data:squeeze()

   setmetatable(cdataset, {__index = function(self,index)
                                 --print('TRY FLIP with probability 0.5')
                                 --print('Add noise to targets?')
                                       --print(dataset)
                                       
                                       -- Check if index is currently loaded file
                                       --print("INDEX "..index)
                                       
                                       --print("FLIP DATASET")

                                       if not (index >= currentFile.start and index <= currentFile.stop) then
                                          local found = false
                                          for i = 1,#fileData do
                                             f = fileData[i]
                                             --print(f.start,f.stop)
                                             if index >= f.start and index <= f.stop then
                                                currentFile = f
                                                print("Index "..index.." not in current file, Loading File: "..currentFile.filename)
                                                
                                                dataset = torch.load(currentFile.filename)
                                                --dataset.data = dataset.data:squeeze()
                                                found = true
                                                break
                                             end

                                          end

                                          if not found then
                                             print("index "..index.. " out of bounds")
                                             error()
                                          end
                                       end
                                       --print(dataset)
                                       local fileIndex = index-currentFile.start+1 
                                       --print(fileIndex)
                                       local input = dataset.data[fileIndex]
                                       if flip>0 then 
                                          -- flip data with some probability
                                          if torch.rand(1)[1] < flip then
                                             --print("flipped")
                                             input = flipAA(input)
                                          else
                                             --print("not_flipped")
                                          end
                                       end

                                       input = input:view(1,1,inputdim)

                                       --input[{{},{},{1,861}}] = input[{{},{},{1,861}}] - torch.mean(input[{{},{},{1,861}}])
                                       local label = torch.zeros(nclasses)
                                       local label_idx = dataset.labels[fileIndex]
                                       label[{{label_idx}}] = 1
                                       local example = {input, label}
                                       return example
                                       --return 10
                                    end})
   return cdataset
end
