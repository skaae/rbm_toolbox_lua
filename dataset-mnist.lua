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

function mnist.loadFlatDataset(fileName, maxLoad)

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
   

   -- copy x and y 
   local x_ = tensor[{ {}, {1,dim-1}}]
   local ynum_ = tensor[{ {}, dim}]
   local ynum = ynum_:clone(ynum_)
   local x = x_:clone(x_)

   local y_vec = torch.zeros(nExample,10)
   local c
   for i = 1, nExample do
         c = ynum[i]+1
         y_vec[{ i,c }] = 1
   end
   local dataset = {}
   dataset.x = x:div(255)
   dataset.y = ynum
   dataset.y_vec = y_vec 

   return dataset
end