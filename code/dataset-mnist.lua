mnist = {}

mnist.path_remote = 'http://data.neuflow.org/data/mnist-th7.tgz'
mnist.path_dataset = 'mnist-th7'
mnist.path_trainset = paths.concat(mnist.path_dataset, 'train.th7')
mnist.path_testset = paths.concat(mnist.path_dataset, 'test.th7')

-- function mnist.reshape(x,h,w)
--     local n_samples = #x.data
--     x.data:reshape(n_samples,h,w)
-- --     for i = 
-- -- end


function mnist.unpack(mnistfolder,tar_folder)
     print(mnistfolder)
   local train = paths.filep(paths.concat(mnistfolder, 'train.th7'))
   local test = paths.filep(paths.concat(mnistfolder, 'test.th7'))
     
   if not train or not test then 
      print("unpacking data...")
      os.execute('tar xvf '.. tar_folder.. '/mnist_lua.tar.gz -C ../')
   else
      print("data already unpacked...skip")
   end
   
end


function mnist.loadFlatDataset(fileName, square)

   local f = torch.DiskFile(fileName, 'r')
   f:binary()

   local nExample = f:readInt()
   local dim = f:readInt()
   print('<mnist> reading ' .. nExample .. ' examples with ' .. dim-1 .. '+1 dimensions...')
   local tensor = torch.Tensor(nExample, dim)
   tensor:storage():copy(f:readFloat(nExample*dim))
   print('<mnist> done')
   

   -- copy x and y
   local x_
   if square then
    x_ = tensor[{ {}, {1,dim-1}}]:reshape(nExample,28,28)
   else
    x_ = tensor[{ {}, {1,dim-1}}]
   end
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

function mnist.createdatasets(mnist_folder,rescale,tar_folder,square)
     if type(tar_folder) ~= 'string' then
      tar_folder =  '../data' 
     end
     square = square or false    
     if not rescale then rescale = 1 end
     local sc = function(s) return math.floor(s*rescale) end
     local getLabels = function(x) 
                            local _,m
                            _,m =torch.max(x,2) 
                            return torch.squeeze(m)
                          end

     mnist.unpack(mnist_folder,tar_folder)

     local train = {}; 
     local val = {};
     local test = {};
     local data_test = mnist.loadFlatDataset(paths.concat(mnist_folder,'test.th7'),square)
     local data_train = mnist.loadFlatDataset(paths.concat(mnist_folder,'train.th7'),square)
     train.data = data_train.x[{{1,sc(50000)},{}}]
     train.labels = getLabels( data_train.y_vec[{{1,sc(50000)},{}}] )
     val.data = data_train.x[{{sc(50001),sc(60000)},{}}]
     val.labels= getLabels( data_train.y_vec[{{sc(50001),sc(60000)},{}}] ) 
     test.data = data_test.x
     test.labels = getLabels( data_test.y_vec )
     return train, val, test
end
