regularization = {}

function regularization.applyregularization(rbm)
          if rbm.sparsity > 0 then
               rbm.db:add(-rbm.sparsity)
               rbm.dc:add(-rbm.sparsity)
               rbm.dd:add(-rbm.sparsity)
          end
          
          if rbm.L1 > 0 then
               rbm.dW:add( -torch.sign(rbm.dW):mul(rbm.L1)  )
               rbm.dU:add( -torch.sign(rbm.dU):mul(rbm.L1)  )              
          end
          
          if rbm.L2 > 0 then
               rbm.dW:add( -torch.mul(rbm.dW,rbm.L2 ) )
               rbm.dU:add( -torch.mul(rbm.dU,rbm.L2 ) )
          end
end


function regularization.applydropoutordropconnect(rbm)
           -- For Dropout and dropconnect keep copy of original weights.
      -- Dropout - maybe add some uption to use the same mask for several samples if cloning is slow?
     if rbm.dropout > 0 or rbm.dropconnect > 0 then
          local W_org,U_org, c_org
          rbm.W_org = rbm.W:clone();   
          rbm.U_org = rbm.U:clone(); 
          rbm.c_org = rbm.c:clone(); 
     end

     -- dropout knocks out rows of weight matrices eq to knocking the corresponding hidden, i.e row 3 = hidden 3
     -- set the weights to something large negatie number to foce the sigm to output 0
     if rbm.dropout > 0 then
          local mask_dropout
          rbm.mask_dropout = regularization.mask2sub( torch.lt( torch.rand(rbm.W:size(1)),rbm.dropout ):typeAs(rbm.W) )
          for i = 1, rbm.mask_dropout:size(1) do rbm.W[{rbm.mask_dropout[i],{} }] =-1000000000 end
          for i = 1, rbm.mask_dropout:size(1) do rbm.U[{rbm.mask_dropout[i],{} }] =-1000000000 end
          for i = 1, rbm.mask_dropout:size(1) do rbm.c[{rbm.mask_dropout[i],{} }] =-1000000000 end
     end

     -- dropconnect randomly knocks out connections
     if rbm.dropconnect > 0 then
          local mask_dropout_W, mask_dropout_U, mask_dropout_c 
          rbm.mask_dropconnect_W = torch.gt( torch.rand( rbm.W:size() ), rbm.dropconnect ):typeAs(rbm.W)
          rbm.mask_dropconnect_U = torch.gt( torch.rand( rbm.U:size() ), rbm.dropconnect ):typeAs(rbm.U)
          rbm.mask_dropconnect_c = torch.gt( torch.rand( rbm.c:size() ), rbm.dropconnect ):typeAs(rbm.c)
          rbm.W = torch.cmul(rbm.W, rbm.mask_dropconnect_W)
          rbm.U = torch.cmul(rbm.U, rbm.mask_dropconnect_U)
          rbm.c = torch.cmul(rbm.c, rbm.mask_dropconnect_c)
     end
     
end

function regularization.mask2sub(x)
     local sub
     sub = torch.zeros(x:sum())
     j = 1
     for i = 1, x:size(1) do
          if x[i] == 1 then
               sub[j] = i
               j = j+1
          end
     end
     return(sub)
end