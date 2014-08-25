regularization = {}

function regularization.applyregularization(rbm)
         if rbm.sparsity > 0 then
              -- rbm.db:add(-rbm.sparsity)   -- db is bias of visible layer
               rbm.dc:add(-rbm.sparsity)   -- dc is bias of hidden layer
              -- rbm.dd:add(-rbm.sparsity)   -- dd is bias of "label" layer
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


function regularization.dropout(rbm)
     -- Create dropout mask for hidden units
     if rbm.dropout > 0 then
          rbm.dropout_mask = torch.lt( torch.rand(1,rbm.n_hidden),rbm.dropout ):typeAs(rbm.W)
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