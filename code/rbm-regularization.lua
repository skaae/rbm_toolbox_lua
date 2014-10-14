regularization = {}

function regularization.applyregularization(rbm)
         if rbm.sparsity > 0 then
              -- rbm.db:add(-rbm.sparsity)   -- db is bias of visible layer
               rbm.dc:add(-rbm.sparsity)   -- dc is bias of hidden layer
              -- rbm.dd:add(-rbm.sparsity)   -- dd is bias of "label" layer
          end
          
          if rbm.L1 > 0 then
               rbm.dW:add( -torch.sign(rbm.dW):mul(rbm.L1)  )

               if rbm.toprbm then
                  rbm.dU:add( -torch.sign(rbm.dU):mul(rbm.L1)  )       
               end       
          end
          
          if rbm.L2 > 0 then
               rbm.dW:add( -torch.mul(rbm.dW,rbm.L2 ) )

               if rbm.toprbm then
                  rbm.dU:add( -torch.mul(rbm.dU,rbm.L2 ) )
               end
          end
end


function regularization.dropout(rbm)
     -- Create dropout mask for hidden units
     if rbm.dropout > 0 then
            rbm.dropout_mask = torch.lt( torch.rand(1,rbm.n_hidden),rbm.dropout ):typeAs(rbm.W)
     end     
end