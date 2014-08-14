require('nn')
require('pl')
require('torch')
require('rbm-util')
require('rbm-regularization')
require('rbm-grads')
require ('socket')  -- for timing 



function rbmtrain(rbm,x_train,y_train,x_val,y_val,x_semisup)
-- train RBM
-- Reset gradient accums
-- Print rbm
--print(x_train)
--print(y_train)
--print(x_val)
--print(y_val)
--print(x_semisup)
printRBM(rbm,x_train,x_val,x_semisup)

rbm.time = {}

local x_tr,y_tr,x_semi, total_time, epoch_time,acc_train
local best_val_err = 1/0
local patience = rbm.patience
total_time  = socket.gettime()

for epoch = 1, rbm.numepochs do 
     epoch_time = socket.gettime()
     rbm.cur_err = torch.zeros(1)

     for i = 1, x_train:size(1) do  -- iter over samples


          x_tr,y_tr,x_semi = getsamples(rbm,x_train,y_train,x_semisup,i)
          regularization.applydropoutordropconnect(rbm)        -- cp org weights, drops weights if enabled
          grads.calculategrads(rbm,x_tr,y_tr,x_semi)                -- calculates dW, dU, db, dc and dd
          regularization.applyregularization(rbm)              -- regularizes dW, dU, db, dc and dd
          updategradsandmomentum(rbm) 
           
          -- update vW, vU, vb, vc and vd, formulae: vx = vX*mom + dX
          restoreorgweights(rbm)                               -- restore weights from before dropping                        
          updateweights(rbm)                                   -- updates W,U,b,c and d, formulae: X =  X + vX
          
          
          if (i %  5000) == 0 then
               io.write(".") 
               io.flush()
          end
          
          -- Force garbagecollector to collect. Reduces memory with atleast factor of 3.
          if (i % 100) == 0 then
               collectgarbage()
          end

          end  -- samples loop
     epoch_time = socket.gettime() - epoch_time 
     rbm.err_recon_train[epoch]    = rbm.cur_err:div(rbm.n_samples)
     
     
     --timer = torch.Timer()
     err_train = 1-accuracy(rbm,x_train,y_train)
     rbm.err_train[epoch] = err_train
     --print(timer:time().real)
     
     if x_val and y_val then
          err_val = 1-accuracy(rbm,x_val,y_val)
          rbm.err_val[epoch] = err_val
     if err_val < best_val_err then
          best_val_err = err_val
          patience = rbm.patience
          saverbm(rbm.tempfile,rbm) -- save current best
          best_rbm = cprbm(rbm)     -- save best weights
          best = '***'
     else
          patience = patience - 1 
          best = ''
     end
     end
     
     print(string.format("%i/%i -LR: %f, MOM %f -  Recon err: %4.1f err TR: %f err VAL: %f  time: %f Patience %i  %s", epoch, rbm.numepochs,                     rbm.learningrate,rbm.momentum, rbm.cur_err[1],err_train,err_val or 1/0,epoch_time, patience,best))
     
     
     if patience < 0 then  -- Stop training
          -- Cp weights from best_rbm
          rbm.W = best_rbm.W:clone()
          rbm.U = best_rbm.U:clone()
          rbm.b = best_rbm.b:clone()
          rbm.c = best_rbm.c:clone()
          rbm.d = best_rbm.d:clone()
          break
     end
     
     end

total_time = socket.gettime() - total_time 
print("Mean epoch time:", total_time / rbm.numepochs)
return(rbm)

end

function getsamples(rbm,x_train,y_train,x_semisup,i_tr)
     local x_tr, y_tr
     x_tr = x_train[i_tr]:resize(1,x_train:size(2))
     y_tr = y_train[i_tr]:resize(1,y_train:size(2))
     if rbm.beta > 0 then
           i_semi = (i_tr-1) % x_semisup:size(1) +1;
           x_semi = x_semisup[i_tr]:resize(1,x_semisup:size(2))
     end
     
     return x_tr,y_tr,x_semi
end

function updategradsandmomentum(rbm)
      -- multiply updates by learningrate
      rbm.dW:mul(rbm.learningrate)
      rbm.dU:mul(rbm.learningrate)
      rbm.db:mul(rbm.learningrate)
      rbm.dc:mul(rbm.learningrate)
      rbm.dd:mul(rbm.learningrate)
      
      -- If momentum is zero this will be zero
      --if momentum > 0 then
      
      
     if rbm.momentum > 0 then  
          rbm.vW:mul(rbm.momentum)
          rbm.vU:mul(rbm.momentum)
          rbm.vb:mul(rbm.momentum)
          rbm.vc:mul(rbm.momentum)
          rbm.vd:mul(rbm.momentum)
     else
          rbm.vW:fill(0)
          rbm.vU:fill(0)
          rbm.vb:fill(0)
          rbm.vc:fill(0)
          rbm.vd:fill(0) 
     end
     
      -- Add current update to momentum
      rbm.vW:add(rbm.dW)
      rbm.vU:add(rbm.dU)
      rbm.vb:add(rbm.db)
      rbm.vc:add(rbm.dc)
      rbm.vd:add(rbm.dd)
end

function updateweights(rbm)
      -- update gradients
      rbm.W:add(rbm.vW)
      rbm.U:add(rbm.vU)
      rbm.b:add(rbm.vb)
      rbm.c:add(rbm.vc)
      rbm.d:add(rbm.vd)
end




function restoreorgweights(rbm)
     if rbm.dropout > 0 or rbm.dropconnect > 0 then
          -- TODO: not sure if i need to clone here
          rbm.W = rbm.W_org:clone();    
          rbm.U = rbm.U_org:clone(); 
          rbm.c = rbm.c_org:clone(); 
     end
end

function cprbm(rbm)
     newrbm = {}
     newrbm.W = rbm.W:clone()
     newrbm.U = rbm.U:clone()
	newrbm.b = rbm.b:clone()
	newrbm.c = rbm.c:clone()
	newrbm.d = rbm.d:clone()
     return(newrbm)
end



