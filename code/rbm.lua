require 'paths'
require('nn')
require('pl')
require('torch')
require 'sys'
require 'xlua'
require(codeFolder.. 'rbm-util')
require(codeFolder.. 'rbm-helpers')
require(codeFolder.. 'rbm-regularization')
require(codeFolder..'rbm-grads')
require(codeFolder..'MyConfusionMatrix')

function rbmtrain(rbm,train,val,semisup)
     local x_train,y_train,x_val,y_val,x_semisup
     if semisup then
        print("semisupervised not implemented")
        error()
     end

     -- train RBM
     local x_tr,y_tr,x_semi, total_time, epoch_time,acc_train, best_val_err,patience, best_rbm,best
     printrbm(rbm,train,val,semisup)

     patience = rbm.patience
     total_time  = os.time()

     -- extend error tensors if resuming training
     if rbm.err_train:size(1) < rbm.numepochs then
          best_val_err = rbm.err_val[rbm.currentepoch]
          rbm.err_recon_train = extendTensor(rbm, rbm.err_recon_train,rbm.numepochs)
          rbm.err_train = extendTensor(rbm,rbm.err_train,rbm.numepochs)
          rbm.err_val = extendTensor(rbm,rbm.err_val,rbm.numepochs) 
          best_rbm = cprbm(rbm)  
     end

     best_val_err = best_val_err or 1/0
     print("Best Val err",best_val_err)
     --print(y_train)
     for epoch = rbm.currentepoch, rbm.numepochs do 
      --print("epcoh",epoch)
          epoch_time = os.time()
          rbm.cur_err = torch.zeros(1)
          rbm.currentepoch = epoch
          
          for i = 1, train:size() do  -- iter over samples
               --x_tr,y_tr,x_semi = getsamples(rbm,x_train,y_train,x_semisup,i)
               
               if rbm.boost == 'none' then
                  train_sample = train:next()
               else 
                  train_sample,skipped =train:nextboost()
               end
               
               x_tr = train_sample[1]:view(1,-1)
               y_tr = train_sample[2]:view(1,-1)
               regularization.dropout(rbm)                          -- create dropout mask for hidden units
               calculategrads(rbm,x_tr,y_tr,x_semi,i)           -- calculates dW, dU, db, dc and dd
               
               -- update vW, vU, vb, vc and vd, formulae: vx = vX*mom + dX                    
               updateweights(rbm,i)
               --print(">>>>updateweights rbm.db: ",rbm.db)                                    -- updates W,U,b,c and d, formulae: X =  X + vX
               
              if rbm.progress > 0 and (i % 100) == 0 then
                xlua.progress(i, train:size())
              end
               
               -- Force garbagecollector to collect
               collectgarbage()

               if rbm.csv then
                 if i % rbm.csv == 1 then
                    sf = paths.concat('e'..epoch..'s'..i)
                    os.execute('mkdir -p ' .. sf)
                    
                    writerbmtocsv(rbm,sf)
                    print('Saving to '..sf)
                 end
             end


             if rbm.boost ~= 'none' and skipped + i > train:size() then
                -- we have passed once through the dataset 
                break
             end
          end  -- end samples loop          
          epoch_time = os.time() - epoch_time 

          -- calc. train recon err and train pred error
          rbm.err_recon_train[epoch] = rbm.cur_err:div(rbm.n_samples)
          
          if rbm.toprbm then
            local err,probs
            
            err,probs = geterror(rbm,train)
            rbm.err_train[epoch] = err 
            if rbm.boost ~= 'none' then
              train:setyprobs(probs)
            end
          end
          
          if val and rbm.toprbm then
              rbm.err_val[epoch] = geterror(rbm,val)
              if rbm.err_val[epoch] < best_val_err then
                  best_val_err = rbm.err_val[epoch]
                  patience = rbm.patience
                  
                  if rbm.tempfile then
                    torch.save(rbm.tempfile,rbm)
                  end
                  best_rbm = cprbm(rbm)     -- save best weights
                  best = '***'
              else
                  patience = patience - 1 
                  best = ''
              end
          end         
          displayprogress(rbm,epoch,epoch_time,patience,best or '')
          


          if patience < 0  then  -- Stop training
                 -- Cp weights from best_rbm
                 rbm.W = best_rbm.W:clone()
                 rbm.U = best_rbm.U:clone()
                 rbm.b = best_rbm.b:clone()
                 rbm.c = best_rbm.c:clone()
                 rbm.d = best_rbm.d:clone()
                 print("BREAK")
                 break
          end

     end  -- end epoch loop
     total_time = os.time() - total_time 
     
     if rbm.finalfile then
      torch.save(rbm.finalfile,rbm)
     end
     print("Mean epoch time:", total_time / rbm.numepochs)
     return(rbm)
end


function calculategrads(rbm,x_tr,y_tr,x_semi,samplenum)
      -- add the grads to dW
      local dW_gen, dU_gen, db_gen, dc_gen, dd_gen, vkx, tcwx 
      local dW_dis, dU_dis, dc_dis, dd_dis, p_y_given_x
      local dW_semi, dU_semi,db_semi, dc_semi, dd_semi, y_semi
      local h0,h0_rnd, hk,vkx,vkx_rnd,vky_rnd
      
      -- reset accumulators
      -- Assert correct formats
      assert(isMatrix(x_tr))
      
      if rbm.toprbm then
        assert(isRowVec(y_tr))
      end

      if x_semi then
        assert(isMatrix(x_semi))
      end
     
     if rbm.precalctcwx == 1 then
      tcwx = torch.mm( x_tr,rbm.W:t() ):add( rbm.c:t() )   -- precalc tcwx
     end
      -- GENERATIVE GRADS
      if rbm.alpha > 0 then
        stat_gen = rbm.generativestatistics(rbm,x_tr,y_tr,tcwx)  
        
        --print(x_tr:type(),y_tr:type(),h0_gen:type(),hk_gen:type(),vkx_rnd_gen:type(),vky_rnd_gen:type())
        grads_gen = rbm.generativegrads(rbm,x_tr,y_tr,stat_gen)
        rbm.dW:add( grads_gen.dW:mul( rbm.alpha*rbm.learningrate ))
        rbm.db:add( grads_gen.db:mul( rbm.alpha*rbm.learningrate ))
        rbm.dc:add( grads_gen.dc:mul( rbm.alpha*rbm.learningrate ))

        if rbm.toprbm then
          rbm.dU:add( grads_gen.dU:mul( rbm.alpha*rbm.learningrate ))
          rbm.dd:add( grads_gen.dd:mul( rbm.alpha*rbm.learningrate ))
        end
        rbm.cur_err:add( torch.sum(torch.add(x_tr,-stat_gen.vkx):pow(2)) ) 


        rbm.stat_gen = stat_gen
        rbm.grads_gen = grads_gen 
      end
     
   --   DISCRIMINATIVE GRADS
      if rbm.alpha < 1 then
        grads_dis, p_y_given_x  = rbm.discriminativegrads(rbm,x_tr,y_tr,tcwx)
        rbm.dW:add( grads_dis.dW:mul( (1-rbm.alpha)*rbm.learningrate ))
        rbm.dU:add( grads_dis.dU:mul( (1-rbm.alpha)*rbm.learningrate ))
        rbm.dc:add( grads_dis.dc:mul( (1-rbm.alpha)*rbm.learningrate ))
        rbm.dd:add( grads_dis.dd:mul( (1-rbm.alpha)*rbm.learningrate ))
      end
      
      -- SEMISUPERVISED GRADS
      if rbm.beta > 0 then
              if rbm.precalctcwx == 1 then
                tcwx_semi = torch.mm( x_semi,rbm.W:t() ):add( rbm.c:t() )   -- precalc tcwx
              end
              
              if rbm.toprbm then
                p_y_given_x = p_y_given_x or rbm.pygivenx(rbm,x_tr,tcwx_semi)
                y_semi = samplevec(p_y_given_x,rbm.rand):resize(1,rbm.n_classes)
              else
                y_semi = {}
              end

              stat_semi = rbm.generativestatistics(rbm,x_semi,y_semi,tcwx_semi)
              grads_semi = rbm.generativegrads(x_semi,y_semi,stat_semi)
              print("FIX problem with PCD chains in semisupevised learning")

              rbm.dW:add( grads_semi.dW:mul( rbm.beta*rbm.learningrate ))      
              rbm.db:add( grads_semi.db:mul( rbm.beta*rbm.learningrate ))
              rbm.dc:add( grads_semi.dc:mul( rbm.beta*rbm.learningrate ))
              
              if rbm.toprbm then
                rbm.dU:add( grads_semi.dU:mul( rbm.beta*rbm.learningrate ))
                rbm.dd:add( grads_semi.dd:mul( rbm.beta*rbm.learningrate ))   
              end   
      end
end

function displayprogress(rbm,epoch,epoch_time,patience,best)
     local strepoch, lrmom, err_recon, err_train, err_val, epoch_time_patience
     
     strepoch   = string.format("%i/%i | ",epoch,rbm.numepochs)
     lrmom = string.format("LR: %f MOM %f | ",rbm.learningrate,rbm.momentum)
     err_recon  = string.format("ERROR: Recon %4.1f ",rbm.err_recon_train[epoch])
     err_train     = string.format("TR ERR: %f ", rbm.err_train[epoch] )
     err_val       = string.format("VAL ERR: %f |", rbm.err_val[epoch] )
     epoch_time_patience = string.format("time: %4.0f Patience %i",epoch_time,patience)
     
     outstr = strepoch .. lrmom .. err_recon .. err_train .. err_val .. epoch_time_patience 
              .. best
     print(outstr)
  
end

function updateweights(rbm,currentsample)
    -- update gradients

    -- fore every minibatch update weights
    if (currentsample % rbm.batchsize) == 0 then 
        regularization.applyregularization(rbm)         -- APPLY REGULATIZATIO BEFORE WEIGHT UPDATE
        if rbm.momentum > 0 then 
            rbm.vW:add( rbm.dW ):mul(rbm.momentum)
            rbm.vb:add( rbm.db ):mul(rbm.momentum) 
            rbm.vc:mad( rbm.dc ):mul(rbm.momentum) 

            -- add momentum to dW 
            rbm.dW:add(rbm.vW)
            rbm.db:add(rbm.vb)
            rbm.dc:add(rbm.vc)

            -- update momentum variable

            if rbm.toprbm then
              rbm.vU:add( rbm.dU ):mul(rbm.momentum)
              rbm.vd:add( rbm.dd ):mul(rbm.momentum)
              rbm.dU:add(rbm.vU)
              rbm.dd:add(rbm.vd)

            end
            
        end

        -- normalize weight update
        if rbm.batchsize > 1 then
          rbm.dW:mul(1/rbm.batchsize)   
          rbm.db:mul(1/rbm.batchsize)
          rbm.dc:mul(1/rbm.batchsize)

          if rbm.toprbm then
            rbm.dU:mul(1/rbm.batchsize)
            rbm.dd:mul(1/rbm.batchsize)
          end
        end

        -- update weights
        rbm.W:add(rbm.dW)
        rbm.b:add(rbm.db)
        rbm.c:add(rbm.dc)

        -- reset weights
        rbm.dW:fill(0)
        rbm.db:fill(0)
        rbm.dc:fill(0)

        if rbm.toprbm then
          rbm.d:add(rbm.dd)
          rbm.U:add(rbm.dU)

          rbm.dd:fill(0)
          rbm.dU:fill(0)
        end
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

-- extend old tensor to 
function extendTensor(rbm,oldtensor,newsize,fill)
     if fill then fill = fill else fill = -1 end
     local newtensor
     newtensor = torch.Tensor(newsize):fill(fill)
     newtensor[{{1,rbm.currentepoch}}] = oldtensor[{{1,rbm.currentepoch}}]:clone()
     return newtensor
end