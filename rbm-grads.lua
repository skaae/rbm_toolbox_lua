grads = {}
ProFi = require('ProFi')

-- Calculate generative weights
-- tcwx is  tcwx = torch.mm( x,rbm.W:t() ):add( rbm.c:t() )
function grads.generative(rbm,x,y,tcwx,chx,chy)
     local visx_rnd, visy_rnd, h0, h0_rnd,ch_idx,drop, vkx, vkx_rnd, vky_rnd,hk
     h0 = sigm( torch.add(tcwx, torch.mm(y,rbm.U:t() ) ) ) --   UP
     
     if rbm.dropout >  0 then
          drop = 1
          h0:cmul(rbm.dropout_mask)   -- Apply dropout on p(h|v)
     end
     
     -- Switch between CD and PCD
     if rbm.traintype == 0 then -- CD
          -- Use training data as start for negative statistics
          h0_rnd = sampler(h0,rbm.rand)   -- use training data as start
     else
          -- use pcd chains as start for negative statistics
          ch_idx = math.floor( (torch.rand(1) * rbm.npcdchains)[1]) +1
          h0_rnd = sampler( rbmup(rbm, chx[ch_idx]:resize(1,x:size(2)), chy[ch_idx]:resize(1,y:size(2)), drop), rbm.rand)
     end
     
     -- If CDn > 1 update chians n-1 times
     for i = 1, (rbm.cdn - 1) do
          visx_rnd = sampler( rbmdownx( rbm, h0_rnd ), rbm.rand)   
          visy_rnd = samplevec( rbmdowny( rbm, h0_rnd), rbm.rand)
          hid_rnd  = sampler( rbmup(rbm,visx_rnd, visy_rnd, drop), rbm.rand)
     end
     
               
     -- Down-Up dont sample hiddens, because it introduces noise
     vkx = rbmdownx(rbm,h0_rnd)                            
     vkx_rnd = sampler(vkx,rbm.rand)                      
     vky_rnd = samplevec( rbmdowny(rbm,h0_rnd), rbm.rand)                
     hk = rbmup(rbm,vkx_rnd,vky_rnd,drop)   
     
     -- If PCD: Update status of selected PCD chains
     if rbm.traintype == 1 then
          chx[{ ch_idx,{} }] = vkx_rnd
          chy[{ ch_idx,{} }] = vky_rnd
     end

     -- Calculate generative gradients
     local dW =   torch.mm(h0:t(),x) :add(-torch.mm(hk:t(),vkx_rnd)) 
     local dU =   torch.mm(h0:t(),y):add(-torch.mm(hk:t(),vky_rnd)) 
     local db =   torch.add(x,  -vkx_rnd):t()  
     local dc =   torch.add(h0, -hk ):t() 
     local dd =   torch.add(y,  -vky_rnd):t() 
     return dW, dU, db, dc ,dd, vkx
end

-- Calculate discriminative weights
-- tcwx is  tcwx = torch.mm( x,rbm.W:t() ):add( rbm.c:t() )
function grads.discriminative(rbm,x,y,tcwx)
     local p_y_given_x, F, mask_expanded,F_sigm, F_sigm_prob,F_sigm_prob_sum,F_sigm_dy
     local dW,dU,dc,dd
     
     -- Switch between dropout version and non dropout version of pygivenx
     if rbm.dropout > 0 then
          p_y_given_x, F,mask_expanded = grads.pygivenxdropout(rbm,x,tcwx)
     else  
          p_y_given_x, F = grads.pygivenx(rbm,x,tcwx)
     end   

     F_sigm = sigm(F)
     
     -- Apply dropout mask
     if rbm.dropout > 0 then
          F_sigm:cmul(mask_expanded)
     end
     
     F_sigm_prob  = torch.cmul( F_sigm, torch.mm( rbm.hidden_by_one,p_y_given_x ) )

     F_sigm_prob_sum = F_sigm_prob:sum(2)
     F_sigm_dy = torch.mm(F_sigm, y:t())


     dW =  torch.add( torch.mm(F_sigm_dy, x), -torch.mm(F_sigm_prob_sum,x) )
     dU =  torch.add( -F_sigm_prob, torch.cmul(F_sigm, torch.mm( torch.ones(F_sigm_prob:size(1),1),y ) ) )
     dc = torch.add(-F_sigm_prob_sum, F_sigm_dy)
     dd = torch.add(y, -p_y_given_x):t()


     return dW, dU, dc, dd,p_y_given_x

end


function grads.pygivenx(rbm,x,tcwx_pre_calc)
     local tcwx,F,pyx
     tcwx_pre_calc = tcwx_pre_calc or torch.mm( x,rbm.W:t() ):add( rbm.c:t() )
     F = torch.add( rbm.U, torch.mm(tcwx_pre_calc:t(), rbm.one_by_classes) )
     pyx = softplus(F):sum(1)        -- p(y|x) logprob
     pyx:add(-torch.max(pyx))       -- divide by max,  log domain
     pyx:exp()                      -- convert to real domain
     pyx:mul( ( 1/pyx:sum() ))      -- normalize probabilities
     return pyx,F
end

function grads.pygivenxdropout(rbm,x,tcwx_pre_calc)
     -- Dropout version of pygivenx
     local tcwx,F,pyx, mask_expanded
     mask_expanded = torch.mm(rbm.dropout_mask:t(), rbm.one_by_classes)
     tcwx_pre_calc = tcwx_pre_calc or torch.mm( x,rbm.W:t() ):add( rbm.c:t() )
  
     F   = torch.add( rbm.U, torch.mm(tcwx_pre_calc:t(), rbm.one_by_classes) )
     F:cmul(mask_expanded)          -- Apply dropout mask
     
     F_softplus = softplus(F)
     F_softplus:cmul(mask_expanded) -- Apply dropout mask
     
     pyx = F_softplus:sum(1)        -- p(y|x) logprob
     pyx:add(-torch.max(pyx))       -- divide by max,  log domain
     pyx:exp()                      -- convert to real domain
     pyx:mul( ( 1/pyx:sum() ))      -- normalize probabilities
     return pyx,F,mask_expanded
end


function grads.calculategrads(rbm,x_tr,y_tr,x_semi)
      local dW_gen, dU_gen, db_gen, dc_gen, dd_gen, vkx 
      local dW_dis, dU_dis, dc_dis, dd_dis, p_y_given_x
      local dW_semi, dU_semi,db_semi, dc_semi, dd_semi, y_semi
      local tcwx = torch.mm( x_tr,rbm.W:t() ):add( rbm.c:t() )   -- precalc tcwx
      
      -- reset accumulators
      rbm.dW:fill(0)
      rbm.dU:fill(0)
      rbm.db:fill(0)
      rbm.dc:fill(0)
      rbm.dd:fill(0)

      -- GENERATIVE GRADS
      if rbm.alpha > 0 then
        dW_gen, dU_gen, db_gen, dc_gen, dd_gen, vkx  = grads.generative(rbm,x_tr,y_tr,tcwx,rbm.chx,rbm.chy)
        rbm.dW:add( dW_gen:mul( rbm.alpha ))
        rbm.dU:add( dU_gen:mul( rbm.alpha ))
        rbm.db:add( db_gen:mul( rbm.alpha ))
        rbm.dc:add( dc_gen:mul( rbm.alpha ))
        rbm.dd:add( dd_gen:mul( rbm.alpha ))
        rbm.cur_err:add(   torch.sum(torch.add(x_tr,-vkx):pow(2)) )  
      end
     
      -- DISCRIMINATIVE GRADS
      if rbm.alpha < 1 then

        dW_dis, dU_dis, dc_dis, dd_dis, p_y_given_x  = grads.discriminative(rbm,x_tr,y_tr,tcwx)
        rbm.dW:add( dW_dis:mul( 1-rbm.alpha ))
        rbm.dU:add( dU_dis:mul( 1-rbm.alpha ))
        rbm.dc:add( dc_dis:mul( 1-rbm.alpha ))
        rbm.dd:add( dd_dis:mul( 1-rbm.alpha ))

      end

      
      -- SEMISUPERVISED GRADS
      if rbm.beta > 0 then
               grads.p_y_given_x = p_y_given_x or grads.pygivenx(rbm,x_tr,tcwx)
               y_semi = samplevec(p_y_given_x,rbm.rand):resize(1,rbm.n_classes)
               dW_semi, dU_semi,db_semi, dc_semi, dd_semi = grads.generative(rbm,x_semi,y_semi,tcwx,rbm.chx_semisup,rbm.chy_semisup)
               rbm.dW:add( dW_semi:mul( rbm.beta ))
               rbm.dU:add( dU_semi:mul( rbm.beta ))
               rbm.db:add( db_semi:mul( rbm.beta ))
               rbm.dc:add( dc_semi:mul( rbm.beta ))
               rbm.dd:add( dd_semi:mul( rbm.beta ))      
      end
end