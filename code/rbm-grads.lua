grads = {}
-- Calculate generative weights
-- tcwx is  tcwx = torch.mm( x,rbm.W:t() ):add( rbm.c:t() )
function grads.generativestatistics(rbm,x,y,tcwx)
     local visx, visy, h0,ch_idx,drop, vkx, vkx_rnd, vky_rnd,hk,vky
     local stat = {}
     if rbm.toprbm then
        h0 = sigm( torch.add(tcwx, torch.mm(y,rbm.U:t() ) ) ) --   UP
     else
         h0 = sigm( tcwx ) 
     end

     if rbm.dropout >  0 then
          h0:cmul(rbm.dropout_mask)
          drop = 1
     end
     
     -- Switch between CD and PCD
     if rbm.traintype == 'CD' then -- CD
          -- Use training data as start for negative statistics
          hid = rbm.hidsampler(h0,rbm.rand)   -- sample the hidden derived from training state
     elseif rbm.traintype == 'PCD' then
          -- use pcd chains as start for negative statistics
          ch_idx = math.floor( (torch.rand(1) * rbm.npcdchains)[1]) +1

          local chx =  rbm.chx[ch_idx]:resize(1,x:size(2))
          local chy
          if rbm.toprbm then
            chy = rbm.chy[ch_idx]:resize(1,y:size(2))
          else
            chy = {}
          end
          hid = rbm.hidsampler( rbm.up(rbm, chx, chy, drop), rbm.rand)
     elseif rbm.traintype == 'meanfield' then
           hid = rbm.hidsampler(h0,rbm.rand) 
     end
     

     
     -- If CDn > 1 update chians n-1 times
     for i = 1, (rbm.cdn - 1) do
          visx = rbm.downx( rbm, hid )
          if rbm.traintype ~= 'meanfield' then
            visx = rbm.visxsampler( visx, rbm.rand)   
          end

          
          if rbm.toprbm then
            visy = rbm.downy( rbm, hid)
            if rbm.traintype ~= 'meanfield' then
                visy = samplevec( visy, rbm.rand)
            end
          else
            visy = {}
          end

          hid = rbm.up(rbm,visx, visy, drop)
          if rbm.traintype ~= 'meanfield' then
            hid  = rbm.hidsampler( hid, rbm.rand)
          end


     end
     
               
     -- Down-Up dont sample last hiddens, because it introduces noise
     -- for meanfield we do not sample 
     vkx = rbm.downx(rbm,hid) 
     stat.vkx_unsampled = vkx                           
     if rbm.traintype ~= 'meanfield' then
        vkx = rbm.visxsampler(vkx,rbm.rand)
     end
     if rbm.toprbm then
        vky =   rbm.downy(rbm,hid)                    
        if rbm.traintype ~= 'meanfield' then
            vky = samplevec( vky, rbm.rand) 
        end 
     else
        vky = {}
     end              
     hk = rbm.up(rbm,vkx,vky,drop)   
     
     -- If PCD: Update status of selected PCD chains
     if rbm.traintype == 'PCD' then
          rbm.chx[{ ch_idx,{} }] = vkx
        
          if rbm.toprbm then
            rbm.chy[{ ch_idx,{} }] = vky
          end
     end

     
     stat.h0 = h0
     --stat.h0_rnd = h0_rnd
     stat.hk = hk
     stat.vkx = vkx
     --stat.vkx_rnd = vkx_rnd
     
     if rbm.toprbm then
        stat.vky = vky
        --stat.vky_rnd = vky_rnd
     end
     return stat
end

function grads.generativegrads(rbm,x,y,stat)
     assert(isRowVec(x))
     local grads = {}
     -- Calculate generative gradients  
     grads.dW = torch.mm(stat.h0:t(),x) :add(-torch.mm(stat.hk:t(),stat.vkx)) 
     grads.db = torch.add(x,  -stat.vkx):t()  
     grads.dc = torch.add(stat.h0, -stat.hk ):t() 
     
     if rbm.toprbm then
         assert(isRowVec(y))
        grads.dU = torch.mm(stat.h0:t(),y):add(-torch.mm(stat.hk:t(),stat.vky)) 
        grads.dd = torch.add(y,  -stat.vky):t() 
     end
     return grads

end

-- Calculate discriminative weights
-- tcwx is  tcwx = torch.mm( x,rbm.W:t() ):add( rbm.c:t() )
function grads.discriminativegrads(rbm,x,y,tcwx)
     assert(isRowVec(x))
     assert(isRowVec(y))
     --print("kakkakak")
     local p_y_given_x, F, mask_expanded,F_sigm, F_sigm_prob,F_sigm_prob_sum,F_sigm_dy
     local dW,dU,dc,dd
     
     -- Switch between dropout version and non dropout version of pygivenx
     if rbm.dropout > 0 then
          p_y_given_x, F,mask_expanded = rbm.pygivenxdropout(rbm,x,tcwx)
     else  
          p_y_given_x, F = rbm.pygivenx(rbm,x,tcwx)
     end   

     F_sigm = sigm(F)
     
     -- Apply dropout mask
     if rbm.dropout > 0 then
          F_sigm:cmul(mask_expanded)
     end
     
     F_sigm_prob  = torch.cmul( F_sigm, torch.mm( rbm.hidden_by_one,p_y_given_x ) )
     F_sigm_prob_sum = F_sigm_prob:sum(2)
     F_sigm_dy = torch.mm(F_sigm, y:t())


     dW = torch.add( torch.mm(F_sigm_dy, x), -torch.mm(F_sigm_prob_sum,x) )
     dU = torch.add( -F_sigm_prob, torch.cmul(F_sigm, torch.mm( torch.ones(F_sigm_prob:size(1),1),y ) ) )
     dc = torch.add(-F_sigm_prob_sum, F_sigm_dy)
     dd = torch.add(y, -p_y_given_x):t()

     local grads = {}
     grads.dW = dW
     grads.dU = dU
     grads.dc = dc
     grads.dd = dd
     return grads,p_y_given_x

end


function grads.pygivenx(rbm,x,tcwx_pre_calc)
    assert(isRowVec(x))

    local tcwx,F,pyx
    tcwx_pre_calc = tcwx_pre_calc or torch.mm( x,rbm.W:t() ):add( rbm.c:t() )
    assert(isRowVec(tcwx_pre_calc))  -- 1xn_hidden

    F = torch.add( rbm.U, torch.mm(tcwx_pre_calc:t(), rbm.one_by_classes) )
    pyx = softplus(F):sum(1)        -- p(y|x) logprob
    pyx:add(-torch.max(pyx))       -- subtract max for numerical stability
    pyx:exp()                      -- convert to real domain
    pyx:mul( ( 1/pyx:sum() ))      -- normalize probabilities
    
    assert(pyx:size(1) == 1 and pyx:size(2) == rbm.n_classes)
    return pyx,F
end

function grads.pygivenxdropout(rbm,x,tcwx_pre_calc)
    -- Dropout version of pygivenx
    assert(isRowVec(x))
    local tcwx,F,F_softplus,pyx, mask_expanded
    mask_expanded = torch.mm(rbm.dropout_mask:t(), rbm.one_by_classes)
    tcwx_pre_calc = tcwx_pre_calc or torch.mm( x,rbm.W:t() ):add( rbm.c:t() )
    assert(isRowVec(tcwx_pre_calc))  -- 1xn_hidden

    F   = torch.add( rbm.U, torch.mm(tcwx_pre_calc:t(), rbm.one_by_classes) )
    F:cmul(mask_expanded)          -- Apply dropout mask

    F_softplus = softplus(F)
    F_softplus:cmul(mask_expanded) -- Apply dropout mask

    pyx = F_softplus:sum(1)        -- p(y|x) logprob
    pyx:add(-torch.max(pyx))       -- subtract max for numerical stability
    pyx:exp()                      -- convert to real domain
    pyx:mul( ( 1/pyx:sum() ))      -- normalize probabilities

    assert(pyx:size(1) == 1 and pyx:size(2) == rbm.n_classes)
    return pyx,F,mask_expanded
end