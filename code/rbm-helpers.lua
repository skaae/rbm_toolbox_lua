-- small math functions and rbmup/down etc
function sigm(x)
     local o = torch.exp(-x):add(1):pow(-1)
     return(o)
end

function normalizeexprowvec(x)
     -- Calculate exp(x) / sum(exp(x)) in numerically stable way
     -- x is a row vector 
     exp_x = torch.exp(x - torch.max(x))
     normalizer = torch.mm(exp_x:sum(2), torch.ones(1,x:size(2)))
     return exp_x:cdiv( normalizer )
end

function softplus(x)  
    local o = torch.exp(x):add(1):log()
    --local o = nn.SoftPlus():forward(x)
    return(o)
end


function rbmup(rbm,x,y,drop) 
    -- drop == 1 applies dropout to p(h|v)
    assert(isRowVec(x))
    assert(x:size(1)*x:size(2) == rbm.n_visible)
    
    local act_hid
    act_hid = torch.mm(x,rbm.W:t()):add(rbm.c:t())  -- x * rbm.W' + rbm.c'
    
    if rbm.toprbm then
        assert(isRowVec(y) and y:size(2) == rbm.n_classes)
        act_hid:add( torch.mm(y,rbm.U:t()) )
    end
    act_hid = sigm(act_hid) 

    if drop == 1 then  
      act_hid:cmul(rbm.dropout_mask)
    end
      
    rbm.act_up = act_hid
    return act_hid
     
end



function rbmdownx(rbm,act_hid)
    -- bernoulli units
    assert(isRowVec(act_hid))

    local act_vis_x
    --act_vis_x =  -- hid_act * rbm.W + rbm.b'
    act_vis_x = sigm(torch.mm(act_hid,rbm.W):add(rbm.b:t()) );
    return act_vis_x
end


function rbmdowny(rbm,act_hid)
     local act_vis_y
     act_vis_y = torch.mm( act_hid,rbm.U ):add( rbm.d:t() )

     act_vis_y = normalizeexprowvec(act_vis_y)
     return act_vis_y
end

-- ##########PRETRAIN FUNCTIONS #################
-- ##  modified for pretraining see Deep boltzmann machines salakhutdinov 2009 sec 3.1
-- ##  basically doubles the input
function rbmuppretrain(rbm,x,y,drop) 
    -- drop == 1 applies dropout to p(h|v)
    -- in the code provided they do not double the biases
    assert(isRowVec(x))
    assert(x:size(1)*x:size(2) == rbm.n_visible)
    
    local act_hid
    act_hid = torch.mm(x,rbm.W:t()):mul(2)  --MODIFIED
    act_hid:add(rbm.c:t())  -- x * rbm.W' + rbm.c'
    
    if rbm.toprbm then
        --assert(isRowVec(y) and y:size(2) == rbm.n_classes)
        act_hid:add( torch.mm(y,rbm.U:t()):mul(2) )  --MODIFIED
    end
    
    act_hid = sigm(act_hid) 

    if drop == 1 then  
      act_hid:cmul(rbm.dropout_mask)
    end
      
    rbm.act_up = act_hid
    return act_hid
     
end

function rbmdownxpretrain(rbm,act_hid)
    -- bernoulli units
    assert(isRowVec(act_hid))

    local act_vis_x
    --act_vis_x =  -- hid_act * rbm.W + rbm.b'
    act_vis_x = torch.mm(act_hid,rbm.W):mul(2)
    act_vis_x:add(rbm.b:t())
    --act_vis_x:mul(2)   -- MODIFICATION
    act_vis_x = sigm(act_vis_x );
    return act_vis_x
end


function rbmdownypretrain(rbm,act_hid)
     local act_vis_y
     act_vis_y = torch.mm( act_hid,rbm.U ):mul(2)
     act_vis_y:add( rbm.d:t() )

     act_vis_y = normalizeexprowvec(act_vis_y)
     return act_vis_y
end
--###### END PRETRAIN FUNCTIONS



function samplevec(x,ran)
    assert(isRowVec(x))
    local r,x_c,larger,sample
    r = ran(1,1):expand(x:size())
    x_c = torch.cumsum(x,2)
    larger = torch.ge(x_c,r)
    sample = torch.eq(torch.cumsum(larger,2),1):typeAs(x) 
    return sample
end

function bernoullisampler(dat,ran)
    local ret = torch.gt(dat, ran(1,dat:size(2))):typeAs(dat)
    return(ret)
end

function gausssampler(dat,ran)
    -- returns ~N(dat,1)
    return torch.randn(dat:size()):add(dat)
end

function classprobs(rbm,x)
    
    local probs,x_i,p_i
     
    -- Iter over examples and calculate the class probs
    probs = torch.Tensor(x:size(1),rbm.n_classes)
    for i = 1, x:size(1) do
      x_i =x[i]:resize(1,rbm.n_visible)

      p_i = rbm.pygivenx(rbm,x_i)
      probs[{i,{}}] = p_i
    end
    return(probs)
end

function predict(rbm,x)
    --print(x)
    --assert(x:dim() == 3)
    --assert(x:size(2)*x:size(3) == rbm.n_visible)

    local probs,_,pred
    probs = classprobs(rbm,x)
    --print(probs)
    -- probs is cases X n_classes
    assert(probs:size(1) == x:size(1) and probs:size(2) == rbm.n_classes)


    vec,pred=torch.max(probs,2)
    local n_samples = x:size(1)
    local labels_vec = torch.zeros(1,rbm.n_classes):float()
    for i =1,n_samples do
        pred_idx = pred[{i,1}]
        labels_vec[{1, pred_idx }] = labels_vec[{1, pred_idx }] + 1
    end  
    
    pred = pred:view(-1)
    return pred:typeAs(x),probs
end

function geterror(rbm,data,errorfunction)
      rbm.conf:zero()
      local probs = torch.Tensor(data:size(),rbm.n_classes)
      for i = 1,data:size() do
        local sample = data[i]
        local x = sample[1]:view(1,-1)
        local _,y_index=torch.max(sample[2],1)        

        local x_pred,x_probs
        x_pred,x_probs = predict(rbm,x)
        probs[{ i,{} }] = x_probs
        rbm.conf:add(x_pred[1], y_index[1])
      end

      local err
      if errorfunction then
        err = errorfunction(rbm.conf)
      else
        err = rbm.errorfunction(rbm.conf)
      end
      return err,probs
end

function rbmuppass(rbm,data,returnLabels)
    -- takes a rbm and calculates the activation of hidden units for all samples
    -- if returnLabels is non nil the function also returns a array of the labels for the 
    -- correpsonding class. This functionality can be used to construct datasets
    -- not for conv rbms?

    local sample,x,y,hid,up_size,n_samples,labels
    n_samples = data:size()
    up_size = rbm.act_up:size(2)
    hid     =  torch.Tensor(n_samples,1,up_size)
    -- also return array of labels
    if returnLabels then
      local class_names = trainData:classnames()
      labels = torch.Tensor(n_samples,#class_names)
    end

    for i = 1,n_samples do
        sample = data:next()
        x = sample[1]:view(1,-1)
        if rbm.toprbm then
            y = sample[2]:view(1,-1)
        else
            y = {}
        end
        
          rbm.up(rbm,x,y,false)   -- updates rbm.act_up in rbm false = no dropout
        act_hid = rbm.act_up:clone()
        hid[{ i,{},{} }] = act_hid:view(1,-1)

        if returnLabels then
          labels[{ i,{} }] = sample[2]:view(1,-1)
        end
        
    end
    return hid,labels
end


function oneOfK(nclasses,labels)
  -- If labels are numeric encodes the function returns the 
  -- lables encoded as one-of-K
  local n_classes, n_samples, labels_vec,i
    n_samples = labels:size(1)
    labels_vec = torch.zeros(n_samples,nclasses)
    for i =1,n_samples do
      labels_vec[{i, labels[i] }] = 1
    end  
  
  return labels_vec

end