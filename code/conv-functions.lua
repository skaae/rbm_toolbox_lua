local conv = {}

function conv.generativestatistics(rbm,x,y,tcwx)
    assert(isRowVec(x))
    assert(x:size(2) == rbm.n_visible)
    
    if rbm.toprbm then
        assert(isRowVec(y))
        assert(y:size(2) == rbm.n_classes)   
    end
-- tcwx will be nil
    local h0,h0_rnd,vkx,vky,vkx_rnd,vky_rnd,hk
    h0  = rbm.up(rbm,x,y,drop)

    if rbm.dropout >  0 then
          h0:cmul(rbm.dropout_mask)
          drop = 1
    end

    h0_rnd  = rbm.hidsampler(h0,rbm.rand) 
    
    if rbm.dropout >  0 then
          h0_rnd:cmul(rbm.dropout_mask)   -- Apply dropout on p(h|v)
    end

    vkx  = rbm.downx(rbm,h0_rnd)
    
    if rbm.toprbm then
        vky  = rbm.downy(rbm,h0_rnd)                     
        vky_rnd = samplevec( vky, rbm.rand)  
    else
        vky,vky_rnd = {},{}
    end
    vkx_rnd = rbm.visxsampler(vkx,rbm.rand) 
    hk  = rbm.up(rbm,vkx,vky_rnd,drop)   -- Why not vkx_RND?????

    local stat = {}
    stat.h0 = h0
    stat.h0_rnd = h0_rnd
    stat.hk = hk
    stat.vkx = vkx
    stat.vkx_rnd = vkx_rnd
    
    if rbm.toprbm then
        stat.vky = vky
        stat.vky_rnd = vky_rnd
    end

    return stat
end

function conv.creategenerativegrads(sizes)
    local fw = sizes.filter_size * sizes.filter_size
    local fs = sizes.filter_size
    local is = sizes.input_size  
    local nf = sizes.n_filters
    local ni = sizes.n_input
    local ns = sizes.n_visible
    local nc = sizes.n_classes

    -- Setup networks 
    local nn_dW_pos, nn_dW_neg
    nn_dW_pos = nn.SpatialConvolutionMM(n_input, 1, sizes.hid_w, sizes.hid_h)
    nn_dW_neg = nn.SpatialConvolutionMM(n_input, 1, sizes.hid_w, sizes.hid_h)
    nn_dW_pos.bias = torch.zeros(nn_dW_pos.bias:size()) 
    nn_dW_neg.bias = torch.zeros(nn_dW_neg.bias:size())

    local debug = {}
    debug.nn_dW_pos = nn_dW_pos
    debug.nn_dW_neg = nn_dW_neg

    -- preallocate dW
    --local dW = torch.Tensor(sizes.n_filters,fs,fs)

    local conv_generative_grads = function(rbm,x,y,stat)
        local grads = {}
        assert(isRowVec(x)  and x:size(2) == sizes.n_visible and x:isSameSizeAs(stat.vkx_rnd))
        assert(isRowVec(stat.h0) and stat.h0:size(2) == sizes.n_hidden and stat.h0:isSameSizeAs(stat.hk))
 
        if rbm.toprbm then
            assert(isRowVec(y)  and y:size(2) == sizes.n_classes and y:isSameSizeAs(stat.vky_rnd))
            grads.dd = torch.add(y,  -stat.vky_rnd):t() 
            grads.dU = torch.mm(stat.h0:t(),y):add(-torch.mm(stat.hk:t(),stat.vky_rnd)) 

            assert(grads.dU:size(1) == sizes.n_hidden and grads.dU:size(2) == sizes.n_classes)
            assert(grads.dd:size(1) == sizes.n_classes and grads.dd:size(2) == 1)
        end
       
        -- shape h0,hk,x, and vkx in multidimensional shapes
        local h0_m, hk_m, x_m, vkx_m
        h0_m  = stat.h0:view(nf, sizes.hid_h, sizes.hid_h)
        hk_m  = stat.hk:view(nf, sizes.hid_h, sizes.hid_h)
        x_m   = x:view(1,ni,is,is)
        vkx_m = stat.vkx:view(1, ni, is, is)


        grads.dc = ( h0_m:sum(3):sum(2) - hk_m:sum(3):sum(2) ):view(sizes.n_filters,1) 
        grads.db = (x_m:sum(4):sum(3)   - vkx_m:sum(4):sum(3) ):view(sizes.n_input,1) 

        assert(ni == 1)   -- Current implementation possibly breaks with more than one input channel, refactor nn_dW_*** weights 
        local dw_pos,dw_neg,l
        grads.dW = torch.Tensor(sizes.n_filters,fs,fs)
        for l = 1,nf do
            nn_dW_pos.weight = h0_m[{l,{},{}}]:view(ni, sizes.hid_w*sizes.hid_h)
            nn_dW_neg.weight = hk_m[{l,{},{}}]:view(ni, sizes.hid_w*sizes.hid_h)

            dW_pos = nn_dW_pos:forward(x_m)--:clone()
            dW_neg = nn_dW_neg:forward(vkx_m)--:clone()
            grads.dW[{l,{},{}}] = dW_pos - dW_neg
            
        end
        
        -- normalize
        grads.db:mul(1/(is * is))
        grads.dc:mul(1/(sizes.hid_w*sizes.hid_h))
        grads.dW:mul(1/( (sizes.hid_h - 2 * fs + 2) *  (sizes.hid_w - 2 * fs + 2) ))

        -- Reshape gradients
        grads.db = grads.db:view(-1,1)  -- visbias
        grads.dc = grads.dc:view(-1,1)  -- hidbias
        grads.dW = conv.toFlat(grads.dW)

        assert(isRowVec(grads.dW) and grads.dW:size(2) == sizes.n_W)
        assert(grads.db:size(1) == sizes.n_input and grads.db:size(2) == 1)
        assert(grads.dc:size(1) == sizes.n_filters and grads.db:size(2) == 1)
        
        
        return grads
    end

    return conv_generative_grads,debug

end


conv.calcconvsizes = function(filter_size,n_filters,n_classes,input_size,pool_size,train)
    -- calculates sizes of layers etc in a conv RBM

    -- NUMBER OF HIDDEN UNITS
    -- The number of hidden units is the number of filters times 
    -- the size of the convolution after the rbmup.
    local sizes = {}
    sizes.input_size = input_size
    sizes.n_input    = train.data:size(2)
    sizes.n_hidden   = math.pow(input_size - filter_size +1,2)*n_filters
    sizes.n_visible  = train.data:size(3)*sizes.n_input 

    sizes.hid_w      = input_size - filter_size + 1 
    sizes.hid_h      = input_size - filter_size + 1 

    -- store W as a row vector
    sizes.n_W       = sizes.n_input * n_filters * math.pow(filter_size,2)
    sizes.n_U       = sizes.n_hidden * n_classes  -- for reference 
    
    -- b,c and d are column vectors
    sizes.n_b       = sizes.n_input -- bias of visible layer
    sizes.n_c       = n_filters -- bias of hidden layer
    sizes.n_d       = n_classes
    sizes.pad       = filter_size -1         -- Zero padding size
    sizes.filter_size = filter_size
    sizes.n_filters = n_filters
    sizes.n_classes = n_classes
    sizes.pool_size = pool_size

    -- hidden filter
    --sizes.h_filter_w = sizes.hid_h - filter_size+1 - filter_size + 1
    --sizes.h_filter_h = sizes.hid_w - filter_size+1 - filter_size + 1
    print(sizes)
    assert(sizes.hid_w % pool_size == 0)   -- otherwise maxpooling fails 
    return sizes
end

conv.setupsettings = function(opts,sizes)
    -- takes an opts struct and initialize W,U,b,c and d in correct dimensions
    -- W is drawn from  N(0,10^-6)
    -- U is drawn from uniform distribution see rbm-util / initcrbm
    -- b,c and d are initialized at zero
    -- 
    -- INPUT
    --    sizes  : output table from conv.calcConvSizes
    --
    -- RETURNS
    --     empty - modofies the supplied opts struct
    opts.W              = initcrbm(1,sizes.n_W,'gauss',-2)
    opts.U              = initcrbm(sizes.n_hidden,sizes.n_classes,'crbm')
    opts.b              = torch.zeros(sizes.n_input,1)
    opts.c              = torch.zeros(sizes.n_filters,1)
    opts.d              = torch.zeros(sizes.n_classes,1)
    opts.n_hidden       = sizes.n_hidden
    opts.n_visible      = sizes.n_visible
    opts.n_classes      = sizes.n_classes
    opts.precalctcwx    = 0   -- dont precalc when we use convNETS

end

function conv.setupfunctions(rbm,sizes,vistype,usemaxpool)
    if vistype  == nil then
        vistype = 'binary'
    end

    

    local conv_rbmup,conv_rbmdownx,conv_rbmdownxgauss,conv_pygivenx,conv_pygivenxdropout,debugupdown = conv.createupdownpygivenx(rbm,sizes,usemaxpool)
    local generativegrads,debuggen = conv.creategenerativegrads(sizes)  

    rbm.up = conv_rbmup
    rbm.downy = rbmdowny
    rbm.generativegrads = generativegrads
    rbm.generativestatistics = conv.generativestatistics
    rbm.pygivenx = conv_pygivenx
    rbm.pygivenxdropout = conv_pygivenxdropout

    if vistype == 'binary' then
        rbm.downx = conv_rbmdownx
        rbm.visxsampler = bernoullisampler -- bernoulli sampler
    elseif vistype == 'gauss' then
        rbm.downx = conv_rbmdownxgauss
        rbm.visxsampler = gausssampler
    else
        assert('false')
    end


    -- debug is reference to modelup and modeldownx
    local debug = {}
    debug.modelup = debugupdown.modelup
    debug.modeldownx = debugupdown.modeldownx
    debug.nn_dW_pos = debuggen.nn_dW_pos
    debug.nn_dW_neg = debuggen.nn_dW_neg
    return debug

end


function conv.maxPool(x,pool_size)
	-- maxpool over several filters
	-- INPUTS
    -- x         : should be a 3d matrix with dimensions 
    --             [n_filters x filter_size x filter_size]
    -- pool_size : size of maxpool 
    -- 
    -- RETURN
    --      New matrix of x:size() with pooled result + sum of each maxpool
    --
    -- SEE [1] H. Lee, R. Grosse, R. Ranganath, and A. Ng, “Convolutional deep 
    -- belief networks for scalable unsupervised learning of hierarchical 
    -- representations,” …  Mach. Learn., 2009.
	local function maxpool(x)
		--Calculate exp(x) / [sum(exp(x)) +1] in numerically stable way
		local m = torch.max(x)
		local exp_x = torch.exp(x - m)
		-- normalizer = sum(exp(x)) + 1   in scaled domain
		local normalizer = torch.exp(-m) + exp_x:sum()
		exp_x:cdiv( torch.Tensor(exp_x:nElement()):fill(normalizer) )


		return exp_x
	end

	local function maxpoollayer(h,h_pool_res,p_pool_res,pool_size)
		-- Performs probabilistic maxpooling.
		-- For each block of pool_size x pool_size calculate 
		-- exp(h_i) / [sum_i(exp(h_i)) + 1]
		-- h should be a 2d matrix
	    --print(hf:size())
        local height = h:size(1)
		local width =  h:size(2)
		--poshidprobs = torch.Tensor(height,width):typeAs(hf)
		-- notation h_(i,j)

        local i_pool,j_pool = 0,0
		local h_maxpool, p_maxpool
        --print("maxpoollayer: ", h_pool_res:size(),p_pool_res:size())
        for i_start = 1,height,pool_size do
            j_pool = 0
            
            i_pool = i_pool + 1
		    i_end = i_start+pool_size -1
			for j_start = 1,width,pool_size do   -- columns
		    	j_end = j_start+pool_size -1
                j_pool = j_pool + 1 

                h_maxpool = maxpool(h[{{i_start,i_end},{j_start,j_end}}])
                p_maxpool = h_maxpool:sum()

		    	h_pool_res[{{i_start,i_end},{j_start,j_end}}] = h_maxpool
                p_pool_res[{i_pool,j_pool}] = p_maxpool
		    end
		end   
	end

    local n_filter = x:size(1)
    local h_filter = x:size(2)
    local w_filter = x:size(3)
    h_maxpooled = torch.Tensor(n_filter,h_filter,w_filter):typeAs(x)
    p_maxpooled = torch.Tensor(n_filters,
                               h_filter / pool_size,
                               w_filter / pool_size):typeAs(x)
    

    for i = 1, x:size(1) do
    	maxpoollayer( x[{i,{},{}}], 
                       h_maxpooled[{ i,{},{} }], 
                       p_maxpooled[{ i,{},{} }],
                       pool_size )
    end
    return h_maxpooled,p_maxpooled
end

conv.flatToDownW = function(W,dest)
    -- Convert Weights from RBMUP NN to RBMDOWN format
    -- Converts between spatialConvolutionMM format and spatialConvolution 
    -- format. Furthermore each filter is INVERTED
    -- INPUTS
    --       W        : [1xn] row vector of weights
    --       dest     : mem reference where result is stored. The size of dest
    --                  is [n_input x n_filters x filter_size x filter_size]
    -- 
    -- RETURN
    --    empty, result is stored in dest
    --
    -- The dimensions of the returned 
    -- to the format used by spatialConvolution
    -- INVERTS weights in kernels
    function invertWeights(x)
        -- invert a mxn matrix
        xc = torch.Tensor(x:size())
        xc = xc:view(-1)    -- view as vector
        idx = xc:nElement()
        for i = 1,x:size(1) do --rows
            for j = 1,x:size(2) do  -- columns
                xc[idx] = x[{i,j}] 
                idx = idx-1
            end
        end
        xc = xc:viewAs(x)
        return xc
    end


    -- Change the weiw of the flat mtrix to correct format
    -- Wf is [n_input X n_filters X filter_size x filter_size] 
    local fs = dest:size(4)   --filter_size
    local ni = dest:size(1)   --n_input
    local nf = dest:size(2)   -- n_filters

 
    local Wf = conv.flatTo4D(W,dest)
    for input_dim = 1,ni do
        for filter_num = 1,nf do
        dest[{input_dim,filter_num,{},{}}] = invertWeights(Wf[{input_dim,filter_num,{},{}}])
        end
    end
end

conv.flatTo4D = function(x,sizeas)
    -- Convert x to sizeas. Used to convert rowvector x to spatialConv weights
    -- 
    -- INPUTS
    --    x       : [1xn] row vector of weights
    --    sizeas  : [n_input x n_filters x filter_size x filter_size] matrix
    -- 
    -- RETURNS
    --    X in same view as sizeas. No mem copy
    local fs = sizeas:size(4)   --filter_size
    local ni = sizeas:size(1)
    local s = torch.LongStorage({fs*fs,fs*fs*ni,fs,1})
    local sz= sizeas:size()
    return torch.Tensor():set(x:storage(), 1, sz,s)
end

conv.toFlat = function(x)
    -- Convert x to row vector (1xn)
    local sz = torch.LongStorage({1,x:nElement()})
    local s  = torch.LongStorage({x:nElement(),1})
    return torch.Tensor():set(x:storage(), 1, sz,s)
end

conv.flatTo2D = function(x,sizeas)
    -- Convert x to sizeas. Used to convert rowvector x to spatialConvMM weights
    -- 
    -- INPUTS
    --    x       : [1xn] row vector of weights
    --    sizeas  : [n_filters x (n_input*filter_size*filter_size)] matrix
    -- 
    -- RETURNS
    --    X in same view as sizeas. No mem copy
    local sz = sizeas:size()
    local s = sizeas:stride()
    return torch.Tensor():set(x:storage(), 1, sz,s)
end

conv.flatToUpW = function(W,dest)
    -- Converts to format used by spatialConvolutionMM
    --
    -- INPUTS
    --     W  : row vector of weights
    --   dest : mem reference to weights. Should be a
    --          [n_filters x (n_input*filter_size*filter_size)] matrix  
    -- 
    --  Returns
    --     empty, results are stored in dest
    -- 
    W2d = conv.flatTo2D(W,dest)
    dest:set(W2d:storage(),1,W2d:size(),W2d:stride())

end


conv.createupdownpygivenx = function(rbm,sizes,usemaxpool)
    local debug = {}
    local pad = filter_size - 1
    local modelup = nn.Sequential()
    modelup:add(nn.Reshape(sizes.n_input,sizes.input_size,sizes.input_size))
    modelup:add(nn.SpatialConvolutionMM(sizes.n_input,sizes.n_filters,
                sizes.filter_size,sizes.filter_size))
    --modelup:add(nn.Sigmoid())

    if  usemaxpool == nil then
        usemaxpool = true
    end


    local modeldownx = nn.Sequential()
    modeldownx:add(nn.Reshape(sizes.n_filters,sizes.hid_h,sizes.hid_w))
    modeldownx:add(nn.SpatialZeroPadding(sizes.pad, sizes.pad, sizes.pad, sizes.pad)) -- pad (filterwidth -1) 
    modeldownx:add(nn.SpatialConvolution(sizes.n_filters,sizes.n_input,
                                         sizes.filter_size,sizes.filter_size))
    --modeldownx:add(nn.Sigmoid())

    debug.modelup = modelup
    debug.modeldownx = modeldownx

    -- SET TESTING WEIGHTS OF UP MODEL
    conv.flatToUpW(rbm.W,modelup.modules[2].weight)            
    modelup.modules[2].bias   =  rbm.c:view(-1)-- torch.zeros(n_filters)    

    -- Test that the underlying storages are equal
    assert(rbm.W:storage() == modelup.modules[2].weight:storage())
    assert(rbm.c:storage() == modelup.modules[2].bias:storage())

    -- -- SET TESTING WEIGHTS OF DOWNX MODEL
    conv.flatToDownW(rbm.W,modeldownx.modules[3].weight)     
    modeldownx.modules[3].bias =  rbm.b:view(-1)--torch.zeros(n_input)

    assert(rbm.b:storage() == modeldownx.modules[3].bias:storage())
    -- The memory between weighs in modeldownx and rbm.W are not shared
    -- because of weight inversion
    --assert(rbm.W:storage() == modeldownx.modules[3].weight:storage())

    local pygivenx_conv = function(rbm,x,tcwx_pre_calc)
        assert(isRowVec(x))
        assert(x:size(1)*x:size(2) == rbm.n_visible)
        --print("pygivenx_conv:I could implement tcwx reuse with a bit of work...")
        local F,pyx, mask_expanded

        -- check shared memory and sizes
        assert(rbm.W:storage() == modelup.modules[2].weight:storage())
        assert(rbm.c:storage() == modelup.modules[2].bias:storage())

        tcwx =  tcwx_pre_calc or modelup:forward(x)
        tcwx = tcwx:view(1,-1)  -- to flat representation
        

        F = torch.add( rbm.U, torch.mm(tcwx:t(), rbm.one_by_classes) )
        pyx = softplus(F):sum(1)        -- p(y|x) logprob
        pyx:add(-torch.max(pyx))       -- subtract max for numerical stability
        pyx:exp()                      -- convert to real domain
        pyx:mul( ( 1/pyx:sum() ))      -- normalize probabilities

        assert(pyx:size(1) == 1 and pyx:size(2) == rbm.n_classes)
        return pyx,F
    end

    local pygivenxdropout_conv = function(rbm,x,tcwx_pre_calc)
        assert(isRowVec(x))
        assert(x:size(1)*x:size(2) == rbm.n_visible)

        --print("pygivenx_conv:I could implement tcwx reuse with a bit of work...")
        local tcwx,F,F_softplus,pyx, mask_expanded
        assert(tcwx == nil)
        
        mask_expanded = torch.mm(rbm.dropout_mask:t(), rbm.one_by_classes)

        tcwx =  tcwx_pre_calc or modelup:forward(x)
        tcwx = tcwx:view(1,-1)  -- to flat representation
        
        F   = torch.add( rbm.U, torch.mm(tcwx:t(), rbm.one_by_classes) )
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


    print("Maxpool in RBMUP: ", usemaxpool)
    local rbmup_conv = function(rbm,x,y,drop) 
        assert(isRowVec(x))
        assert(x:size(1)*x:size(2) == rbm.n_visible)
        
        local hid_act
        
        --- calculate contribution from features X
        
        -- Makes sure that the storages are the same
        -- I.e no need to update weights.
        assert(rbm.W:storage() == modelup.modules[2].weight:storage())
        assert(rbm.c:storage() == modelup.modules[2].bias:storage())
        hid_act = modelup:forward(x):clone()
        -- The output from modelup is [1 x n_filters X hidh X hidw]
        -- remove first dimension
        
        

        -- Max pool calculates hidden act using eq on top p. 4 Lee 2009
        if usemaxpool then
            hid_act = torch.view(hid_act, hid_act:size(2),hid_act:size(3),hid_act:size(4))
            hid_act,p_act =  conv.maxPool(hid_act,sizes.pool_size)
            rbm.act_up = p_act:view(1,-1)
        else
            rbm.act_up = hid_act:view(1,-1)
        end
        hid_act =  hid_act:view(1,-1) --flat view

        -- Calculate contribution from labels Y
        
        if rbm.toprbm then
            assert(isRowVec(y) and y:size(2) == rbm.n_classes)
            hid_act:add( torch.mm(y,rbm.U:t()) )
        end

        if drop == 1 then  
          hid_act:cmul(rbm.dropout_mask)
        end

        return hid_act
    end

    local rbmdownx_conv = function(rbm,hid_act)

        -- I Need to call flatToDownW because we use inverted weights
        assert(rbm.b:storage() == modeldownx.modules[3].bias:storage()) -- biases are shared
        conv.flatToDownW(rbm.W,modeldownx.modules[3].weight)   
        local  vis_act = modeldownx:forward(hid_act):clone()
        vis_act = vis_act:view(1,-1)   -- to flat view
        return sigm(vis_act)
    end

    local rbmdownxgauss_conv = function(rbm,hid_act)

        -- I Need to call flatToDownW because we use inverted weights
        assert(rbm.b:storage() == modeldownx.modules[3].bias:storage()) -- biases are shared
        conv.flatToDownW(rbm.W,modeldownx.modules[3].weight)   
        local  vis_act = modeldownx:forward(hid_act):clone()
        vis_act = vis_act:view(1,-1)   -- to flat view
        return vis_act
    end
    return rbmup_conv,rbmdownx_conv,rbmdownxgauss_conv,pygivenx_conv,pygivenxdropout_conv,debug
end


return conv