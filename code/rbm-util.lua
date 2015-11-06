-- setup,  printing functions, saving functions,


function rbmconvsetup(settings,train,convopts)
    -- handles the ugly setup for a convRBM
    local sizes = conv.calcconvsizes(settings.filter_size,settings.n_filters,
                    settings.n_classes,
                    settings.input_size,
                    settings.pool_size,
                    train)
    local opts = {}

    conv.setupsettings(opts,sizes)





    opts.toprbm = settings.toprbm
    -- merge useropts and functions opts. convopts will be
    -- overwritten with content in convopts_functions
    if convopts ~= nil then
        for k,v in pairs(opts) do
            if convopts[k] ~= nil then
                print("Overwriting settings: ", k,convopts[k], "with", v)
            end
            convopts[k] = v
        end
    else
        convopts = opts
    end

    local rbm = rbmsetup(convopts,train)
    local debug = conv.setupfunctions(rbm,sizes,settings.vistype,settings.usemaxpool)

    return rbm,opts,debug

end

function trainstackconvtorbm(convsettings,convopts,toprbmopts,train,val)
    -- settings.usemaxpool = true
    -- settings.vistype = 'binary'
    -- settings.filter_size = filter_size
    -- settings.n_filters = n_filters
    -- settings.n_classes = n_classes
    -- settings.input_size = input_size
    -- settings.pool_size = pool_size
    -- convops any settings that apply to normal rbm. Settings not used in
    -- conv or with inferred value will be overwritten
    -- rbmopts settings for toprbm

    print("WRITE SOME test where i assert that the correct values are overwritten etc")
    print(" and check that the bottom convrbm is a generative model without labels")
    print("Skip the ugly convsettings/convopts and just let the user specify a opts settings with the nessesary values")
    print("Maybe write a print function for conv rbms")
    convsettings.toprbm = false
    local convrbm,convopts_functions,convdebug  = setupconvrbm(settings,train,convopts)


    convrbm = rbmtrain(convrbm,train,val)

    local data2 = {}
    train2.labels = train.labels:clone()
    train2.data = rbmuppass(convrbm,train)

    local val2 = {}
    val2.labels = val.labels:clone()
    val2.data = rbmuppass(convrbm,val)

    toprbm = rbmsetup(toprbmopts,train2)
    toprbm = rbmtrain(train2,val2)

    return convrbm,toprbm
end

function trainstackrbmrbm()

end

function printrbm(rbm,xt,xv,xs)
     print("---------------------RBM------------------------------------")

     if xt then print(string.format("Number of trainig samples    :  %i",xt:size())) end
     if xv then print(string.format("Number of validation samples :  %i",xv:size())) end
     if xs then print(string.format("Number of semi-sup samples   :  %i",xs:size())) end

     local ttype
     if rbm.alpha == 1 then ttype = "GENERATIVE"
     elseif rbm.alpha == 0 then ttype = "DISCRIMINATIVE"
     elseif rbm.alpha > 0 and rbm.alpha < 1 then ttype = "HYBRID"
     else assert(false, "alpha must be numeric between 0 and 1") end

     if rbm.beta > 0 then ttype = ttype .. " + SEMISUP" end

     print(string.format("Training type                :  %s",ttype))
     print(string.format("Pretraining                  :  %s",rbm.pretrain))
     print(string.format("Top RBM                      :  %s",tostring(rbm.toprbm)))
     print(string.format("Number of visible            :  %i",rbm.n_visible))
     print(string.format("Number of hidden             :  %i",rbm.n_hidden))
     if rbm.toprbm then print(string.format("Number of classes            :  %i",rbm.n_classes)) end
     print("")
     print(string.format("Number of epocs              :  %i",rbm.numepochs))
     print(string.format("Current   epoc               :  %i",rbm.currentepoch))
     print(string.format("Learning rate                :  %f",rbm.learningrate))
     print(string.format("Momentum                     :  %f",rbm.momentum))
     print(string.format("alpha                        :  %f",rbm.alpha))
     print(string.format("beta                         :  %f",rbm.beta))
     print(string.format("batchsize                    :  %i",rbm.batchsize))
     print(string.format("Temp file                    :  %s",rbm.tempfile))
     print("")


     print("TRAINING TYPE")
     print(string.format("Type                         :  %s",rbm.traintype))
     print(string.format("Gibbs steps                  :  %i",rbm.cdn))
     if rbm.traintype == 'PCD' then print(string.format("Number of PCD chains         :  %i",rbm.npcdchains)) end

     print("")
     print("REGULARIZATON")
     print(string.format("Patience                     :  %i",rbm.patience))
     print(string.format("Sparisty                     :  %f",rbm.sparsity))
     print(string.format("L1                           :  %f",rbm.L1))
     print(string.format("L2                           :  %f",rbm.L2))
     print(string.format("DropOut                      :  %f",rbm.dropout))
     print("------------------------------------------------------------")

end

function initcrbm(m,n,inittype,std)
    -- Creates initial weights.
    -- If inittype is 'crbm'  then init weights from uniform distribution
    -- initilize weigts from uniform distribution. As described in
    -- Learning Algorithms for the Classification Restricted Boltzmann
    -- machine
    -- if inittype is 'gauss' init from N(0,std^2), std defualts to 10^-3
    -- If inittype is not specified use 'crbm'
    local weights
    if inittype == nil then
        inittype = 'crbm'
    end

    if std == nil then
     std = -2
    end

    if inittype == 'crbm' then
         local M,interval_max, interval_min
         M = math.max(m,n);
         interval_max = math.pow(M,-0.5);
         interval_min = -interval_max;
         weights = torch.rand(m,n):mul( interval_min + (interval_max-interval_min) )
    elseif inittype == 'gauss' then
         weights = torch.randn(m,n) * math.pow(10,std)

    else
     assert(false)  -- unknown inittype
    end
    return weights
end


function rbmsetup(opts,train,semisup)
     local rbm = {}

     rbm.progress = opts.progress or 1

     --assert(train.data:dim() == 3)
     -- if semisup then
     --    assert(semisup.data:dim() == 3)
     -- end

     -- or idiom does not work for booleans
     if opts.toprbm ~= nil then
        rbm.toprbm  = opts.toprbm
     else
        rbm.toprbm = true
     end
     local n_visible,n_samples,n_classes,n_input,n_hidden
     n_samples = train:size()
     local geometry = train:geometry()
     n_visible = geometry[1]*geometry[2]-- channels * channelwidth
     n_hidden  = opts.n_hidden or assert(false)




     rbm.boost = opts.boost or 'none'
     if opts.boost == 'diff' then
        rbm.yprobs = torch.Tensor(train:size(),#train:classnames())
     end

     rbm.batchsize = opts.batchsize or 1

     rbm.W = opts.W or initcrbm(n_hidden,n_visible)
     rbm.b = opts.b or torch.zeros(n_visible,1)
     rbm.c = opts.c or torch.zeros(n_hidden,1)

     rbm.vW = torch.zeros(rbm.W:size()):zero()
     rbm.vb = torch.zeros(rbm.b:size()):zero()
     rbm.vc = torch.zeros(rbm.c:size()):zero()

     rbm.dW = torch.Tensor(rbm.W:size()):zero()
     rbm.db = torch.Tensor(rbm.b:size()):zero()
     rbm.dc = torch.Tensor(rbm.c:size()):zero()
     rbm.rand  = function(m,n) return torch.rand(m,n) end
     rbm.n_visible       = n_visible
     rbm.n_samples       = n_samples
     rbm.n_hidden        = n_hidden
     rbm.errorfunction   = opts.errorfunction or function(conf) return 1-conf:accuracy() end

     rbm.hidden_by_one   = torch.ones(rbm.W:size(1),1)

     rbm.numepochs       = opts.numepochs or 1
     rbm.currentepoch    = 1
     rbm.learningrate    = opts.learningrate or 0.05
     rbm.momentum        = opts.momentum or 0
     rbm.traintype       = opts.traintype or 'CD'   -- CD or PCD
     rbm.cdn             = opts.cdn or 1
     rbm.npcdchains      = opts.npcdchains or 1

     -- OBJECTIVE
     rbm.alpha           = opts.alpha or 1
     rbm.beta            = opts.beta or 0

     -- REGULARIZATION
     rbm.dropout         = opts.dropout or 0
     rbm.L1              = opts.L1 or 0
     rbm.L2              = opts.L2 or 0
     rbm.sparsity        = opts.sparsity or 0
     rbm.patience        = opts.patience or 15


     rbm.pretrain = opts.pretrain or 'none'

     -- Set up and down functions + generative statistics functions
     -- see Deep boltzmann machines salakhutdinov 2009 sec 3.1
     -- pretraining also modifies downy
     if rbm.pretrain == 'none' then
        rbm.up = opts.up or rbmup
        rbm.downx = opts.downx or rbmdownx
        if rbm.toprbm then
            rbm.downy = opts.downy or rbmdowny
        end
     elseif rbm.pretrain == 'top' then
        -- we double the downweights
        rbm.up = rbmup
        rbm.downx = rbmdownxpretrain
        if rbm.toprbm then
            rbm.downy = rbmdownypretrain
        end
     elseif rbm.pretrain == 'bottom' then
        -- double up weights
        rbm.up = rbmuppretrain
        rbm.downx = rbmdownx
        if rbm.toprbm then
            rbm.downy = rbmdowny
        end
     else
        print('unknown pretrain options')
        error()
     end


     rbm.visxsampler = opts.visxsampler or bernoullisampler
     rbm.hidsampler = opts.hidsampler or bernoullisampler


     rbm.generativestatistics = opts.generativestatistics or grads.generativestatistics
     rbm.generativegrads      = opts.generativegrads or grads.generativegrads

     -- -
     rbm.tempfile        = opts.tempfile or "temp_rbm.asc"
     rbm.finalfile        = opts.finalfile or "final_rbm.asc"
     rbm.isgpu           = opts.isgpu or 0
     rbm.precalctcwx     = opts.precalctcwx or 1

     rbm.err_recon_train = torch.Tensor(rbm.numepochs):fill(-1)
     rbm.err_train       = torch.Tensor(rbm.numepochs):fill(-1)
     rbm.err_val         = torch.Tensor(rbm.numepochs):fill(-1)
     rbm.cur_err = torch.zeros(1)

     if rbm.traintype == 'PCD' then  -- init PCD chains
          rbm.chx = torch.Tensor(rbm.npcdchains,n_visible)
          if rbm.toprbm then
            rbm.chy =  torch.Tensor(rbm.npcdchains,#train:classnames())
          end

          for i = 1,rbm.npcdchains do
            local idx = math.floor(torch.uniform(1,train:size()+0.999999999))
            rbm.chx[{ i,{} }] = trainData[idx][1]:clone():view(1,-1)
            if rbm.toprbm then
                rbm.chy[{ i,{} }] =  trainData[idx][2]:clone():view(1,-1)
            end

          end


          if rbm.beta > 0 then
               local kk_semisup = torch.randperm(semisup.x:size(1))
               kk_semisup =  kk_semisup[{ {1, rbm.npcdchains} }]
               rbm.chx_semisup = x_semisup[{kk_semisup,{} }]:clone()
               if rbm.toprbm then
                    rbm.chy_semisup = y_semisup[{kk_semisup,{} }]:clone()
               end
          end
     elseif "CD" then
        --
     elseif "meanfield" then
         --
     else
        print("unknown traintype")
        error()
     end


     if rbm.toprbm then
         n_classes = #train:classnames()
         rbm.n_classes       = n_classes
         rbm.conf = ConfusionMatrix(train:classnames())
         rbm.d = opts.d or torch.zeros(n_classes,1)
         rbm.U = opts.U or initcrbm(n_hidden,n_classes)
         rbm.vU = torch.zeros(rbm.U:size()):zero()
         rbm.vd = torch.zeros(rbm.d:size()):zero()
         rbm.dU = torch.Tensor(rbm.U:size()):zero()
         rbm.dd = torch.Tensor(rbm.d:size()):zero()
         rbm.one_by_classes  = torch.ones(1,rbm.U:size(2))
         rbm.discriminativegrads     = opts.discriminativegrads or grads.discriminativegrads
         rbm.pygivenx  = opts.pygivenx or grads.pygivenx
         rbm.pygivenxdropout = opts.pygivenxdropout or grads.pygivenxdropout
     end

     if rbm.toprbm == false then
        assert(rbm.alpha == 1)  -- for non top rbms it does not make sense to discriminative training
     end

     return(rbm)
end

function checkequality(t1,t2,prec,pr)
     not_same_dim = not t1:isSameSizeAs(t2)

     if pr then
          print(t1)
          print(t2)
     end
     local prec = prec or -4

     local diff = t1 - t2
     err = diff:abs():max()
     numeric_err = (err > math.pow(10,prec) )
     if numeric_err then
        print('ASSERT: Numeric Error')
     elseif not_same_dim then
        print('ASSERT: Dimension Error')
     else
        print('Assert: Passed')
     end
     return (not not_same_dim) and (not numeric_err)

end

-- Stupid function to save an RBM in CSV...
-- Use loadrbm.m to load the RBM in matlab
function writerbmtocsv(rbm,folder)
     folder = folder or ''
     require('csvigo')
     function createtable(weight)
          local weighttable = {}
          for i = 1,weight:size(1) do   --rows
               local row = {}
               for j = 1,weight:size(2) do -- columns
                    row[j] = weight[{i,j}]
               end
          weighttable[i] = row
          end

          return weighttable

     end

     function readerr(err)
          e = {}
          for i = 1, err:size(1) do
               if err[i] ~= -1 then
                    e[i] = err[i]
               end
          end
          ret = {}
          ret[1] = e
          return(ret)
      end
    csvigo.save{data=createtable(rbm.stat_gen.hk), path=paths.concat(folder,'rbmhk.csv'),verbose = false}
    csvigo.save{data=createtable(rbm.stat_gen.vkx), path=paths.concat(folder,'rbmvkx.csv'),verbose = false}
    csvigo.save{data=createtable(rbm.stat_gen.vky), path=paths.concat(folder,'rbmvky.csv'),verbose = false}
    csvigo.save{data=createtable(rbm.stat_gen.h0), path=paths.concat(folder,'rbmh0.csv'),verbose = false}
     csvigo.save{data=createtable(rbm.W), path=paths.concat(folder,'rbmW.csv'),verbose = false}
     csvigo.save{data=createtable(rbm.dU), path=paths.concat(folder,'rbmdU.csv'),verbose = false}
     csvigo.save{data=createtable(rbm.dW), path=paths.concat(folder,'rbmdW.csv'),verbose = false}
     csvigo.save{data=createtable(rbm.U), path=paths.concat(folder,'rbmU.csv'),verbose = false}
     csvigo.save{data=createtable(rbm.b), path=paths.concat(folder,'rbmb.csv'),verbose = false}
     csvigo.save{data=createtable(rbm.c), path=paths.concat(folder,'rbmc.csv'),verbose = false}
     csvigo.save{data=createtable(rbm.d), path=paths.concat(folder,'rbmd.csv'),verbose = false}
     csvigo.save{data=readerr(rbm.err_val), path=paths.concat(folder,'rbmerr_val.csv'),verbose = false}
     csvigo.save{data=readerr(rbm.err_train), path=paths.concat(folder,'rbmerr_train.csv'),verbose = false}
     csvigo.save{data=readerr(rbm.err_recon_train), path=paths.concat(folder,'rbmerr_recon_train.csv'),verbose = false}
end


function writetensor(tensor,filename)
    -- writes tensor to csv file
     require('csvigo')
     function createtable(weight)
          local weighttable = {}
          for i = 1,weight:size(1) do   --rows
               local row = {}
               for j = 1,weight:size(2) do -- columns
                    row[j] = weight[{i,j}]
               end
          weighttable[i] = row
          end

          return weighttable

     end

     local tab = createtable(tensor)
    csvigo.save{data=tab, path=filename,verbose = false}
end



function isRowVec(x)
     -- checks if x is a vector is 1xn
     if x:dim() == 2  and  x:size(1) == 1 then
        res   = true
    else
        print ("isRowVector vec size: ",x:size() )
        res = false
    end
    return  res
end

function isVec(x)
     -- checks if x is a vector
     if x:dim() == 1 then
        res   = true
    else
        print ("isVec size: ",x:size() )
        res = false
    end
    return  res
end

function isMatrix(x)
     -- checks if x is a vector is mxn
     if x:dim() == 2 then
        res   = true
    else
        print ("IsMatrix vec size: ",x:size() )
        res = falses
    end
    return  res
end

