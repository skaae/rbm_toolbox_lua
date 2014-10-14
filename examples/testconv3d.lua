

codeFolder = '../code/'

require('torch')
require(codeFolder..'rbm')
conv = require(codeFolder..'conv-functions')

include(codeFolder..'mnist.lua')
require(codeFolder..'ProFi')
require 'paths'
torch.setdefaulttensortype('torch.FloatTensor')
require 'nn'

num_threads = 2
torch.setnumthreads(num_threads)
-- W  : vis - hid weights      [ #hid       x #vis ]
-- U  : label - hid weights    [ #hid       x #n_classes ]
-- b  : bias of visible layer  [ #vis       x 1]
-- c  : bias of hidden layer   [ #hid       x 1]
-- d  : bias of label layer    [ #n_classes x 1]

-- --n_input = 3                  -- color channels
-- filter_size = 2              --
-- n_input = 3
-- n_filters = 5                -- how many filters to use
-- pool_size =2                 -- how large the probabilistic maxpooling
-- input_size = 5               -- the size of the input image
-- n_classes = 4

-- -- create test data
-- -- X is always a row vector when it is stored.
-- x3d = torch.Tensor({
--     0.0500,    0.1000,    0.3500,    0.0500,         0,
--     0.1500,    0.2000,    0.1500,    0.2000,    0.0500,
--     0.1000,    0.1500,    0.2000,    0.0500,    0.1000,
--     0.2500,    0.2000,    0.1500,    0.0500,    0.0500,
--     0.1500,    0.1500,    0.1000,    0.0500,    0.1000,

--     0.0400,    0.0800,    0.1200,    0.1200,    0.1200,
--     0.1200,    0.1600,    0.1200,    0.1600,    0.0400,
--     0.0800,    0.1200,    0.1600,    0.0400,    0.0800,
--     0.2000,    0.1600,    0.1200,    0.0400,    0.0400,
--     0.1200,    0.1200,    0.0800,    0.0400,    0.0800,

--     0.0500,   -0.1000,    0.3500,    0.0500,         0,
--     0.1500,   -0.2000,    0.1500,    0.2000,    0.0500,
--     0.1000,    0.1500,    0.2000,    0.0500,    0.1000,
--     0.2500,    0.2000,    0.1500,    0.0500,    0.0500,
--     0.1500,    0.4000,    0.1000,    0.0500,    0.1000}):resize(1,n_input,math.pow(input_size,2))

-- x2d = x3d:view(1,75)

-- y = torch.Tensor({0,0,1, 0}):resize(1,n_classes)


-- -- Rows are 1 filters for 3 color channels
-- W = torch.Tensor({ 1,-2,-3,7,  1 ,2, 3,4,   -2 ,4,5 ,-2,
--                    2,1,-3,2 , -1,-4,2,1 ,   2 ,3,-4,2,
--                    2,1,-3,2 , -1,-4,2,1 ,   2 ,3,-4,2,
--                    2,1,-3,2 , -1,-4,2,1 ,   2 ,3,-4,2,
--                    2,1,-3,2 , -1,-4,2,1 ,   2 ,3,-4,2}):resize(1,60)  

-- labels = torch.Tensor{4}
-- train = {}
-- train.data = x3d--:view(1,1,75)
-- train.labels = labels 



-- sizes = conv.calcconvsizes(filter_size,n_filters,n_classes,input_size,pool_size,train)
-- --opts.n_hidden = 10
-- opts = {}
-- conv.setupsettings(opts,sizes)
-- opts.W = W   -- Testing weights
-- opts.U = torch.zeros(opts.U:size())  -- otherwise tests fails
-- opts.numepochs = 1
-- rbm = rbmsetup(opts,train)
-- debug_1 = conv.setupfunctions(rbm,sizes)  -- modofies RBM to use conv functions

-- opts_2 = {}
-- conv.setupsettings(opts_2,sizes)
-- opts_2.W = W:clone()   -- Testing weights
-- opts_2.U = torch.zeros(opts.U:size())  -- otherwise tests fails
-- opts.numepochs = 5
-- rbm_2 = rbmsetup(opts_2,train)
-- debug_2 = conv.setupfunctions(rbm_2,sizes)
-- --print(rbm)
-- --- Clone original weighs


-- W_org = rbm.W:clone()
-- U_org = rbm.U:clone()
-- b_org = rbm.b:clone()
-- c_org = rbm.c:clone()
-- d_org = rbm.d:clone()




-- -- Setup a trainng session with a single epoch and generative training
-- rbm.rand  = function(m,n) return torch.Tensor(m,n):fill(1):mul(0.53)end -- for testing
-- rbm.alpha = 1   -- generative training
-- rbm.momentum = 0


-- -- Test the values grads.calculate grads produce
-- grads.calculategrads(rbm,x2d,y) 
-- dc_cgrads = rbm.dc:clone()
-- db_cgrads = rbm.db:clone()
-- dW_cgrads = rbm.dW:clone()


-- print("#######################---EVALAUTE---#######################")

-- h0,h0_rnd,hk,vkx,vkx_rnd,vky_rnd = rbm.generativestatistics(rbm,x2d,y,tcwx)
-- dW, dU, db, dc, dd= rbm.generativegrads(x2d,y,h0,hk,vkx_rnd,vky_rnd)
-- rbm.generativegrads(x2d,y,h0,hk,vkx_rnd,vky_rnd)
--                updategradsandmomentum(rbm) 
--                --print(">>>>updategradsandmomentum rbm.db: ",rbm.db) 
--                -- update vW, vU, vb, vc and vd, formulae: vx = vX*mom + dX                    
--                --updateweights(rbm)

-- print(">>>>>>RBMTRAIN")

-- rbm = rbmtrain(rbm,train)
-- print("<<<<<<<<<<<<<<<")

-- print(">>>>>>RBMTRAIN MULTIPLE updates")
-- rbm_2.numepochs = 5
-- rbm_2 = rbmtrain(rbm_2,train,train)
-- print("<<<<<<<<<<<<<<<")

-- print("OK\n\n")
-- -- TESTING VALUES
-- h0_test = torch.Tensor({
--     0.2437,    0.0720,    0.4036,    0.1796,
--     0.0368,    0.6303,    0.2626,    0.1180,
--     0.3400,    0.2146,    0.2562,    0.2036,
--     0.1024,    0.3299,    0.2077,    0.2562,

--     0.0326,    0.7167,    0.4101,    0.0376,
--     0.0612,    0.0999,    0.1063,    0.3162,
--     0.1335,    0.1680,    0.1829,    0.2021,
--     0.4997,    0.0718,    0.2042,    0.2469,

--     0.0326,    0.7167,    0.4101,    0.0376,
--     0.0612,    0.0999,    0.1063,    0.3162,
--     0.1335,    0.1680,    0.1829,    0.2021,
--     0.4997,    0.0718,    0.2042,    0.2469,

--     0.0326,    0.7167,    0.4101,    0.0376,
--     0.0612,    0.0999,    0.1063,    0.3162,
--     0.1335,    0.1680,    0.1829,    0.2021,
--     0.4997,    0.0718,    0.2042,    0.2469,

--     0.0326,    0.7167,    0.4101,    0.0376,
--     0.0612,    0.0999,    0.1063,    0.3162,
--     0.1335,    0.1680,    0.1829,    0.2021,
--     0.4997,    0.0718,    0.2042,    0.2469
--     }):resize(1,5*4*4) 
-- h0_rnd_test = torch.Tensor({
--      1     1     1     1     1     1
--      1     1     1     1     1     0
--      1     1     1     1     1     1
--      1     0     0     1     1     0
--      1     0     0     1     1     1
--      1     0     1     0     0     0

--      0     0     0     1     1     0
--      1     0     0     1     1     0
--      1     0     0     0     1     0
--      0     1     1     1     0     0
--      1     0     0     0     1     0
--      1     0     0     1     0     0

--      1     1     1     0     0     1
--      0     0     0     0     0     0
--      0     0     0     1     0     1
--      1     1     0     0     0     1
--      0     1     1     0     0     1
--      1     0     1     1     1     1

--      0     0     0     0     0     1
--      1     1     1     1     1     1
--      0     1     1     1     1     1
--      1     1     0     0     1     1
--      1     1     1     1     1     1
--      1     0     0     1     0     0

--      1     0     1     1     0     0
--      0     0     0     0     0     1
--      0     0     0     0     0     1
--      0     1     1     1     0     1
--      0     1     0     1     0     0
--      0     1     1     0     1     1}):resize(1,5*6*6) 
-- vkx_rnd_test = torch.Tensor({
--     0.5046,    5.4483,    6.4071,    1.3132,   -0.2088,
--    -0.5959,   -5.2479,    0.3576,    7.6641,    2.5863,
--     0.5627,   -0.9298,    5.1102,    0.5802,    3.7569,
--     1.4782,    3.4863,    1.3514,    2.8530,    3.5173,
--    -6.3037,    2.8630,   -0.1891,   -0.6443,    3.7686,

--     0.1133,   -2.8290,  -12.5596,   -5.7247,   -0.2421,
--     0.7840,    6.3792,    7.1448,    1.7716,   -3.9546,
--     0.4063,    1.1692,    1.8246,    1.3398,   -1.0906,
--     0.1914,   -3.8660,    2.6643,    0.4013,   -1.8153,
--     4.3049,    3.9727,    3.8634,    4.3912,    2.0124,

--    -0.2266,    6.9558,   11.3613,    6.4770,    1.1691,
--     1.1131,  -10.9126,    5.0921,    7.3895,    4.2077,
--    -0.4076,    5.8450,    2.9765,    0.2859,    5.5336,
--     3.3570,    5.0936,    2.6688,    3.4780,    5.1977,
--    -7.4833,    4.2936,   -2.3139,   -1.4514,    1.4628
--     }):resize(1,n_input*5*5)
--  vkx_sigm_rnd_test = torch.Tensor({
--      1,     1,     1,     1,     0,
--      0,     0,     0,     1,     1,
--      1,     1,     1,     1,     1,
--      0,     1,     1,     1,     1,
--      0,     1,     0,     0,     1,

--      1,     0,     0,     0,     1,
--      1,     1,     1,     1,     0,
--      0,     0,     0,     0,     0,
--      1,     1,     1,     1,     0,
--      1,     1,     1,     1,     1,

--      0,     1,     1,     1,     1,
--      1,     0,     0,     1,     1,
--      1,     1,     1,     1,     1,
--      0,     1,     1,     1,     1,
--      0,     1,     0,     0,     1}):resize(1,n_input*5*5)
-- vkx_sigm_test = torch.Tensor({
--     0.7311,    0.9975,    1.0000,    0.9526,    0.1192,
--     0.0474,    0.0180,    0.0003,    1.0000,    1.0000,
--     0.9999,    0.9997,    1.0000,    0.9975,    1.0000,
--     0.0009,    0.9933,    0.9991,    1.0000,    1.0000,
--     0.0000,    0.9933,    0.0003,    0.5000,    1.0000,

--     0.7311,    0.1192,    0.0000,    0.0000,    0.8808,
--     0.9526,    1.0000,    1.0000,    0.9999,    0.0000,
--     0.0474,    0.0000,    0.0000,    0.0067,    0.0000,
--     0.9991,    0.9820,    1.0000,    0.8808,    0.0025,
--     0.9997,    0.9991,    1.0000,    1.0000,    0.9997,

--     0.1192,    1.0000,    1.0000,    1.0000,    0.9820,
--     0.9933,    0.0000,    0.2689,    1.0000,    1.0000,
--     0.9975,    1.0000,    1.0000,    0.9820,    1.0000,
--     0.0474,    0.9933,    0.9933,    1.0000,    1.0000,
--     0.0000,    1.0000,    0.0000,    0.0067,    0.9975}):resize(1,n_input*5*5)
-- hk_sigm_test = torch.Tensor({
--     0.7304,    0.5001,    0.5285,    0.6989,
--     0.5001,    0.5006,    0.5105,    0.5005,
--     0.5039,    0.5288,    0.6708,    0.5033,
--     0.7013,    0.5039,    0.5033,    0.5651,

--     0.5002,    0.7309,    0.7311,    0.5000,
--     0.5000,    0.5000,    0.5000,    0.5000,
--     0.7309,    0.5002,    0.5006,    0.5002,
--     0.5000,    0.5000,    0.5001,    0.7303,

--     0.5002,    0.7309,    0.7311,    0.5000,
--     0.5000,    0.5000,    0.5000,    0.5000,
--     0.7309,    0.5002,    0.5006,    0.5002,
--     0.5000,    0.5000,    0.5001,    0.7303,

--     0.5002,    0.7309,    0.7311,    0.5000,
--     0.5000,    0.5000,    0.5000,    0.5000,
--     0.7309,    0.5002,    0.5006,    0.5002,
--     0.5000,    0.5000,    0.5001,    0.7303,

--     0.5002,    0.7309,    0.7311,    0.5000,
--     0.5000,    0.5000,    0.5000,    0.5000,
--     0.7309,    0.5002,    0.5006,    0.5002,
--     0.5000,    0.5000,    0.5001,    0.7303}):resize(1,5*4*4)
-- h0_filter_test = torch.Tensor({
--     0.6303,    0.2626,
--     0.2146,    0.2562,

--     0.0999,    0.1063,
--     0.1680,    0.1829,

--     0.0999,    0.1063,
--     0.1680,    0.1829,

--     0.0999,    0.1063,
--     0.1680,    0.1829,

--     0.0999,    0.1063,
--     0.1680,    0.1829}):resize(n_filters,1,2,2)
-- h0_filter_test = sigm(h0_filter_test)
-- x_filter_test = torch.Tensor({
--     0.2000,    0.1500,    0.2000,
--     0.1500,    0.2000,    0.0500,
--     0.2000,    0.1500,    0.0500,

--     0.1600,    0.1200,    0.1600,
--     0.1200,    0.1600,    0.0400,
--     0.1600,    0.1200,    0.0400,

--    -0.2000,    0.1500,    0.2000,
--     0.1500,    0.2000,    0.0500,
--     0.2000,    0.1500,    0.0500}):resize(n_input,3,3)
-- x_filter_test = sigm(x_filter_test)
-- db_test =  torch.Tensor({-0.5540,-0.4976,-0.6080}):resize(3,1)
-- dc_test = torch.Tensor({ 0.0003,-0.0041,-0.0041,-0.0041,-0.0041}):resize(5,1)
-- dW_test = torch.Tensor({
--    -0.1972,   -0.3401,
--    -0.4511,   -0.4852,

--    -0.1706,   -0.1828,
--    -0.2187,   -0.2459,

--    -0.2624,   -0.3401,
--    -0.4511,   -0.4852,

--    -0.1566,   -0.2953,
--    -0.4066,   -0.4402,

--    -0.1751,   -0.1861,
--    -0.1754,   -0.2022,

--    -0.2091,   -0.2953,
--    -0.4066,   -0.4402,

--    -0.1566,   -0.2953,
--    -0.4066,   -0.4402,

--    -0.1751,   -0.1861,
--    -0.1754,   -0.2022,

--    -0.2091,   -0.2953,
--    -0.4066,   -0.4402,

--    -0.1566,   -0.2953,
--    -0.4066,   -0.4402,

--    -0.1751,   -0.1861,
--    -0.1754,   -0.2022,

--    -0.2091,   -0.2953,
--    -0.4066,   -0.4402,

--    -0.1566,   -0.2953,
--    -0.4066,   -0.4402,

--    -0.1751,   -0.1861,
--    -0.1754,   -0.2022,

--    -0.2091,   -0.2953,
--    -0.4066,   -0.4402,
--     }):resize(1,60)


-- -- Calculate expected values before update
-- --print(W)
-- W_train_test = W_org + dW_test * rbm.learningrate 
-- b_train_test = b_org + db_test * rbm.learningrate 
-- c_train_test = c_org + dc_test * rbm.learningrate 
-- -- Setup check after RBMtrain
-- -- dd and dU are not tested 




-- print "################ TESTS ############################" 
-- -- The debug table has references to the up/down/unroll networks

-- -- a) Assert that rbm.W and b and modelup.weights share memeory

-- print('Test network memory sharing')
-- checkequality(conv.toFlat(rbm.W),conv.toFlat(debug_1.modelup.modules[2].weight),-4,false)
-- assert(rbm.W:storage() == debug_1.modelup.modules[2].weight:storage())
-- assert(rbm.c:storage() == debug_1.modelup.modules[2].bias:storage())

-- -- b) for modeldownx the bias should be shared
-- assert(rbm.b:storage() == debug_1.modeldownx.modules[3].bias:storage())

-- -- ADD MORE TESTS WITH NETWORKS
-- -- E.g. test conversion functions
-- print("OK")


-- print("Testing Statistics...")
-- assert(checkequality(h0,h0_test,-4,false))

-- assert(checkequality(h0_rnd,h0_rnd_test,-4,false))
-- assert(checkequality(vkx,vkx_sigm_test,-4,false))
-- assert(checkequality(vkx_rnd,vkx_sigm_rnd_test,-4,false))
-- assert(checkequality(hk,hk_sigm_test,-4,false))
-- print("OK")

-- print("Testing Gradients...")
-- assert(checkequality(db,db_test,-4,false))
-- assert(checkequality(dc,dc_test,-4,false))
-- assert(checkequality(dW,dW_test,-4,false))

-- assert(checkequality(dc_cgrads,dc_test,-4,false))
-- assert(checkequality(db_cgrads,db_test,-4,false))
-- assert(checkequality(dW_cgrads,dW_test,-4,false))

-- assert(checkequality(rbm.c,c_train_test,-4,false))
-- assert(checkequality(rbm.b,b_train_test,-4,false))
-- assert(checkequality(rbm.W,W_train_test,-4,false))

-- print('OK')


print("################--Check against MEDAL----###############")
n_classes = 4
n_input = 1
input_size = 8
n_filters = 5
filter_size = 3
pool_size = 2

y = torch.Tensor({0,0,1, 0}):resize(1,n_classes)

x3d = torch.Tensor({
    0.0118,    0.0706,    0.0706,    0.0706,    0.4941,    0.5333,    0.6863,    0.1020,    
    0.6667,    0.9922,    0.9922,    0.9922,    0.9922,    0.9922,    0.8824,    0.6745,    
    0.9922,    0.9922,    0.9922,    0.9922,    0.9922,    0.9843,    0.3647,    0.3216,    
    0.9922,    0.9922,    0.7765,    0.7137,    0.9686,    0.9451,         0,         0,         
    0.9922,    0.8039,    0.0431,         0,    0.1686,    0.6039,         0,         0,        
    0.9922,    0.3529,         0,         0,         0,         0,         0,         0,         
    0.9922,    0.7451,    0.0078,         0,         0,         0,         0,         0,         
    0.7451,    0.9922,    0.2745,         0,         0,         0,         0,         0       
  }):resize(1,n_input,input_size*input_size)


x2d = x3d:view(1,input_size*input_size)
W = torch.Tensor({
   -0.0184,   -0.0439,   -0.0697,
    0.0490,   -0.0785,   -0.0343,
   -0.1111,   -0.0906,   -0.0229,

    0.0086,   -0.0657,    0.0379,
   -0.0180,    0.0840,   -0.0184,
    0.0412,   -0.1050,    0.0130,

   -0.0799,    0.1041,    0.0836,
   -0.0671,   -0.0415,    0.0877,
    0.0668,    0.0427,   -0.0922,

   -0.1024,   -0.0893,    0.0074,
   -0.0734,   -0.0175,    0.0426,
    0.0840,    0.1018,   -0.0410,

    0.0414,    0.0556,   -0.0488,
    0.0744,    0.1086,    0.0643,
   -0.1070,    0.0551,   -0.0882
}):resize(1,45)

labels_medal = torch.Tensor{4}
train_medal = {}
train_medal.data = x3d--:view(1,1,81)
train_medal.labels = labels_medal 

sizes_medal = conv.calcconvsizes(filter_size,n_filters,n_classes,input_size,pool_size,train_medal)
--opts.n_hidden = 10
opts_medal = {}
conv.setupsettings(opts_medal,sizes_medal)
opts_medal.W = W   -- Testing weights
opts_medal.U = torch.zeros(opts_medal.U:size())  -- otherwise tests fails
opts_medal.numepochs = 1
rbm_medal = rbmsetup(opts_medal,train_medal)
debug_3 = conv.setupfunctions(rbm_medal,sizes_medal)  -- modofies RBM to use conv functions


W_org = rbm_medal.W:clone()
U_org = rbm_medal.U:clone()
b_org = rbm_medal.b:clone()
c_org = rbm_medal.c:clone()
d_org = rbm_medal.d:clone()


print("------> TEST MEDAL STATISTICS <-----------------")
conv_rbmup_m,conv_rbmdownx_m,conv_pygivenx_m,conv_pygivenxdropout_m,debugupdown_m = conv.createupdownpygivenx(rbm_medal,sizes_medal)

rbm_medal.rand  = function(m,n) return torch.Tensor(m,n):fill(1):mul(0.2)end

stat_gen = rbm_medal.generativestatistics(rbm_medal,x2d,y,tcwx)
grads_gen= rbm_medal.generativegrads(rbm_medal,x2d,y,stat_gen)

rbm_medal = rbmtrain(rbm_medal,train_medal,train_medal)


h0_test = torch.Tensor({
    0.1919,    0.1948,    0.1899,    0.1859,    0.1761,    0.1960,
    0.1744,    0.1770,    0.1820,    0.1793,    0.1744,    0.2041,
    0.1675,    0.1869,    0.1893,    0.1815,    0.1775,    0.1981,
    0.1845,    0.2184,    0.2039,    0.1961,    0.1939,    0.2166,
    0.1824,    0.2079,    0.2010,    0.1939,    0.1957,    0.1994,
    0.1807,    0.2032,    0.1977,    0.2037,    0.2016,    0.2016,

    0.2015,    0.2005,    0.2038,    0.1985,    0.1974,    0.2028,
    0.1959,    0.2007,    0.2010,    0.1952,    0.1952,    0.2045,
    0.1960,    0.2070,    0.2022,    0.2040,    0.1943,    0.2025,
    0.2032,    0.1958,    0.1982,    0.1959,    0.2004,    0.2011,
    0.1897,    0.2087,    0.2002,    0.2012,    0.1938,    0.2023,
    0.1951,    0.2035,    0.2010,    0.1988,    0.2013,    0.2013,

    0.1937,    0.1886,    0.1961,    0.2057,    0.2087,    0.1941,
    0.2181,    0.2118,    0.2040,    0.2057,    0.2108,    0.2008,
    0.2174,    0.2094,    0.2022,    0.1934,    0.2045,    0.1997,
    0.1987,    0.1900,    0.2038,    0.2172,    0.2041,    0.1845,
    0.2082,    0.1965,    0.1983,    0.2099,    0.2100,    0.1904,
    0.1890,    0.2031,    0.1996,    0.1961,    0.1999,    0.1999,

    0.2208,    0.2143,    0.2219,    0.2138,    0.2162,    0.1975,
    0.1886,    0.1789,    0.1784,    0.1822,    0.1950,    0.1799,
    0.2079,    0.1899,    0.1849,    0.1840,    0.1879,    0.1974,
    0.1931,    0.1845,    0.2008,    0.2012,    0.1865,    0.1991,
    0.1890,    0.1976,    0.1991,    0.1975,    0.1912,    0.1930,
    0.1942,    0.2117,    0.2042,    0.1996,    0.2053,    0.2053,

    0.1955,    0.2008,    0.1954,    0.1998,    0.2090,    0.2057,
    0.2111,    0.2126,    0.2106,    0.2154,    0.2164,    0.1927,
    0.2193,    0.2096,    0.2222,    0.2216,    0.2321,    0.1868,
    0.2028,    0.1911,    0.1842,    0.1948,    0.2106,    0.1931,
    0.2156,    0.1860,    0.2008,    0.1982,    0.2055,    0.2024,
    0.2236,    0.1855,    0.1965,    0.2022,    0.1974,    0.1974}):resize(1,5*6*6)


h0_rnd_test = torch.Tensor({
     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     1,
     0,     0,     0,     0,     0,     0,
     0,     1,     1,     0,     0,     1,
     0,     1,     1,     0,     0,     0,
     0,     1,     0,     1,     1,     1,

     1,     1,     1,     0,     0,     1,
     0,     1,     1,     0,     0,     1,
     0,     1,     1,     1,     0,     1,
     1,     0,     0,     0,     1,     1,
     0,     1,     1,     1,     0,     1,
     0,     1,     1,     0,     1,     1,

     0,     0,     0,     1,     1,     0,
     1,     1,     1,     1,     1,     1,
     1,     1,     1,     0,     1,     0,
     0,     0,     1,     1,     1,     0,
     1,     0,     0,     1,     1,     0,
     0,     1,     0,     0,     0,     0,

     1,     1,     1,     1,     1,     0,
     0,     0,     0,     0,     0,     0,
     1,     0,     0,     0,     0,     0,
     0,     0,     1,     1,     0,     0,
     0,     0,     0,     0,     0,     0,
     0,     1,     1,     0,     1,     1,

     0,     1,     0,     0,     1,     1,
     1,     1,     1,     1,     1,     0,
     1,     1,     1,     1,     1,     0,
     1,     0,     0,     0,     1,     0,
     1,     0,     1,     0,     1,     1,
     1,     0,     0,     1,     0,     0}):resize(1,5*6*6)

vkx_test = torch.Tensor({
    0.4766,    0.4483,    0.4631,    0.4156,    0.4798,    0.5527,    0.5080,    0.4973,
    0.4676,    0.5447,    0.5515,    0.5356,    0.5232,    0.5845,    0.6026,    0.5244,
    0.4979,    0.5279,    0.6336,    0.6318,    0.5969,    0.5844,    0.4536,    0.4994,
    0.4859,    0.5273,    0.4733,    0.4614,    0.5572,    0.5460,    0.4665,    0.4620,
    0.5154,    0.6438,    0.4025,    0.3296,    0.4227,    0.6698,    0.4696,    0.4874,
    0.4957,    0.4503,    0.4749,    0.5213,    0.4989,    0.3901,    0.4399,    0.5029,
    0.5085,    0.5067,    0.3467,    0.5560,    0.4530,    0.4821,    0.4372,    0.4787,
    0.4733,    0.5339,    0.4965,    0.4099,    0.4877,    0.4524,    0.4639,    0.4873
 }):resize(1,1*8*8)   

hk_test = torch.Tensor({
    0.1916,    0.1905,    0.1891,    0.1881,    0.1878,    0.1912,
    0.1906,    0.1893,    0.1926,    0.1911,    0.1893,    0.1931,
    0.1866,    0.1887,    0.1931,    0.1909,    0.1892,    0.1890,
    0.1912,    0.1974,    0.1926,    0.1882,    0.1897,    0.1969,
    0.1905,    0.1939,    0.1941,    0.1889,    0.1919,    0.1902,
    0.1910,    0.1911,    0.1890,    0.1944,    0.1922,    0.1925,

    0.2004,    0.1977,    0.1991,    0.1990,    0.1985,    0.2022,
    0.1986,    0.2016,    0.2012,    0.1991,    0.1996,    0.1982,
    0.1973,    0.2009,    0.2009,    0.2011,    0.1944,    0.2014,
    0.2023,    0.1980,    0.1979,    0.1980,    0.2039,    0.1988,
    0.1963,    0.2034,    0.1981,    0.2025,    0.1955,    0.2009,
    0.2002,    0.1978,    0.2019,    0.1965,    0.2017,    0.1999,

    0.1989,    0.1989,    0.2012,    0.2059,    0.2059,    0.1990,
    0.2066,    0.2042,    0.1995,    0.2015,    0.2027,    0.2020,
    0.2069,    0.2094,    0.2023,    0.1958,    0.2027,    0.2030,
    0.1984,    0.1943,    0.2015,    0.2108,    0.2028,    0.1977,
    0.2069,    0.1946,    0.1989,    0.2053,    0.2068,    0.2002,
    0.1984,    0.2077,    0.2038,    0.1995,    0.1987,    0.2024,

    0.1988,    0.2004,    0.2031,    0.2032,    0.2026,    0.1970,
    0.1988,    0.1960,    0.1925,    0.1950,    0.1974,    0.1950,
    0.2029,    0.1977,    0.1918,    0.1909,    0.1980,    0.2010,
    0.1966,    0.1945,    0.2028,    0.2054,    0.1970,    0.1945,
    0.1971,    0.1947,    0.2013,    0.2011,    0.1963,    0.1960,
    0.1984,    0.2023,    0.1971,    0.1944,    0.1997,    0.2007,

    0.2001,    0.2022,    0.1993,    0.1991,    0.2047,    0.2032,
    0.2045,    0.2069,    0.2076,    0.2073,    0.2038,    0.2023,
    0.2047,    0.2013,    0.2062,    0.2059,    0.2087,    0.1968,
    0.2053,    0.2012,    0.1969,    0.2027,    0.2042,    0.2046,
    0.2073,    0.2013,    0.2064,    0.1971,    0.2046,    0.2026,
    0.2024,    0.2008,    0.2012,    0.2071,    0.2021,    0.2016
  }):resize(1,5*6*6)



dW_test =  torch.Tensor({

    0.6897,    0.1193,   -0.4294,
    0.7322,   -0.1370,   -0.7868,
    0.1229,   -0.8993,   -1.6596,

    0.7539,    0.2024,   -0.3725,
    0.8539,   -0.0078,   -0.7207,
    0.2052,   -0.8324,   -1.6701,

    0.7792,    0.2744,   -0.3199,
    0.8652,    0.0210,   -0.7132,
    0.2138,   -0.8103,   -1.6942,

    0.6137,    0.0841,   -0.4521,
    0.8015,   -0.0256,   -0.7237,
    0.2071,   -0.7876,   -1.6409,

    0.8560,    0.3313,   -0.3115,
    0.9545,    0.1280,   -0.6752,
    0.2382,   -0.7432,   -1.6661

  }):mul(1/( (6 - 2 * 3 + 2) *  (6 - 2 * 3 + 2) )):resize(1,5*3*3)

db_test = torch.Tensor({-2.9119}):mul(1/(input_size * input_size)):resize(1,1)
dc_test = torch.Tensor({0.0317,0.0099,-0.0115,-0.0395,0.0304}):mul(1/(6*6)):resize(5,1)

W_train_test = W_org + dW_test * rbm_medal.learningrate 
b_train_test = b_org + db_test * rbm_medal.learningrate 
c_train_test = c_org + dc_test * rbm_medal.learningrate 


assert(checkequality(stat_gen.h0,h0_test,-4,false))
assert(checkequality(stat_gen.h0_rnd,h0_rnd_test,-4,false))
assert(checkequality(stat_gen.vkx,vkx_test,-4,false))
assert(checkequality(stat_gen.hk,hk_test,-4,false))

assert(checkequality(grads_gen.dW,dW_test*rbm_medal.learningrate,-4,false))
assert(checkequality(grads_gen.db,db_test*rbm_medal.learningrate,-4,false))
assert(checkequality(grads_gen.dc,dc_test*rbm_medal.learningrate,-4,false))

assert(checkequality(rbm_medal.c,c_train_test,-3,false))
assert(checkequality(rbm_medal.b,b_train_test,-3,false))
assert(checkequality(rbm_medal.W,W_train_test,-3,false))


print("TODO")
print("FIGURE OUT WHY THEY USE ")


rbm_medal.numepochs = 5
rbm_medal = rbmtrain(rbm_medal,train_medal,train_medal)




rbm_medal.toprbm = false
rbm_medal.currentepoch = 1
rbm_medal.U = nil 
rbm_medal.dU = nil 
rbm_medal.d = nil 
rbm_medal.dd = nil 
rbm_medal.vd = nil 
rbm_medal.vU = nil
rbm_medal = rbmtrain(rbm_medal,train_medal,train_medal)


--uppass2 = rbmuppass(rbm_medal,train_medal)



-- TEST UPPASS FUNCTION

-- with usemaxpool uppass should be a factor of max_pool^2 smaller than the
-- number of hidden units


settings = {}
settings.usemaxpool = true
settings.vistype = 'binary'


settings.filter_size = filter_size
settings.n_filters = n_filters
settings.n_classes = n_classes
settings.input_size = input_size
settings.pool_size = pool_size
settings.toprbm = true



rbm,opts,debug__ = setupconvrbm(settings,train_medal)
rbm = rbmtrain(rbm,train_medal,train_medal)


uppass1 = rbmuppass(rbm,train_medal)
assert(uppass1:size(1) == train_medal.data:size(1))
assert(uppass1:size(3) == (  rbm.n_hidden / math.pow(pool_size,2))  )

settings.usemaxpool = false
rbm,opts,debug__ = setupconvrbm(settings,train_medal)
rbm = rbmtrain(rbm,train_medal,train_medal)

uppass2 = rbmuppass(rbm,train_medal)
assert(uppass2:size(1) == train_medal.data:size(1))
assert(uppass2:size(3) == rbm.n_hidden  )


settings.toprbm = false
settings.usemaxpool = true
rbm,opts,debug__ = setupconvrbm(settings,train_medal)
rbm = rbmtrain(rbm,train_medal,train_medal)
uppass1 = rbmuppass(rbm,train_medal)
assert(rbm.U == nil and rbm.dU == nil and rbm.vU == nil)
assert(rbm.d == nil and rbm.dd == nil and rbm.vd == nil)
assert(uppass1:size(1) == train_medal.data:size(1))
assert(uppass1:size(3) == (  rbm.n_hidden / math.pow(pool_size,2))  )


settings.toprbm = false
settings.usemaxpool = false

testopts = {}
testopts.n_hidden = 1001231

rbm,opts,debug__ = setupconvrbm(settings,train_medal,testopts)
rbm = rbmtrain(rbm,train_medal,train_medal)
uppass2 = rbmuppass(rbm,train_medal)
assert(rbm.U == nil and rbm.dU == nil and rbm.vU == nil)
assert(rbm.d == nil and rbm.dd == nil and rbm.vd == nil)
assert(uppass2:size(1) == train_medal.data:size(1))
assert(uppass2:size(3) == rbm.n_hidden  )



print("TEST STACKING")
 trainstackconvtorbm(convsettings,convopts,toprbmopts,train,val)
-- settings.usemaxpool = false
-- rbm,opts,debug__ = setupconvrbm(settings,train_medal)
-- rbm = rbmtrain(rbm,train_medal,train_medal)



-- print("\n\n#########################TRAINING ON SMALL MNIST##############")
-- torch.manualSeed(123)
-- train_mnist,val_mnist,test_mnist  = mnist.loadMnist(1000,false)
-- print("TRAINING DATA:\n", train_mnist)
-- filter_size =11            --

-- n_filters = 40              -- how many filters to use
-- pool_size =2                -- how large the probabilistic maxpooling
-- input_size = 28               -- the size of the input image
-- n_classes = 10
-- n_input = 1

-- sizes_mnist = conv.calcconvsizes(filter_size,n_filters,n_classes,input_size,pool_size,train_mnist)
-- print(sizes_mnist)
-- --opts.n_hidden = 10
-- opts_mnist = {}
-- conv.setupsettings(opts_mnist,sizes_mnist)
-- --print(opts_mnist)
-- --opts.W = W   -- Testing weights
-- --opts.U = torch.zeros(opts.U:size())  -- otherwise tests fails
-- opts_mnist.numepochs = 100
-- rbm_mnist = rbmsetup(opts_mnist,train_mnist)
-- vistype = 'binary'  -- | 'gauss'
-- debug_mnist = conv.setupfunctions(rbm_mnist,sizes_mnist,vistype)  -- modofies RBM to use conv functions

-- --gfx.image(train_mnist.data[{1,5,{}}]:resize(28,28), {zoom=2, legend=''})
-- rbm_mnist.learningrate = 0.001
-- rbm_mnist.L2 = 0.01
-- rbm_mnist.batchsize = 10
-- rbm_mnist.sparsity = 0.000
-- rbm_mnist.c:fill(-0.1) -- hidden bias as in honglak lee
-- rbm_mnist.U:fill(0)
-- print("#######->>>>>DATASETS COMPOSITIONS---<<<<<<<#########")
-- print("train: ", torch.histc(train_mnist.labels,10):resize(1,10) ) 
-- print("val: ", torch.histc(val_mnist.labels,10):resize(1,10) )
-- print("test: ", torch.histc(test_mnist.labels,10):resize(1,10) )
-- print("######################################################")
-- rbm_mnist = rbmtrain(rbm_mnist,train_mnist,val_mnist)
--stat_gen = rbm_medal.generativestatistics(rbm_medal,x2d_mnist,y,tcwx)


