codeFolder = '../code/'

require('torch')
require(codeFolder..'rbm')
require(codeFolder..'ProFi')
require(codeFolder..'dataset-from-tensor.lua')
require 'paths'
torch.manualSeed(101)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(2)
 
--train_mnist,val_mnist,test_mnist  = mnist.loadMnist(50)
-- {
--   1 : FloatTensor - size: 1x32x32
--   2 : FloatTensor - size: 10
-- }

opts = {}
opts.n_hidden     = 500
opts.numepochs    = 500
opts.learningrate = 0.05
opts.alpha = 0
opts.beta = 0
opts.dropconnect = 0



 n_classes = 4;
 n_hidden  = 7;
 n_visible = 3;

-- create test data and test weights
 x = torch.Tensor({0.4170,0.7203, 0.0001}):resize(1,1,n_visible)
 x2d = x:view(1,3)
 y = torch.Tensor({0,0,1, 0}):resize(1,n_classes)

 U = torch.Tensor({{ 0.0538,   -0.0113,   -0.0688,  -0.0074},
   						{ 0.0564,    0.0654,    0.0357,    0.0584},
   						{-0.0593,    0.0047,    0.0698,   -0.0295},
   						{-0.0658,    0.0274,    0.0355,   -0.0303},
   						{-0.0472,   -0.0264,   -0.0314,   -0.0529},
   						{ 0.0540,    0.0266,    0.0413,   -0.0687},
   						{-0.0574,    0.0478,   -0.0567,    0.0255}})
 W =  torch.Tensor({{-0.0282 , -0.0115 ,   0.0084},
   						 {-0.0505  ,  0.0265 ,  -0.0514},
   						 {-0.0582  , -0.0422 ,  -0.0431},
   						 {-0.0448  ,  0.0540 ,   0.0430},
  				         {-0.0221  , -0.0675 ,   0.0669},
   		                 {-0.0147  ,  0.0244 ,  -0.0267},
   		                 {0.0055   , -0.0118 ,   0.0275}})
 b = torch.zeros(n_visible,1)
 c = torch.zeros(n_hidden,1)
 d = torch.zeros(n_classes,1)
 vW = torch.Tensor(W:size()):zero()
 vU = torch.Tensor(U:size()):zero()
 vb = torch.Tensor(b:size()):zero()
 vc = torch.Tensor(c:size()):zero()
 vd = torch.Tensor(d:size()):zero()



 rbm = {W = W:clone(), U = U:clone(), d = d:clone(), b = b:clone(), c = c:clone(),
       vW =vW,vU = vU,vb = vb, vc = vc, vd = vd,};
 rbm.dW = torch.Tensor(rbm.W:size()):zero()
 rbm.dU = torch.Tensor(rbm.U:size()):zero()
 rbm.db = torch.Tensor(rbm.b:size()):zero()
 rbm.dc = torch.Tensor(rbm.c:size()):zero()
 rbm.dd = torch.Tensor(rbm.d:size()):zero()

rbm.rand  = function(m,n) return torch.Tensor(m,n):fill(1):mul(0.5)end -- for testing
rbm.n_classes = n_classes
rbm.n_visible = n_visible
rbm.n_samples = 1
rbm.n_input = 1
rbm.numepochs = 1
rbm.learningrate = 0.1
rbm.alpha = 1
rbm.beta = 0
rbm.momentum = 0
rbm.dropout = opts.dropout or 0
rbm.dropouttype = "bernoulli"
rbm.dropconnect = opts.dropconnect or 0
rbm.L1 = opts.L1 or 0
rbm.L2 = opts.L2 or 0
rbm.sparsity = opts.sparsity or 0
rbm.err_recon_train = torch.Tensor(1):fill(-1)
rbm.err_train = torch.Tensor(1):fill(-1)
rbm.err_val = torch.Tensor(1):fill(-1)
rbm.temp_file = "blabla"
rbm.patience = 15
rbm.one_by_classes = torch.ones(1,rbm.U:size(2))
rbm.hidden_by_one  = torch.ones(rbm.W:size(1),1)
rbm.traintype = 'CD' -- CD
rbm.npcdchains = 1
rbm.cdn = 1
rbm.n_hidden  = n_hidden
rbm.currentepoch = 1
rbm.bottomrbm = 1
rbm.toprbm = true
rbm.samplex = false
rbm.lrdecay = 0 -- no decay
rbm.up = rbmup
rbm.downx = rbmdownx
rbm.downy = rbmdowny
rbm.errorfunction = function(conf) return 1-conf:accuracy() end
rbm.precalctcwx = 1
rbm.generativestatistics = grads.generativestatistics
rbm.generativegrads  = grads.generativegrads 
rbm.discriminativegrads     =  grads.discriminativegrads
rbm.pygivenx  = grads.pygivenx
rbm.pygivenxdropout = grads.pygivenxdropout
rbm.batchsize = 1
rbm.visxsampler = bernoullisampler
rbm.hidsampler =  bernoullisampler
rbm.conf = ConfusionMatrix(4)

---------------------------------------------------------
-- TRUE VALUES rbm-util
---------------------------------------------------------
 h0_true = torch.Tensor({ 0.4778,0.5084,0.5038,0.5139,0.4777,0.5132,0.4843}):resize(1,n_hidden)
 h0_rnd_true = torch.Tensor({0,1,1,1,0,1,0}):resize(1,n_hidden)
 vkx_true = torch.Tensor({ 0.4580,0.5156,0.4805}):resize(1,n_visible)
 vkx_rnd_true = torch.Tensor({0,1,0}):resize(1,n_visible)
 vky_true = torch.Tensor({ 0.2319,0.2664,0.2824,0.2194}):resize(1,n_classes)
 vky_rnd_true = torch.Tensor({ 0,0,1,0}):resize(1,n_classes)
 p_y_given_x_true =  torch.Tensor({0.2423,0.2672,0.2532,0.2373}):resize(1,n_classes)


---------------------------------------------------------
-- GENERATIVE TRUE WEIGHTS
---------------------------------------------------------
 dw_gen_true = torch.Tensor({
	{0.1992,   -0.1358,    0.0001},
    {0.2120,   -0.1493,    0.0001},
    {0.2101,   -0.1440,    0.0001},
    {0.2143,   -0.1522,    0.0001},
    {0.1992,   -0.1312,    0.0001},
    {0.2140,   -0.1468,    0.0001},
    {0.2020,   -0.1340,    0.0001}
	})
 du_gen_true = torch.Tensor({
	     {0,         0,   -0.0021,         0},
         {0,         0,   -0.0071,         0},
         {0,         0,   -0.0031,         0},
         {0,         0,   -0.0084,         0},
         {0,         0,    0.0024,         0},
         {0,         0,   -0.0032,         0},
         {0,         0,    0.0014,         0}
	})




 db_gen_true = torch.Tensor({0.4170,-0.2797,0.0001}):resize(n_visible,1)
 dc_gen_true = torch.Tensor({-0.0021,-0.0071,-0.0031,-0.0084,0.0024,-0.0032,0.0014}):resize(n_hidden,1)
 dd_gen_true = torch.Tensor({0,0,0,0}):resize(n_classes,1)

---------------------------------------------------------
--- DISCRIMINATIVE TRUE WEIGHTS
---------------------------------------------------------
 dw_dis_true = torch.Tensor({
   {-0.0062,   -0.0107,   -0.0000},
   {-0.0019,   -0.0033,   -0.0000},
   { 0.0075,    0.0130,    0.0000},
   { 0.0044,    0.0076,    0.0000},
   { 0.0008,    0.0014,    0.0000},
   { 0.0028,    0.0049,    0.0000},
   {-0.0049,   -0.0085,   -0.0000}})

 du_dis_true = torch.Tensor({
   {-0.1232,   -0.1315,    0.3568,   -0.1170},
   {-0.1245,   -0.1378,    0.3797,   -0.1220},
   {-0.1143,   -0.1303,    0.3762,   -0.1136},
   {-0.1184,   -0.1368,    0.3838,   -0.1180},
   {-0.1148,   -0.1280,    0.3567,   -0.1121},
   {-0.1251,   -0.1361,    0.3832,   -0.1152},
   {-0.1173,   -0.1364,    0.3616,   -0.1198}})


 dc_dis_true = torch.Tensor({-0.0149,-0.0046,0.0181,0.0106,0.0019,0.0067,-0.0118}):resize(n_hidden,1)
 dd_dis_true = torch.Tensor({ -0.2423,-0.2672,0.7468,-0.2373}):resize(n_classes,1)

---------------------------------------------------------
--- CHECK FOR SIDE EFFECTS
---------------------------------------------------------
-- if they have side effects on x,y or rbm then generative tests will fails
 _h0 = rbmup(rbm,x2d,y)   -- UP
 _h0_rnd = bernoullisampler(_h0,rbm.rand)
 _vkx = rbmdownx(rbm,_h0_rnd)   -- DOWNX
 _vkx_rnd = bernoullisampler(_vkx,rbm.rand)
 _vky = rbmdowny(rbm,_h0_rnd)
 _vky_rnd = samplevec(_vky,rbm.rand)
 _p_y_given_x = grads.pygivenx(rbm,x2d)
rbm.learningrate = 0 -- to avoid updating weights
--rbm = rbmtrain(rbm,x,y)


---------------------------------------------------------
--- CALCULATE VALUES FOR TESTING
---------------------------------------------------------
 tcwx = torch.mm( x2d,rbm.W:t() ):add( rbm.c:t() )

print(x2d,y,tcwx)
print(rbm)
stat_gen= grads.generativestatistics(rbm,x2d,y,tcwx) 
grads_gen = grads.generativegrads(rbm,x2d,y,stat_gen)
grads_dis, p_y_given_x_dis = grads.discriminativegrads(rbm,x2d,y,tcwx) 
-- -- calculte value
 --  h0 = rbmup(rbm,x,y)   -- UP
 --  h0_rnd = sampler(h0,rbm.rand)
 --  vkx = rbmdownx(rbm,h0_rnd)   -- DOWNX
 --  vkx_rnd = sampler(vkx,rbm.rand)
 --  vky = rbmdowny(rbm,h0_rnd)
 --  vky_rnd = samplevec(vky,rbm.rand)
  p_y_given_x =grads.pygivenx(rbm,x2d)


-- ---------------------------------------------------------
-- --- TEST RBM-UTIL FUNCTIONS
-- ---------------------------------------------------------
assert(checkequality(stat_gen.h0, h0_true,-4),'Check h0 failed')
--assert(checkequality(stat_gen.h0_rnd, h0_rnd_true),'Check h0_rnd failed')
--assert(checkequality(stat_gen.vkx, vkx_true),'Check vkx failed')
assert(checkequality(stat_gen.vkx_unsampled, vkx_true),'Check vkx_unsampled failed')


assert(checkequality(stat_gen.vkx, vkx_rnd_true),'Check vkx_rnd failed')


assert(checkequality(stat_gen.vky, vky_rnd_true),'Check vky_rnd failed')

assert(checkequality(p_y_given_x, p_y_given_x_true),'Check p_y_given_x failed')
assert(checkequality(p_y_given_x, p_y_given_x_dis,-3),'Check p_y_given_x_dis failed')



-- print "TEST of RBM-UTIL gradients                                   : PASSED"

-- ---------------------------------------------------------
-- --- TEST GENERATIVE WEIGHTS 
-- ---------------------------------------------------------
assert(checkequality(grads_gen.dW, dw_gen_true,-3),'Check dw failed')
assert(checkequality(grads_gen.dU, du_gen_true),'Check du failed')
assert(checkequality(grads_gen.db, db_gen_true,-3),'Check db failed')
assert(checkequality(grads_gen.dc, dc_gen_true),'Check dc failed')
assert(checkequality(grads_gen.dd, dd_gen_true),'Check dd failed')
print "TEST of GENERATIVE gradients                                 : PASSED"


-- ---------------------------------------------------------
-- --- TEST DISCRIMINATIVE WEIGHTS 
-- ---------------------------------------------------------
assert(checkequality(grads_dis.dW, dw_dis_true,-3),'Check dw failed')
assert(checkequality(grads_dis.dU, du_dis_true),'Check du failed')
assert(checkequality(grads_dis.dc, dc_dis_true,-3),'Check dc failed')
assert(checkequality(grads_dis.dd, dd_dis_true),'Check dd failed')
print "TEST of DISCRIMINATIVE gradients                             : PASSED"


-- ---------------------------------------------------------
-- --- TEST RBMTRAIN 
-- ---------------------------------------------------------

trainData = datatensor.createDataset(x,y,{'A','B','C','D',},{1,3})


rbm.beta = 0
rbm.learningrate = 0.1
rbm.dropout = 0
rbm.dropconnect = 0
rbm.boost = 'none'
rbm.progress = 0

--train ={}
--train.data = x
--train.labels = torch.Tensor({3}):float()

orgrbm = {}
orgrbm.W = rbm.W:clone()
orgrbm.U = rbm.U:clone()
orgrbm.b = rbm.b:clone()
orgrbm.c = rbm.c:clone()
orgrbm.d = rbm.d:clone()


rbm = rbmtrain(rbm,trainData)





-- check generative
rbm.alpha = 1
assert(checkequality(rbm.W, torch.add(W ,torch.mul(dw_gen_true,rbm.learningrate)) ,-3),'Check rbm.W failed')
assert(checkequality(rbm.U, torch.add(U ,torch.mul(du_gen_true,rbm.learningrate)) ,-3),'Check rbm.U failed')
assert(checkequality(rbm.b, torch.add(b ,torch.mul(db_gen_true,rbm.learningrate)) ,-3),'Check rbm.b failed')
assert(checkequality(rbm.c, torch.add(c ,torch.mul(dc_gen_true,rbm.learningrate)) ,-3),'Check rbm.c failed')
assert(checkequality(rbm.d, torch.add(d ,torch.mul(dd_gen_true,rbm.learningrate)) ,-3),'Check rbm.d failed')
print('Generative Training                  : OK')

-- check discriminative

-- rbm.W = orgrbm.W:clone()
-- rbm.U = orgrbm.U:clone()
-- rbm.b = orgrbm.b:clone()
-- rbm.c = orgrbm.c:clone()
-- rbm.d = orgrbm.d:clone()

-- assert(checkequality(rbm.W, orgrbm.W ,-3),'Check rbm.W failed')
-- assert(checkequality(rbm.U, orgrbm.U ,-3),'Check rbm.U failed')
-- assert(checkequality(rbm.b, orgrbm.b ,-3),'Check rbm.b failed')
-- assert(checkequality(rbm.c, orgrbm.c ,-3),'Check rbm.c failed')
-- assert(checkequality(rbm.d, orgrbm.d ,-3),'Check rbm.d failed')


rbm.W = W:clone()
rbm.U = U:clone()
rbm.b = b:clone()
rbm.c = c:clone()
rbm.d = d:clone()
rbm.vW:fill(0)
rbm.vU:fill(0)
rbm.vb:fill(0)
rbm.vc:fill(0)
rbm.vd:fill(0)
rbm.dW:fill(0)
rbm.dU:fill(0)
rbm.db:fill(0)
rbm.dc:fill(0)
rbm.dd:fill(0)
rbm.alpha = 0
rbm = rbmtrain(rbm,trainData)


rbm.learningrate = 0.1
assert(checkequality(rbm.W, torch.add(W ,torch.mul(dw_dis_true,rbm.learningrate)) ,-3),'Check rbm.W failed')
assert(checkequality(rbm.U, torch.add(U ,torch.mul(du_dis_true,rbm.learningrate)) ,-3),'Check rbm.U failed')
--assert(checkequality(rbm.b, torch.add(b ,torch.mul(db_gen_true,rbm.learningrate)) ,-3),'Check rbm.b failed')
assert(checkequality(rbm.c, torch.add(c ,torch.mul(dc_dis_true,rbm.learningrate)) ,-3),'Check rbm.c failed')
assert(checkequality(rbm.d, torch.add(d ,torch.mul(dd_dis_true,rbm.learningrate)) ,-3),'Check rbm.d failed')
print('Discriminative Training                : OK')
-- print "TEST of rbmtrain 


-- check hybrid
rbm.W = W:clone()
rbm.U = U:clone()
rbm.b = b:clone()
rbm.c = c:clone()
rbm.d = d:clone()
rbm.vW:fill(0)
rbm.vU:fill(0)
rbm.vb:fill(0)
rbm.vc:fill(0)
rbm.vd:fill(0)
rbm.dW:fill(0)
rbm.dU:fill(0)
rbm.db:fill(0)
rbm.dc:fill(0)
rbm.dd:fill(0)
rbm.alpha = 0.1
rbm = rbmtrain(rbm,trainData)
assert(checkequality(rbm.W, torch.add(W ,torch.mul(dw_dis_true,rbm.learningrate*(1-rbm.alpha))):add(torch.mul(dw_gen_true,rbm.learningrate*rbm.alpha)) ,-3),'Check rbm.W failed')
assert(checkequality(rbm.U, torch.add(U ,torch.mul(du_dis_true,rbm.learningrate*(1-rbm.alpha))):add(torch.mul(du_gen_true,rbm.learningrate*rbm.alpha)) ,-3),'Check rbm.U failed')
assert(checkequality(rbm.b, torch.add(b ,torch.mul(db_gen_true,rbm.learningrate*(rbm.alpha))) ,-3),'Check rbm.b failed')
assert(checkequality(rbm.c, torch.add(c ,torch.mul(dc_dis_true,rbm.learningrate*(1-rbm.alpha))):add(torch.mul(dc_gen_true,rbm.learningrate*rbm.alpha)) ,-3),'Check rbm.c failed')
assert(checkequality(rbm.d, torch.add(d ,torch.mul(dd_dis_true,rbm.learningrate*(1-rbm.alpha))):add(torch.mul(dd_gen_true,rbm.learningrate*rbm.alpha)) ,-3),'Check rbm.d failed')
print('Hybrid Training                       : OK')





--                                            : PASSED"


-- print(stat_gen['vky'],vky_rnd_true)
-- error()



-- opts_mnist = {}
-- opts_mnist.n_hidden = 10
-- rbm_mnist = rbmsetup(opts_mnist,train_mnist)
-- rbm_mnist.alpha = 0
-- rbm_mnist.beta = 0
-- rbm_mnist.learningrate = 0.1
-- rbm_mnist.dropout = 0.5
-- rbm_mnist.dropconnect = 0
-- rbm_mnist = rbmtrain(rbm_mnist,train_mnist,val_mnist)


-- -- extend training with new objective 
-- rbm_mnist.dropout = 0.1
-- rbm_mnist.numepochs = 10
-- rbm_mnist.learningrate =0.5
-- rbm_mnist.alpha = 0.5
-- rbm_mnist = rbmtrain(rbm_mnist,train_mnist,val_mnist)

-- uppass = rbmuppass(rbm_mnist,train_mnist)
-- uppass = rbmuppass(rbm_mnist,test_mnist)


-- rbm_mnist.toprbm = false
-- rbm_mnist.alpha = 1
-- rbm_mnist.currentepoch = 1
-- rbm_mnist.U = nil 
-- rbm_mnist.dU = nil 
-- rbm_mnist.d = nil 
-- rbm_mnist.dd = nil 
-- rbm_mnist.vd = nil 
-- rbm_mnist.vU = nil 

-- rbm_mnist = rbmtrain(rbm_mnist,train_mnist)

-- uppass = rbmuppass(rbm_mnist,train_mnist)
-- uppass = rbmuppass(rbm_mnist,test_mnist)




-- -- MNIST TEST



-- --assert(checkequality(torch.add(U,du_true):mul(rbm.learningrate), rbm.U,-3),'Check rbm.W failed')
-- --assert(checkequality(torch.add(b,db_true):mul(rbm.learningrate), rbm.b,-3),'Check rbm.b failed')
-- --assert(checkequality(torch.add(c,dc_true):mul(rbm.learningrate), rbm.c,-3),'Check rbm.c failed')
-- --assert(checkequality(torch.add(d,dd_true):mul(rbm.learningrate), rbm.d,-3),'Check rbm.d failed')
-- --assert(checkequality(torch.add(U,du_true), rbm.U,-3),'Check rbm.U failed')
-- --assert(checkequality(torch.add(b,db_true), rbm.b,-3),'Check rbm.b failed')
-- --assert(checkequality(torch.add(c,dc_true), rbm.c,-3),'Check rbm.c failed')
-- --assert(checkequality(torch.add(d,dd_true), rbm.d,-3),'Check rbm.d failed')
-- --assert(checkequality(du, du_true),'Check du failed')
-- --assert(checkequality(db, db_true),'Check db failed')
-- --assert(checkequality(dc, dc_true),'Check dc failed')
-- --assert(checkequality(dd, dd_true),'Check dd failed')

-- -- INit rbm
-- --[[torch.setdefaulttensortype('torch.FloatTensor')





--  x = torch.zeros(10,784)
--  y = torch.zeros(10,10)
-- ]]







-- --[[print(rbm.W)
-- print(rbm.U)
-- print(rbm.b)
-- print(rbm.c)
-- print(rbm.d)--]]


-- -- tcwx = torch.mm( x,rbm.W:t() ):add( rbm.c:t() )
-- --tcwx =  -0.4333    0.1500    0.1500    0.1133    0.1500    0.1500   -0.1167

-- ---print(tcwx)

