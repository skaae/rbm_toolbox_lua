--require('nn')
require('torch')
-- load rbm functions
require('rbm')
require('dataset-mnist')
require('rbm-grads')

torch.manualSeed(101)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)




mnist_folder = '../mnist-th7'
rescale = 1
x_train, y_train, x_val, y_val, x_test, y_test = mnist.createdatasets(mnist_folder,rescale) 


opts = {}
opts.n_hidden     = 500
opts.numepochs    = 500
opts.learningrate = 0.05
opts.alpha = 0
opts.beta = 0
opts.dropconnect = 0


local n_classes = 4;
local n_hidden  = 7;
local n_visible = 3;

-- create test data and test weights
local x = torch.Tensor({0.4170,0.7203, 0.0001}):resize(1,n_visible)
local y = torch.Tensor({0,0,1, 0}):resize(1,n_classes)

local U = torch.Tensor({{ 0.0538,   -0.0113,   -0.0688,  -0.0074},
   						{ 0.0564,    0.0654,    0.0357,    0.0584},
   						{-0.0593,    0.0047,    0.0698,   -0.0295},
   						{-0.0658,    0.0274,    0.0355,   -0.0303},
   						{-0.0472,   -0.0264,   -0.0314,   -0.0529},
   						{ 0.0540,    0.0266,    0.0413,   -0.0687},
   						{-0.0574,    0.0478,   -0.0567,    0.0255}})
local W =  torch.Tensor({{-0.0282 , -0.0115 ,   0.0084},
   						 {-0.0505  ,  0.0265 ,  -0.0514},
   						 {-0.0582  , -0.0422 ,  -0.0431},
   						 {-0.0448  ,  0.0540 ,   0.0430},
  				         {-0.0221  , -0.0675 ,   0.0669},
   		                 {-0.0147  ,  0.0244 ,  -0.0267},
   		                 {0.0055   , -0.0118 ,   0.0275}})
local b = torch.zeros(n_visible,1)
local c = torch.zeros(n_hidden,1)
local d = torch.zeros(n_classes,1)
local vW = torch.Tensor(W:size()):zero()
local vU = torch.Tensor(U:size()):zero()
local vb = torch.Tensor(b:size()):zero()
local vc = torch.Tensor(c:size()):zero()
local vd = torch.Tensor(d:size()):zero()



local rbm = {W = W:clone(), U = U:clone(), d = d:clone(), b = b:clone(), c = c:clone(),
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
rbm.numepochs = 1
rbm.learningrate = 0.1
rbm.alpha = 1
rbm.beta = 0
rbm.momentum = 0
rbm.dropout = opts.dropout or 0
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
rbm.traintype = 0 -- CD
rbm.npcdchains = 1
rbm.cdn = 1
rbm.n_hidden  = n_hidden
rbm.currentepoch = 1

---------------------------------------------------------
-- TRUE VALUES rbm-util
---------------------------------------------------------
local h0_true = torch.Tensor({ 0.4778,0.5084,0.5038,0.5139,0.4777,0.5132,0.4843}):resize(1,n_hidden)
local h0_rnd_true = torch.Tensor({0,1,1,1,0,1,0}):resize(1,n_hidden)
local vkx_true = torch.Tensor({ 0.4580,0.5156,0.4805}):resize(1,n_visible)
local vkx_rnd_true = torch.Tensor({0,1,0}):resize(1,n_visible)
local vky_true = torch.Tensor({ 0.2319,0.2664,0.2824,0.2194}):resize(1,n_classes)
local vky_rnd_true = torch.Tensor({ 0,0,1,0}):resize(1,n_classes)
local p_y_given_x_true =  torch.Tensor({0.2423,0.2672,0.2532,0.2373}):resize(1,n_classes)


---------------------------------------------------------
-- GENERATIVE TRUE WEIGHTS
---------------------------------------------------------
local dw_true = torch.Tensor({
	{0.1992,   -0.1358,    0.0001},
    {0.2120,   -0.1493,    0.0001},
    {0.2101,   -0.1440,    0.0001},
    {0.2143,   -0.1522,    0.0001},
    {0.1992,   -0.1312,    0.0001},
    {0.2140,   -0.1468,    0.0001},
    {0.2020,   -0.1340,    0.0001}
	})
local du_true = torch.Tensor({
	     {0,         0,   -0.0021,         0},
         {0,         0,   -0.0071,         0},
         {0,         0,   -0.0031,         0},
         {0,         0,   -0.0084,         0},
         {0,         0,    0.0024,         0},
         {0,         0,   -0.0032,         0},
         {0,         0,    0.0014,         0}
	})




local db_true = torch.Tensor({0.4170,-0.2797,0.0001}):resize(n_visible,1)
local dc_true = torch.Tensor({-0.0021,-0.0071,-0.0031,-0.0084,0.0024,-0.0032,0.0014}):resize(n_hidden,1)
local dd_true = torch.Tensor({0,0,0,0}):resize(n_classes,1)

---------------------------------------------------------
--- DISCRIMINATIVE TRUE WEIGHTS
---------------------------------------------------------
local dw_dis_true = torch.Tensor({
   {-0.0062,   -0.0107,   -0.0000},
   {-0.0019,   -0.0033,   -0.0000},
   { 0.0075,    0.0130,    0.0000},
   { 0.0044,    0.0076,    0.0000},
   { 0.0008,    0.0014,    0.0000},
   { 0.0028,    0.0049,    0.0000},
   {-0.0049,   -0.0085,   -0.0000}})

local du_dis_true = torch.Tensor({
   {-0.1232,   -0.1315,    0.3568,   -0.1170},
   {-0.1245,   -0.1378,    0.3797,   -0.1220},
   {-0.1143,   -0.1303,    0.3762,   -0.1136},
   {-0.1184,   -0.1368,    0.3838,   -0.1180},
   {-0.1148,   -0.1280,    0.3567,   -0.1121},
   {-0.1251,   -0.1361,    0.3832,   -0.1152},
   {-0.1173,   -0.1364,    0.3616,   -0.1198}})


local dc_dis_true = torch.Tensor({-0.0149,-0.0046,0.0181,0.0106,0.0019,0.0067,-0.0118}):resize(n_hidden,1)
local dd_dis_true = torch.Tensor({ -0.2423,-0.2672,0.7468,-0.2373}):resize(n_classes,1)

---------------------------------------------------------
--- CHECK FOR SIDE EFFECTS
---------------------------------------------------------
-- if they have side effects on x,y or rbm then generative tests will fails
local _h0 = rbmup(rbm,x,y)   -- UP
local _h0_rnd = sampler(_h0,rbm.rand)
local _vkx = rbmdownx(rbm,_h0_rnd)   -- DOWNX
local _vkx_rnd = sampler(_vkx,rbm.rand)
local _vky = rbmdowny(rbm,_h0_rnd)
local _vky_rnd = samplevec(_vky,rbm.rand)
local _p_y_given_x = grads.pygivenx(rbm,x)
rbm.learningrate = 0 -- to avoid updating weights
--rbm = rbmtrain(rbm,x,y)


---------------------------------------------------------
--- CALCULATE VALUES FOR TESTING
---------------------------------------------------------
local tcwx = torch.mm( x,rbm.W:t() ):add( rbm.c:t() )

local dw, du, db, dc ,dd, vkx = grads.generative(rbm,x,y,tcwx) 
local dw_dis, du_dis, dc_dis ,dd_dis, p_y_given_x_dis = grads.discriminative(rbm,x,y,tcwx) 
-- calculte value
local h0 = rbmup(rbm,x,y)   -- UP
local h0_rnd = sampler(h0,rbm.rand)
local vkx = rbmdownx(rbm,h0_rnd)   -- DOWNX
local vkx_rnd = sampler(vkx,rbm.rand)
local vky = rbmdowny(rbm,h0_rnd)
local vky_rnd = samplevec(vky,rbm.rand)
local p_y_given_x =grads.pygivenx(rbm,x)


---------------------------------------------------------
--- TEST RBM-UTIL FUNCTIONS
---------------------------------------------------------
assert(checkequality(h0, h0_true,-4),'Check h0 failed')
assert(checkequality(h0_rnd, h0_rnd_true),'Check h0_rnd failed')
assert(checkequality(vkx, vkx_true),'Check vkx failed')
assert(checkequality(vkx_rnd, vkx_rnd_true),'Check vkx_rnd failed')
assert(checkequality(vky, vky_true),'Check vky failed')
assert(checkequality(vky_rnd, vky_rnd_true),'Check vky_rnd failed')
assert(checkequality(p_y_given_x, p_y_given_x_true),'Check p_y_given_x failed')
assert(checkequality(p_y_given_x, p_y_given_x_dis,-3),'Check p_y_given_x_dis failed')
print "TEST of RBM-UTIL gradients                                   : PASSED"

---------------------------------------------------------
--- TEST GENERATIVE WEIGHTS 
---------------------------------------------------------
assert(checkequality(dw, dw_true,-3),'Check dw failed')
assert(checkequality(du, du_true),'Check du failed')
assert(checkequality(db, db_true,-3),'Check db failed')
assert(checkequality(dc, dc_true),'Check dc failed')
assert(checkequality(dd, dd_true),'Check dd failed')
print "TEST of GENERATIVE gradients                                 : PASSED"


---------------------------------------------------------
--- TEST DISCRIMINATIVE WEIGHTS 
---------------------------------------------------------
assert(checkequality(dw_dis, dw_dis_true,-3),'Check dw failed')
assert(checkequality(du_dis, du_dis_true),'Check du failed')
assert(checkequality(dc_dis, dc_dis_true,-3),'Check dc failed')
assert(checkequality(dd_dis, dd_dis_true),'Check dd failed')
print "TEST of DISCRIMINATIVE gradients                             : PASSED"


---------------------------------------------------------
--- TEST RBMTRAIN 
---------------------------------------------------------
rbm.alpha = 1
rbm.beta = 0
rbm.learningrate = 0.1
rbm.dropout = 0
rbm.dropconnect = 0
rbm = rbmtrain(rbm,x,y)

assert(checkequality(rbm.W, torch.add(W ,torch.mul(dw_true,rbm.learningrate)) ,-3),'Check rbm.W failed')
assert(checkequality(rbm.U, torch.add(U ,torch.mul(du_true,rbm.learningrate)) ,-3),'Check rbm.U failed')
assert(checkequality(rbm.b, torch.add(b ,torch.mul(db_true,rbm.learningrate)) ,-3),'Check rbm.b failed')
assert(checkequality(rbm.c, torch.add(c ,torch.mul(dc_true,rbm.learningrate)) ,-3),'Check rbm.c failed')
assert(checkequality(rbm.d, torch.add(d ,torch.mul(dd_true,rbm.learningrate)) ,-3),'Check rbm.d failed')
print "TEST of rbmtrain                                             : PASSED"

--rbm.alpha = 0
--rbm.beta = 0
--rbm.learningrate = 0.1
--rbm.dropout = 0.5
--rbm.dropconnect = 0
--rbm = rbmtrain(rbm,x,y)



--rbm.dropout = 0.1
--rbm.numepochs = 2
--rbm.learningrate =0.5
-- rbm.err_recon_train = torch.Tensor(rbm.numepochs)
--rbm = rbmtrain(rbm,x,y)


--opts = {}
--opts.n_hidden     = 500
--opts.numepochs    = 10
--opts.learningrate = 0.05
--opts.alpha = 0.1
--opts.beta = 0.4
--rbm1 = rbmsetup(opts,x,y)
--rbm1 = rbmtrain(rbm1,x,y)



-- MNIST TEST



--assert(checkequality(torch.add(U,du_true):mul(rbm.learningrate), rbm.U,-3),'Check rbm.W failed')
--assert(checkequality(torch.add(b,db_true):mul(rbm.learningrate), rbm.b,-3),'Check rbm.b failed')
--assert(checkequality(torch.add(c,dc_true):mul(rbm.learningrate), rbm.c,-3),'Check rbm.c failed')
--assert(checkequality(torch.add(d,dd_true):mul(rbm.learningrate), rbm.d,-3),'Check rbm.d failed')
--assert(checkequality(torch.add(U,du_true), rbm.U,-3),'Check rbm.U failed')
--assert(checkequality(torch.add(b,db_true), rbm.b,-3),'Check rbm.b failed')
--assert(checkequality(torch.add(c,dc_true), rbm.c,-3),'Check rbm.c failed')
--assert(checkequality(torch.add(d,dd_true), rbm.d,-3),'Check rbm.d failed')
--assert(checkequality(du, du_true),'Check du failed')
--assert(checkequality(db, db_true),'Check db failed')
--assert(checkequality(dc, dc_true),'Check dc failed')
--assert(checkequality(dd, dd_true),'Check dd failed')

-- INit rbm
--[[torch.setdefaulttensortype('torch.FloatTensor')





local x = torch.zeros(10,784)
local y = torch.zeros(10,10)
]]







--[[print(rbm.W)
print(rbm.U)
print(rbm.b)
print(rbm.c)
print(rbm.d)--]]


--local tcwx = torch.mm( x,rbm.W:t() ):add( rbm.c:t() )
--tcwx =  -0.4333    0.1500    0.1500    0.1133    0.1500    0.1500   -0.1167

---print(tcwx)

