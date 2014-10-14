

codeFolder = '../code/'

require('torch')
require(codeFolder..'rbm')
require(codeFolder..'dataset-mnist')
require(codeFolder..'ProFi')
require 'paths'
torch.setdefaulttensortype('torch.FloatTensor')
require 'nn'


tester = torch.Tester()

nInput = 1
filterSize = 2
pad = filterSize -1
nFilters = 3
poolSize =2
inputSize = 5
hidSize = inputSize - filterSize +1


maxPool = function(filters,poolsize)
	-- maxpool over several filters
	-- filters should be a 3d matrix 
	local pool = function(x)
		--Calculate exp(x) / [sum(exp(x)) +1] in numerically stable way
		local m = torch.max(x)
		local exp_x = torch.exp(x - m)
		-- normalizer = sum(exp(x)) + 1   in scaled domain
		local normalizer = torch.exp(-m) + exp_x:sum()
		exp_x:cdiv( torch.Tensor(exp_x:nElement()):fill(normalizer) )
		return exp_x
	end

	local maxPoolSingle = function(hf,hfres,poolsize)
		-- Performs probabilistic maxpooling.
		-- For each block of poolsize x poolsize calculate 
		-- exp(h_i) / [sum_i(exp(h_i)) + 1]
		-- hf should be a 2d matrix
	    local height = hf:size(1)
		local width =  hf:size(2)
		--poshidprobs = torch.Tensor(height,width):typeAs(hf)
		-- notation h_(i,j)
		for i_start = 1,height,poolsize do
		    i_end = i_start+poolsize -1
			for j_start = 1,width,poolsize do   -- columns
		    	j_end = j_start+poolsize -1
		    	hfres[{{i_start,i_end},{j_start,j_end}}] = pool(hf[{{i_start,i_end},{j_start,j_end}}])
		    end
		end   
	end



    dest = torch.Tensor(filters:size()):typeAs(filters)
    for i = 1, filters:size(1) do
    	maxPoolSingle(filters[{i,{},{}}],dest[{ i,{},{} }],poolsize)
    end
    return dest
end


function invertWeights(x)
	-- a mxn matrix
	local xtemp = torch.Tensor(x:size())
	local idx = x:size(2)
	for i = 1,x:size(2) do
		xtemp[{{},idx}] = x[{{},i}]
		idx = idx -1 
	end
	return xtemp
end

function invertWeights3d(x)
res = torch.Tensor(x:size())
for i = 1,x:size(1) do
	res[{i,{},{}}] = invertWeights(x[{i,{},{}}])
end
return x
end


W = torch.Tensor({1,-2,-3,7,2,1,-3,2,-1,2,5,2})         -- Filter:  | 1, -2|
W = W:resize(nFilters,4)  				                -- 	      |-3,  7|



modelup = nn.Sequential()
--modelup:add(nn.Reshape(1,inputSize,inputSize))
modelup:add(nn.SpatialConvolutionMM(1,nFilters,filterSize,filterSize))
--modelup:add(nn.Sigmoid())



modeldownx = nn.Sequential()
modeldownx:add(nn.SpatialZeroPadding(pad, pad, pad, pad)) -- pad (filterwidth -1) 
modeldownx:add(nn.SpatialConvolution(nFilters,1,filterSize,filterSize))
--modeldownx:add(nn.Sigmoid())


-- SET TESTING WEIGHTS OF UP MODEL
modelup.modules[1].weight = W       	      
modelup.modules[1].bias   = torch.zeros(nFilters)    


-- -- SET TESTING WEIGHTS OF DOWNX MODEL
Wtilde = invertWeights(W)     			-- Because the weights are unrolled to vectors flipping  
modeldownx.modules[2].weight = Wtilde:resize(modeldownx.modules[2].weight:size())
modeldownx.modules[2].bias = torch.zeros(1)


rbmup = function(x) 
	local res = modelup:forward(x):clone()
	return maxPool(res,filterSize)
end

rbmdownx = function(x)
	return modeldownx:forward(x):clone()
end


x = torch.Tensor({1,2,7,1,0,3,4,3,4,1,2,3,4,1,2,5,4,3,1,1,3,3,2,1,2}):resize(1,5,5) / 20
h0  = rbmup(x)

-- sampler
v1  = rbmdownx(h0)

-- sampler
h1  = rbmup(v1)

nWeights = nFilters * filterSize * filterSize

shrink = filterSize -1
nHidden  = (x:size(2)-shrink) *  (x:size(3)-shrink)
hidBias = ( h0:sum(3):sum(2):squeeze() - h1:sum(3):sum(2):squeeze() ) / nHidden

visBias = torch.Tensor({( x:sum()  - v1:sum() ) / (x:nElement())})


-- W grads

shrink = filterSize -1
x_h = x:size(2)
x_w = x:size(3)
hid_h  =x_h-filterSize+1
hid_w  =x_w-filterSize+1
--(filterSize:x_h-filterSize+1,filterSize:x_w-filterSize+1

x_in = x[  {{}, {filterSize, x_h-filterSize+1  }, {filterSize, x_w-filterSize+1  }}]
h0_in = h0[{{}, {filterSize, hid_h-filterSize+1}, {filterSize, hid_w-filterSize+1}}]
h0_in_filter = h0_in:resize(nFilters,nInput,h0_in:size(2),h0_in:size(3))

nnw  = nn.SpatialConvolution(nInput,nFilters,filterSize,filterSize)
-- -- -- -- gradsnn.weight is poshidprobs(Wfilter:Hhidden-Wfilter+1,Wfilter:Whidden-Wfilter+1,:,:)
nnw.weight = h0_in_filter
nnw.bias = torch.zeros(nFilters) 

dw_pos = nnw:forward(x_in)

-- -- -- gradsnn:add(nn.Reshape(1,inputSize-filterSize,inputSize-filterSize))




--
--  DEFINING TESTS RESULTS
--
x_test = torch.Tensor({
	0.0500,    0.1000,    0.3500,    0.0500,         0,
    0.1500,    0.2000,    0.1500,    0.2000,    0.0500,
    0.1000,    0.1500,    0.2000,    0.0500,    0.1000,
    0.2500,    0.2000,    0.1500,    0.0500,    0.0500,
    0.1500,    0.1500,    0.1000,    0.0500,    0.1000}):resize(1,5,5)

poshidprobs_test = torch.Tensor({
    0.2756,    0.1066,    0.4334,    0.1069,
    0.2042,    0.2898,    0.0792,    0.2500,
    0.2405,    0.1873,    0.1723,    0.1811,
    0.2405,    0.1782,    0.1904,    0.2840,

    0.1723,    0.1904,    0.3180,    0.1058,
    0.2445,    0.2445,    0.1579,    0.2603,
    0.1586,    0.1937,    0.1956,    0.2056,
    0.2749,    0.2141,    0.2056,    0.2162,

    0.2073,    0.3777,    0.2121,    0.2465,
    0.1614,    0.1972,    0.3327,    0.1224,
    0.3485,    0.2582,    0.2598,    0.2024,
    0.1819,    0.1566,    0.2127,    0.2024
    }):resize(3,4,4)

-- -- V1 tests
v1_test = torch.Tensor({
    0.4129,    0.1453,    1.5898,   -0.0524,    0.3851,
    0.2243,    4.4263,    0.8105,    6.0015,    1.4578,
   -0.3297,    2.3153,    4.5030,    1.3770,    2.7639,
    1.1535,    3.4532,    2.9705,    2.7326,    2.1365,
   -0.6367,    2.2036,    1.8644,    1.6806,    2.8251}):resize(1,5,5)


neghidprobs_test = torch.Tensor({
	0.9549,    0.0000,    1.0000,    0.0000,
    0.0000,    0.0451,    0.0000,    0.0000,
    0.9848,    0.0000,    0.2265,    0.0000,
    0.0152,    0.0000,    0.0001,    0.7734,

    0.0558,    0.0000,    0.1035,    0.0000,
    0.2138,    0.7304,    0.0000,    0.8965,
    0.0009,    0.0006,    0.1927,    0.0009,
    0.9923,    0.0062,    0.1175,    0.6887,

    0.0000,    0.9999,    0.0000,    0.0621,
    0.0000,    0.0001,    0.9379,    0.0000,
    0.0000,    1.0000,    0.0287,    0.9688,
    0.0000,    0.0000,    0.0010,    0.0015}):resize(3,4,4)

visBias_test =  torch.Tensor({-1.7306})
hidBias_test = torch.Tensor({ -0.0363,   -0.0401,   -0.0200})
dw_test = torch.Tensor({
   -1.0870,   -0.2431,
   -0.6546,   -0.7233,

   -3.9614,   -0.7434,
   -2.1294,   -3.7218,

   -3.0255,  -10.0109,
   -7.5756,   -4.2333})

dw_pos_test = torch.Tensor({
    0.1324,    0.1054,
    0.1226,    0.0986,

    0.1408,   0.1168,
    0.1363,    0.0956,

    0.1800,    0.1607,
    0.1867,    0.1078
	}):resize(3,2,2)

print "################ TESTS ############################" 
assert(checkequality(x,x_test,-4,false))
assert(checkequality(h0,poshidprobs_test,-4,false))
assert(checkequality(v1,v1_test,-4,false))
assert(checkequality(h1,neghidprobs_test,-4,false))

print("Testing Gradients...")
assert(checkequality(visBias,visBias_test,-4,false))
assert(checkequality(hidBias,hidBias_test,-4,false))
--assert(checkequality(dw_pos,dw_pos_test,-4,false))
print('OK')
-- -- ---UPDATES





print('ADD SIGMOIDS!!! ')
