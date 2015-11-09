-- An example which uses the create_weight_image to display the filter for the
-- rbm.
-- The RBM is trained on a small data set (for 1 epoch) after which the image
-- is generated.

codeFolder = '../code/'

require('torch')
require(codeFolder..'rbm')
require(codeFolder..'dataset-mnist')
require(codeFolder..'rbm-visualisation.lua')
ProFi = require(codeFolder..'ProFi')
require('paths')

-- create the options
if not opts then
	cmd = torch.CmdLine()
	cmd:option('-n_hidden', 500, 'number of hidden units')
	cmd:option('-datasetsize', 'full', 'small|full size of dataset')
	cmd:option('-image_name', 'demo.png', 'the filename with which the generated image should be saved')
	opts = cmd:parse(arg or {})
end

torch.setdefaulttensortype('torch.FloatTensor')


-- The supplied MNIST images are 32x32 pixels in size.
geometry = {32,32}

-- Only load the small dataset to start with.
if opts.datasetsize == 'full' then
    trainData,valData = mnist.loadTrainAndValSet(geometry,'none')
    testData = mnist.loadTestSet(nbTestingPatches, geometry)
else
	dataSize = 2000
	trainData = mnist.loadTrainSet(dataSize, geometry,'none')
	testData = mnist.loadTestSet(dataSize/2, geometry)
	valData = mnist.loadTestSet(dataSize/2, geometry)
end

-- The datasets need to be converted probabilities
trainData:toProbability()
valData:toProbability()
testData:toProbability()

-- Create the rbm
rbm = rbmsetup(opts, trainData)

-- Train the rbm
rbm = rbmtrain(rbm,trainData,valData)

-- Output the rbm weights
create_weight_image(rbm, geometry, opts.image_name)