-- visualisation function for the rbm training process.

require('image')

function create_weight_image(rbm, image_dimensions, filename)
	-- Create an image from the weights of the current rbm.

	-- print(rbm)
	-- print(image_dimensions)
	w = image_dimensions[1]
	h = image_dimensions[2]
	assert(rbm.W:size(2) == w*h)

	-- TODO: This may be the same as rbm.n_hidden
	n_filters = rbm.W:size(1)
	n_channels = 1

	pad = 1
	nrows = math.ceil(math.sqrt(n_filters))

	local weight = rbm.W:view(n_filters, n_channels, w, h)
    local filters = image.toDisplayTensor{input=weight, padding=pad,
    	nrow=nrows, scaleeach=true, symmetric=false}

    image.save(filename, filters)

end