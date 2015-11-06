-- visualisation function for the rbm training process.


require('image')

function create_weight_image(rbm, image_dimensions)
	-- Create an image from the weights of the current rbm.

	-- print(rbm)
	-- print(image_dimensions)

	w = image_dimensions[1]
	h = image_dimensions[2]
	assert(rbm.W:size(2) == w*h)

	-- TODO: This may be the same as rbm.n_hidden
	n_filters = rbm.W:size(1)

	-- For each filter we need a tile that has the required images dimensions
	filter_tile = torch.Tensor(w,h)

	get_filter_tile( rbm, 1, 1, filter_tile)


	dd = image.toDisplayTensor{input=filter_tile}
	image.save('demo.jpg', dd)
end

function get_filter_tile( rbm, i, N, filter_tile )
	-- Copy the i-th filter to the filter tile.

	if i <= N then
		filter_tile = filter_tile:copy(rbm.W[i])
	else
		min_val = rbm.W[i]:min()
		filter_tile = filter_tile:ones():mul(min_val)
	end

end