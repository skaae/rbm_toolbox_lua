#!/bin/bash
# Use the following image name (this should be available on DockerHub)
IMAGE_NAME="kkoba84/itorch-docker"

# Pull the required docker image from DockerHub.
docker pull $IMAGE_NAME

# Use the following path in the container.
CONTAINER_PATH=/root/workspace/rbm_toolbox_lua

# Launch the container.
docker run -w "$CONTAINER_PATH" -p 9999:9999 -v `pwd`:$CONTAINER_PATH -it $IMAGE_NAME bash

# The following would start the itorch server and connect to the container.
# It is disabled by default.
if false
then
	# Start the container with with the iTorch server
	docker run -w "$CONTAINER_PATH" -d -p 9999:9999 -v `pwd`:$CONTAINER_PATH -it $IMAGE_NAME /root/torch/install/bin/itorch notebook --profile=itorch_svr
	# Get the container ID from docker ps
	CONAINER_ID=`docker ps -l -n=1 -a | grep -oE "^\S+\s\s"`
	# Connect to the running container
	docker exec -ti $CONAINER_ID "bash"
fi

