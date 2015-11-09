#!/bin/bash
IMAGE_NAME=itorch

docker run -d -p 9999:9999 -v `pwd`:/root/mount -it $IMAGE_NAME /root/torch/install/bin/itorch notebook --profile=itorch_svr

