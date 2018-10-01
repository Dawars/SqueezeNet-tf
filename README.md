# SqueezeNet-tf
A Tensorflow SqueezeNet implementation of the research https://arxiv.org/abs/1602.07360

## What is SqueezeNet?
SqueezeNet is a convolutional neural network with 80.3% top-5 accuracy on the ImageNet dataset.
This architecture aims to signifanctly reduce the size of the model.

## Goals
- [x] Building the network
- [ ] Training the network on imagenet
  * Currently published pre-trained weights cannot be used in TensorFlow, the goal of this project is to produce easy-to-use save files.
- [ ] Neural Style Transfer (http://arxiv.org/abs/1508.06576) on mobile devices
  *  Running on mobile devices in reasonable time without sacraficing quality too much like other feed-forward approaches.

![Architecture](images/architecture.png?raw=true "Architecture")
