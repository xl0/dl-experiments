# dl-experiments
Experiments in Deep Learning and Neural Networks

# VGG net training / inference / experiments

- [x] Implement VGG 11/13/16/19 as in the paper.
- [x] Trainig 1
    - [x] Imagenet, imagenette, MNIST datasets
    - [x] Train / val loop
    - [x] Log metrics
- [x] Testing 1 (VGG11)
    - [x] Overfit 1 batch imagenette
    - [x] Overfit imagenette
    - [x] Overgit 1 batch ImageNet
    - [x] Overfit ImageNet
    - [ ] Overfit ImageNet with dropout
- [ ] Training 2
    - [x] W&B logging
    - [x] AMP
    - [x] Gradient Accumulation
    - [ ] Multiple GPUs
    - [ ] Data Augmentaiton as in paper
    - [ ] Inference time augmentation
- [ ] Testing 2 (VGG11)
    - [ ] Train on imagenette
    - [ ] Train on ImageNet




VGG11 can't train at all with the paper's original parameter intialization, but works fine with Glorot initializaiton that they also discovered worked better.

Runs
- Overfit on ImageNet in 6 epochs (glorot, dropout=0, wd, no augmentation):
https://wandb.ai/xl0/vgg/runs/30zipp5v






