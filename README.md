## Weakly Supervised Learning Guided by Activation Mapping Applied to a Novel Citrus Pest Benchmark

This [work](https://arxiv.org/pdf/2004.11252.pdf) was accepted by The Agriculture-Vision Workshop in CVPR 2020.

The code content here is an example of how to use Grad-CAM to produce Instances for a Multiple Instance Learning Method. It uses the [CPD](https://github.com/edsonbollis/Citrus-Pest-Benchmark) dataset in the training process. The Weakly Supervised
Multi-Instance Learning code presents a way to classify tiny regions of interest (ROIs) through a Convolutional Neural Network, a Selection Strategy Based on
Saliency Maps (Patch-SaliMap) and a Weighted Evaluation
Method.

![Mite Images](https://github.com/edsonbollis/Weakly-Supervised-Learning-Citrus-Pest-Benchmark/blob/master/mites.png)

Our method consists of four steps. In Step 1, we train a CNN (initially trained on the ImageNet) on the Citrus
Pest Benchmark. In Step 2, we automatically generate multiple patches regarding saliency maps. In Step 3, we fine-tune our
CNN model (trained on the target task) according to a multiple instance learning approach. In Step 4, we apply a weighted
evaluation scheme to predict the image class.



### Additional Information
Please cite it as
```
@article{bollis2020weakly,
  title={Weakly Supervised Learning Guided by Activation Mapping Applied to a Novel Citrus Pest Benchmark},
  author={Bollis, Edson and Pedrini, Helio and Avila, Sandra},
  journal={arXiv preprint arXiv:2004.11252},
  year={2020}
}
```

ATTN: This code is free for academic usage. For other purposes, please contact Edson Bollis (edsonbollis@gmail.com).
