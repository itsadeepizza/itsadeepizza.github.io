---
layout: page
title: Spiking Neural Network
subtitle: Implement a SNN to recognize digits in the MNIST dataset from scratch
---
**Github**: [github.com/itsadeepizza/SNN](https://github.com/itsadeepizza/SNN)

<p align="center">
<img src="//imgs.xkcd.com/comics/trained_a_neural_net.png" title="It also works for anything you teach someone else to do. &quot;Oh yeah, I trained a pair of neural nets, Emily and Kevin, to respond to support tickets.&quot;" alt="Trained a Neural Net" srcset="//imgs.xkcd.com/comics/trained_a_neural_net_2x.png 2x" style="image-orientation:none">
</p>



## Pros and Cons of Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) is a cornerstone in the field of artificial neural networks (ANNs).

In essence, an ANN is a complex nonlinear function shaped by a large set of parameters, namely weights and biases. These parameters are tuned iteratively by computing the gradient of the loss function at each step, adjusting them to minimize this loss.

The core concept is akin to Newton's method of tangents:

![NewtonIteration_Ani.gif](images%2FNewtonIteration_Ani.gif)

Despite its wide success, SGD is not without its challenges:

- Vanishing/exploding gradients: Despite numerous techniques developed to combat this issue, it remains a significant hurdle. Often, the resolution involves experimenting with different hyperparameters, normalization techniques, or simplifying the network's architecture.

- The use of floating-point parameters: Beyond the inherent limited precision, which contributes to the vanishing gradient problem, floats require considerable space for storage and computation. This is particularly apparent when considering that, for inference tasks, it's often feasible to quantize these parameters, greatly reducing their size. Nonetheless, floats are currently indispensable for SGD.

This leads us to ponder: **could there be an ANN training methodology that doesn't rely on SGD?**

That seems very likely, considering it's improbable that the neurons in our brain compute derivatives while learning.


## An Alternative Approach: Spiking Neural Networks

Drawing inspiration from the brain, spiking neural networks (SNNs) have been developed. These networks aim to replicate the functioning of real neurons as closely as possible.

So, what training mechanism could replace SGD in SNNs?

The field has put forth various methods. SNNs, although a niche topic, have demonstrated a wide variety of architectures and implementations. This is in stark contrast to the more standardized world of conventional ANNs, likely because SNNs haven't yet shown the same level of effectiveness, prompting continuous exploration.

The main training techniques for SNNs are:

**Surrogate Gradient Method**
A straightforward approach is to adapt SGD for SNNs, which typically have non-differentiable activation functions. While effective, this method might not be as groundbreaking as others.

**STDP-based Training (R-STDP, BP-STDP)**
This method is more akin to brain neuron functionality. It involves detecting correlations between spikes across consecutive neural layers, modifying the weights to encourage or discourage subsequent spikes. This method leverages the multiple time steps in the inference process, where neurons can spike variably.

## A Personal Insight: Causality as an SGD Alternative

What I present now is a personal insight, which should be taken at face value, as I cannot vouch for its absolute correctness.

Few algorithms can effectively replace gradient descent in minimizing functions, particularly within the ANN framework.

This is mainly because for optimal parameter selection, it's essential to discern how each influences the loss calculation, guiding their adjustment. Gradient computation offers this insight.

But if gradient computation is off the table, what else is there?

Herein lies the intrigue with SNNs: they introduce time as a novel element. Adding this dimension provides a radically different avenue to ascertain a parameter's impact on network output.

With STDP, time enables the establishment of causality between a neuron's spike in intermediate layers and the network's overall output. Intriguingly, STDP's workings bear a resemblance to the concept of [Granger causality](https://en.wikipedia.org/wiki/Granger_causality)

This approach is particularly promising for its innovation and potential flexibility compared to traditional gradient descent algorithms. 

The future of STDP and the advancements it may lead to remain an exciting prospect.


## Our Code

Our objective was to tackle some of the classical machine learning (ML) problems through the use of spiking neural networks:

- The **iris dataset**, to gain familiarity with the technique.
- The **MNIST dataset**, to validate our approach.

We also made a decision to implement our solution without the use of specific libraries, instead exclusively utilizing `numpy` and `cython`.

### R-STDP

Our implementation of R-STDP was mainly based on [SVRS] (particularly for Gaussian encoding) and [BBJHCK] (for reinforcement aspects). We did, however, introduce some changes, such as opting not to use the  eligibility trace.

Updating the weights presented some difficulties due to the computationally intensive nature of the operation, prompting us to carry out this process in cython (`fastspike/fastspike.py`).

Our deployment of R-STDP on the Iris dataset (`spiking_iris_R_STDP.py`) proved effective, as we achieved an accuracy of over 90%, which is consistent with the results from more traditional methods given the dataset's limited size.

Attempts to generalize R-STDP with various hidden layers did not yield successful outcomes.

### Spiking Encoding

The remainder of our code is aimed at digit recognition on the MNIST dataset (https://deepai.org/dataset/mnist). We began by focusing on encoding, experimenting with several methods:

- An encoding technique borrowed from [VCS].
- A more straightforward encoding method where the likelihood of each pixel firing a spike is proportional to its intensity.
- A hybrid approach, which leverages the technique mentioned in the paper but omits the initial feature extraction stage.

### BP-STDP

To modify R-STDP for use with multiple hidden layers, we implemented BP-STDP following the methodology outlined by [TM].

Our application of BP-STDP on the Iris dataset (`BP_STDP_iris.py`) seems to perform better than R-STDP, with improved convergence in a shorter timeframe.

However, the results on the MNIST dataset (`BP_STDP_mnist.py`) are far from the state of the art.

Despite the use of different encoding strategies and hyperparameter tuning attempts with optuna (`BP_STDP_mnist.py`), the highest accuracy achieved remained at 50%.




## Bibliography

[SVRS] A. Sboev, D. Vlasov, R. Rybka, A. Serenko: *Solving a classification task by spiking neurons with STDP and temporal coding* Procedia Computer Science, Vol 123,
pp 494-500, 2018
[https://www.sciencedirect.com/science/article/pii/S1877050918300760](https://www.sciencedirect.com/science/article/pii/S1877050918300760)

[BBJHCK] Z. Bing, I. Baumann, Z. Jiang, K. Huang, C. Cai, A. Knol: *Supervised Learning in SNN via Reward-Modulated Spike-Timing-Dependent Plasticity for a Target Reaching Vehicle*, Frontiers in Neurorobotics, Vol. 13, 2019
[https://www.frontiersin.org/articles/10.3389/fnbot.2019.00018](https://www.frontiersin.org/articles/10.3389/fnbot.2019.00018)



[VCS] R. Vaila, J. Chiasson, V. Saxena: *Deep Convolutional Spiking Neural Networks for Image Classification*, 2019
[https://arxiv.org/abs/1903.12272](https://arxiv.org/abs/1903.12272)

[TM] A. Tavanaei, A. S. Maida: *BP-STDP: Approximating Backpropagation using Spike Timing Dependent Plasticity*, 2017
[https://arxiv.org/abs/1711.04214](https://arxiv.org/abs/1711.04214)




