---
author: ["Minsung Cho"]
title: "StyleGAN : A Style-Based Generator Architecture for Generative Adversarial Networks"
date: "2024-11-01"
description: "Understanding StyleGAN, their causes, and potential ways to mitigate them."
summary: "StyleGAN"
tags: ["AI", "Machine Learning", "Generative models"]
categories: ["Review Paper", "AI"]
ShowToc: true
TocOpen: true


draft: false
hidemeta: false
comments: false
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
editPost:
    URL: "https://arxiv.org/abs/1812.04948"
    Text: "StyleGAN" # edit text
    appendFilePath: true # to append file path to Edit link
---








## Introduction

The motivation for creating StyleGAN started from some issues with GANs. While GANs have a structure that can generate high-quality images, they face big challenges in controlling detailed styles or elements of the image. For example, in previous GAN models, even if they generate a person’s face, there were many limits in adjusting small details or styles of that face .

StyleGAN aims to overcome these limits of regular GAN models and achieve both control over image generation and high-quality, detailed images. StyleGAN’s goal is to create images that are more realistic and controllable. To achieve this, StyleGAN's design borrwed ideas from style transfer literature and added mechanisms to mix different styles.




## Basic Concepts of GANs and Limitations of Existing Models


To better understand StyleGAN, this section will provide a brief explanation of its predecessors, GAN, Deep Convolutional GAN(DCGAN), and Progressive Growing GAN (PGGAN)

### GAN and Its Basic Structure

GANs have a structure of two networks, the Generator and the Discriminator ,which compete and improve together. The Generator takes random noise as input to create fake images, while the Discriminator tries to distinguish between real and fake images. Through their Adverisal interaction in training, the Generator gradually improves and creates images that look more and more real.

![GAN structure](/image/stylegan/ganstructure.png)
*Fig. 1. Architecture of a generative adversarial network. (Image source: [www.kdnuggets.com/2017/01/generative-...-learning.html](https://www.kdnuggets.com/2017/01/generative-adversarial-networks-hot-topic-machine-learning.html))*

For details, these two models compete with each other in training process. G(generator) tries to trick the discriminator. Otherwise D(discriminator) works hard not to be fooled.

In other words, D and G are playing a **minmax** game to optimize the following loss function: 
![Loss function](/image/stylegan/minmaxgame.png)


### Deep Convolutional GAN (DCGAN)

DCGAN is a model that starts with a latent vector and progressively upsamples it through a convolutional network. This approach has improved performance in the image domain. By using transposed convolutions, DCGAN effectively increases the image resolution, step by step. With this structure, vector arithmetic became possible, leading to advancements in GANs that aimed to control semantic information within images.

![DCGAN structure](/image/stylegan/dcganstructure.png)
*Fig. 2. Architecture of a DCGAN. (Image source: [https://www.researchgate.net/figure/DCGAN-Deep-Convolutional-Generative-Adversarial-Network-generator.....](https://www.researchgate.net/figure/DCGAN-Deep-Convolutional-Generative-Adversarial-Network-generator-used-for-LSUN_fig1_340884113))*


### Progressive Growing GAN (PGGAN)

Progressive Growing GAN, the predecessor of StyleGAN, is a model that improves GAN training stability by sequentially adding layers to gradually increase the image resolution. Instead of creating high-resolution images all at once, this method ensures stability by incrementally adding layers during training.

![PGGAN structure](/image/stylegan/pggan.png)
*Fig. 3. Architecture of a PGGAN. (Image source: [https://python.plainenglish.io/a-friendly-introduction-to-generative-adversarial-networks-gans-101f8de8d3b6]*)

PGGAN applied WGAN-GP Loss to improve training stability, making it possible to generate high-resolution images. However, it had limitations in controlling specific details or styles within the images.


## Main Idea of StyleGAN

These are brief sketches of the main idea.

- **Mapping Network**: Instead of directly using the z-vector sampled from the Gaussian distribution, it is mapped to the w domain for use.

- **Constant input** : Start with 4x4x512 tensor rather than later vector

- **AdaIN**: Using AdaIN, which showed effective performance in feed-forward style transfer networks.  We can extract and apply style information from other selected data.



- **Stochastic Variation**: 



- **Style Mixing** : 

### Mapping Network

![Mapping Network](/image/stylegan/mappingnetwork.png)
*Fig. 4. Mapping Network for StyleGAN. (Image source: [ Karras, T., Laine, S., & Aila, T. (2019). "A Style-Based Generator Architecture for Generative Adversarial Networks". IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4401–4410.]*)

Let’s assume that the distribution of the training set is like (a) in Figure 4, where there is almost no data in the top left part of the distribution. Since z is sampled from a Gaussian distribution, it is simillar to sampling from a spherical shape. When we look at (b) in Figure 4, 
we can see that in the top left of the sphere, **rapid changes in style can easily occur** during the interpolation process. Also, we can observe that the factors of variation are not linear, and we refer to this as being **entangled**.

In StyleGAN, instead of using the z vector directly, it is mapped to the w space, where the w vector is then used. In the W space, the factors of variation become more linear ((c) in Figure4), because they no longer need to follow a specific distribution.

![Mapping Network](/image/stylegan/mappingnetwork2.png#center)
*Fig. 5. Mapping Network for StyleGAN. (Image source: [ Karras, T., Laine, S., & Aila, T. (2019). "A Style-Based Generator Architecture for Generative Adversarial Networks". IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4401–4410.]*)


### Constant input

The style-based generator takes w as input, so unlike PGGAN or other GANs, it no longer requires convolution operations on z. As a result, the **synthesis network starts with a 4x4x512 constant tensor**. Starting from a constant rather than latent vector has proven to yield better performance, a result confirmed through empirical observation.


### AdaIN

This method allows for adding multiple style details as the layers progress, creating more diverse images. Since style information is taken from other selected data, there are **no parameters to train**.

It is used to process the result of the convolution operation. Let’s assume that the convolution output is a tensor made up of n channels, and each channel feature is represented by x_i. The latent w undergoes an affine transformation to produce y_{s,i} and y_{b,i}. These values are used to **scale and add bias** to the normalized x_i. So the mean and variance are modified. This can also be seen as a type of style transfer.

![AdaIN](/image/stylegan/adain.png#center)
*Fig. 6. How AdaIN works in Style GAN. (Image source: [https://velog.ioStyleGAN-A-Style-Based-Generator-Architecture-for-Generative-Adversarial-Networks.....](https://velog.io/@minjung-s/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0StyleGAN-A-Style-Based-Generator-Architecture-for-Generative-Adversarial-Networks))*





### Stochastic Variation






### Style Mixing




## Disentanglement Studies

**There** are ongoing efforts to reduce hallucinations in AI systems, mainly focusing on data and model improvement.

### Perceptual path length

One approach is to enhance data quality and use fact-verification systems that validate the model's output. Below is a sample Python function for verifying the accuracy of a statement using a hypothetical fact-checking API.

The SAFE evaluation metric is **F1 @ K**. The motivation is that model response for **long**-form factuality should ideally hit both precision and recall, as the response should be both

- *factual*: measured by precision, the percentage of supported facts among all facts in the entire response.
- *long*: measured by recall, the percentage of provided facts among all relevant facts that should appear in the response. Therefore we want to consider the number of supported facts up to *K*.

Given the model response *y*, the metric **F1 @ K** is defined as:


### Linear separability



## FFHQ DATASET



___


SADLY
