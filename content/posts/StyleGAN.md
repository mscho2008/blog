---
author: ["Minsung Cho"]
title: "StyleGAN : A Style-Based Generator Architecture for Generative Adversarial Networks"
date: "2024-11-01"
description: "Understanding StyleGAN, their causes, and potential ways to mitigate them."
summary: "A deep dive into the phenomenon of hallucinations in AI models, exploring causes, consequences, and mitigation strategies."
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

To better understand StyleGAN, this section will provide a brief explanation of its predecessors, GAN and Progressive Growing GAN (PGGAN) .....

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

There are several reasons why AI models hallucinate:

1. **Training Data Quality**: If the model’s training data is incomplete or inaccurate, it may lack the necessary information to generate factually correct responses.
2. **Over-reliance on Pattern Recognition**: Models are optimized to generate coherent and contextually appropriate responses rather than true or accurate ones.
3. **Bias in Training Data**: If the training data contains biases, the model may produce biased or incorrect information, perpetuating existing inaccuracies.


### Mapping Network



### AdaIN



### Stochastic Variation

This graphic represents different sources of hallucination in AI models, illustrating how model architecture and data quality play a role.




### Style Mixing




## Strategies for Mitigating Hallucination

**There** are ongoing efforts to reduce hallucinations in AI systems, mainly focusing on data and model improvement.

### Data Augmentation and Fact Verification

One approach is to enhance data quality and use fact-verification systems that validate the model's output. Below is a sample Python function for verifying the accuracy of a statement using a hypothetical fact-checking API.

The SAFE evaluation metric is **F1 @ K**. The motivation is that model response for **long**-form factuality should ideally hit both precision and recall, as the response should be both

- *factual*: measured by precision, the percentage of supported facts among all facts in the entire response.
- *long*: measured by recall, the percentage of provided facts among all relevant facts that should appear in the response. Therefore we want to consider the number of supported facts up to *K*.

Given the model response *y*, the metric **F1 @ K** is defined as:
