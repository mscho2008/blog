---
author: ["Minsung Cho"]
title: "StyleGAN : A Style-Based Generator Architecture for Generative Adversarial Networks"
date: "2024-11-01"
description: "Understand the main idea of StyleGAN, understand the new techniques they introduced, and examine its contributions to image generation"
summary: "Introduce StyleGAN "
tags: ["AI", "Deep Learning", "Generative models"]
categories: ["Review Paper"]
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



- **Stochastic Variation**: The model is designed to effectively incorporate stochastic variation during image generation, allowing for control over various probabilistic details.


- **Style Mixing** : During training, a certain percentage of images are generated using two random latent codes instead of just one.

### Mapping Network

![Mapping Network](/image/stylegan/mappingnetwork.png#center)
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

![Stochastic Variation](/image/stylegan/noise.png#center)
*Fig. 7. Total Architecture for StyleGAN. (Image source: [ Karras, T., Laine, S., & Aila, T. (2019). "A Style-Based Generator Architecture for Generative Adversarial Networks". IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4401–4410.]*)



Even when photographing the same person, variables like wind strength or the person's condition can make probabilistic features, such as changes in hair placement or acne. To manage this type of stochastic variation, noise input undergoes an affine transformation, enabling it to influence the AdaIN layer immediately before application.

To make sure the difference between style and noise, Style influences high-level global attributes, like face shape or pose, while Noise, which controls stochastic variation, affects finer details like freckles and skin pores.

Starting from a 4x4x512 tensor, a total of 9 blocks are needed to generate a high-resolution 1024x1024 image, with each block applying style twice. This results in a total of 18 mixed styles, where the first 4 are defined as **Coarse style**, the next 4 as **Middle style**, and the remaining 10 as **Fine styles**. The coarse style mainly determines large elements such as face shape, overall hairstyle, and pose. The middle style adjusts smaller facial features, such as whether the eyes are open or closed, and other finer facial details. Finally, the fine style controls subtle details like color composition and skin texture.

![Styles](/image/stylegan/styles.png#center)
*Fig. 8. Images were generated by copying a specified subset of styles from source B and taking the rest from source A. (Image source: [ Karras, T., Laine, S., & Aila, T. (2019). "A Style-Based Generator Architecture for Generative Adversarial Networks". IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4401–4410.]*)

### Style Mixing

To make the style more localized, a certain percentage of images are generated during training using two random latent codes instead of just one. Specifically, two latent codes, z_1 and z_2  are passed through a mapping network to produce 
w_1 and w_2. When applying the style, the crossover point is chosen, and w_1 is applied before this point, while w_2 is applied after it. This regularization technique prevents the network from assuming that styles in neighboring layers are related, allowing each layer's style to be more ***Localized**.

![Styles](/image/stylegan/mixing.png#center)
*Fig. 9. FIDs in FFHQ for networks trained by enabling the mixing regularization for different percentage of training examples. (Image source: [ Karras, T., Laine, S., & Aila, T. (2019). "A Style-Based Generator Architecture for Generative Adversarial Networks". IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4401–4410.]*)

## FFHQ DATASET

Writers have introduced a new dataset of human faces called Flickr-Faces-HQ (FFHQ), which contains 70,000 high-quality images at a resolution of 1024x1024. This dataset shows much more variety than CELEBA-HQ in terms of age, ethnicity, and background. It also includes a wider range of accessories, such as eyeglasses, sunglasses, and hats.

## Evaluations

Traditionally, FID (Fréchet Inception Distance) has been used to evaluate the performance of GAN networks. Performance was measured by adding each method to the baseline step-by-step, with the following results.

![Styles](/image/stylegan/evaluation.png#center)
*Fig. 10. Fr'echet inception distance (FID) for various generator designs. (Image source: [ Karras, T., Laine, S., & Aila, T. (2019). "A Style-Based Generator Architecture for Generative Adversarial Networks". IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4401–4410.]*)





## Disentanglement Studies

To **quantitatively** measure how well the styles are disentangled, two new methods have been proposed.

### Perceptual path length

![Interpolation method](/image/stylegan/interpolation.png)
*Fig. 11. Two different Methods of Interpolation. (Image source: [https://velog.io/@su1433/StyleGan]*)

There are two main methods for interpolating latent codes: LERP and SLERP. Perceptual Path Length measures how suddenly the features change when interpolating between two latent codes. In other words, it calculates the feature distance between points t and t+ϵ. This measurement uses a pre-trained VGG network.

Since Z follows a Gaussian distribution, SLERP is used for its interpolation, On the other hand, W uses linear interpolation(LERP).


![L_Z](/image/stylegan/lz.png#center)
![L_W](/image/stylegan/lw.png#center)
*Fig. 12. Compute L_Z and l_W. (Image source: [https://velog.io/@su1433/StyleGan]*)

### Linear separability

Linear separability evaluates how linearly separable the attributes are in latent space. CELEBA-HQ dataset (where each face is labeled with 40 binary attributes such as gender) is used for training. For each attribute, 200,000 images are generated and fed into a classification network. Then, the half with the lowest confidence is removed, leaving 100,000 latent vectors. A linear SVM model is used for each attribute to predict the attribute from both z and w.

The conditional entropy H(Y∣X)is then computed, where X are the classes predicted by the SVM, and Y are the classes determined by the pre-trained classifier. The final separability score is defined as  **exp(∑i H(Y_i∣X_i))** where i represents the 40 attributes.

The table shows that StyleGAN has lower separability, which implies that **various features are well-separated, making them easier to control**. Here, the number in the method shows the depth of the mapping network.

![eval2](/image/stylegan/eval2.png#center)
*Fig. 13. The effect of a mapping network in FFHQ. (Image source: [https://velog.io/@su1433/StyleGan]*)






___

Sadly, StyleGAN has its own challenges. Even though it made big improvements in creating images, StyleGAN still has problems with training stability and computational efficiency. The model’s complex design, with its style-based layers and large mapping network, sometimes leads to long training times and high computing costs

Also, while StyleGAN tries to control specific image details well, it often overfits to certain styles during training, which can lead to decreased diversity in the images it generates. Separating style and content fully is also still hard, especially in unsupervised settings.

Some improvements, like better disentanglement and reduced computing costs, were discussed in StyleGAN2 (Karras et al., 2020).