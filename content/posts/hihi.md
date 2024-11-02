---
author: ["Lilian Weng"]
title: "Hallucination in AI Systems"
date: "2024-07-07"
description: "Understanding hallucinations in AI, their causes, and potential ways to mitigate them."
summary: "A deep dive into the phenomenon of hallucinations in AI models, exploring causes, consequences, and mitigation strategies."
tags: ["AI", "Machine Learning", "Hallucinations", "Natural Language Processing"]
categories: ["Research", "AI"]
ShowToc: true
TocOpen: true


draft: true
hidemeta: false
comments: false
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: true
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
    URL: "https://github.com/<path_to_repo>/content"
    Text: "StyleGAN" # edit text
    appendFilePath: true # to append file path to Edit link
---








## Introduction

Large language models (LLMs) like GPT-4 have demonstrated remarkable abilities in generating text that resembles human language, but they also exhibit a concerning behavior: **hallucination**. Hallucination occurs when an AI model generates information that sounds plausible but is actually incorrect or fabricated. Understanding the causes and solutions to hallucination is crucial as AI models become more integrated into critical applications.

<!--more-->

## copy?

Hallucination in AI refers to instances where a model produces responses not based on factual data or real-world knowledge. For example, when asked about a fictional event, an AI might fabricate a plausible-sounding answer rather than admitting it doesn't know. This can mislead users, especially in high-stakes fields like healthcare or law.

### Example Image of Hallucination

![Example of Hallucination](assets/image/structure.png)


In this image, the AI model confidently provides an answer to a question based on nonexistent information, illustrating a typical hallucination case.

## Causes of Hallucination

There are several reasons why AI models hallucinate:

1. **Training Data Quality**: If the model’s training data is incomplete or inaccurate, it may lack the necessary information to generate factually correct responses.
2. **Over-reliance on Pattern Recognition**: Models are optimized to generate coherent and contextually appropriate responses rather than true or accurate ones.
3. **Bias in Training Data**: If the training data contains biases, the model may produce biased or incorrect information, perpetuating existing inaccuracies.



![ffsfs](/image/structure.png)
*Fig. 3. The evaluation framework for the FactualityPrompt benchmark. (Image source: Lee, et al. 2022)*






This graphic represents different sources of hallucination in AI models, illustrating how model architecture and data quality play a role.

## Strategies for Mitigating Hallucination

**There** are ongoing efforts to reduce hallucinations in AI systems, mainly focusing on data and model improvement.

### Data Augmentation and Fact Verification

One approach is to enhance data quality and use fact-verification systems that validate the model's output. Below is a sample Python function for verifying the accuracy of a statement using a hypothetical fact-checking API.

The SAFE evaluation metric is **F1 @ K**. The motivation is that model response for **long**-form factuality should ideally hit both precision and recall, as the response should be both

- *factual*: measured by precision, the percentage of supported facts among all facts in the entire response.
- *long*: measured by recall, the percentage of provided facts among all relevant facts that should appear in the response. Therefore we want to consider the number of supported facts up to *K*.

Given the model response *y*, the metric **F1 @ K** is defined as:
