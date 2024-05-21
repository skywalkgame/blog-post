---
type: docs
bookToc: True
weight: 1
---
# **Accelerating Transformers via Conditional Computation: As Aspect of Mixture of Depths**
*Posted by: Inkwan Hwang, Minjae Park*

## Introduction
“Choice and concentration” is an effective strategies for achieving success in problems. Sometimes, it is not necessary to consume same amount of effort and time into all problems. Expending energy on trivial issues may fail to concentrate on what truly matters. Similarly, in language models, there is a technique that does not focus equally on all tokens but allocates less budget to non-essential tokens. This technique is called conditional computation.

In this post, We will explain conditional computation strategies for Transformers, focusing on a technology announced this year called **Mixture-of-Texture.**


paper:  [<U>Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</U>](https://arxiv.org/abs/2404.02258)


Let's dive in!

## Understanding the challenge: Uniform computation in Transformers

## Conditional Computation for transformers
- Early exiting
  
  Early Exit method is a method when the model decides to end computation on a given token, allowing it skips the remaining layers. Difference between MoD is, MoD can choose whether skip middle layer or not, but Early Exit method can't.
- CoLT5

- Mixture of Experts (MoE)

  MoE is an model which consists of parallel expert models which is fitted to certain domains. Like MoD, token-level routing decisions are made across the network depth. Difference between MoD is, MoD chooses path to transformer or to residual connection, MoE chooses path to transformer(Expert) or to transformer(Expert) or both.
  
## Overview to Mixture-of-Depths (MoD)

<p align="center">
    <img src=./Mixture-of-Depths.png> 
</p>

MoE is an model which consists of parallel expert models which is fitted to certain domains.
Like MoD, token-level routing decisions are made across the network depth.
Difference between MoD is, MoD chooses path to transformer or to residual connection, MoE chooses path to transformer(Expert) or to transformer(Expert) or both.

## Capacity based routing schemes
- Token-choice routing
- Expert-choice routing
- Expert-choice MoD

## Implementation detail

논문 요약

## Open source MoD

https://github.com/astramind-ai/Mixture-of-depths

## Conclusion and discussion

논문 요약 + 내 생각

## Some resources

참고문헌 정리
