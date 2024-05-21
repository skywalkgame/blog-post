---
type: docs
bookToc: True
weight: 1
---

# Megalodon: Efficient LLM Pretraining and Inference with Unlimited Context Length
Authors: Xuezhe Ma, Xiaomeng Yang, Wenhan Xiong, Beidi Chen, Lili Yu, Hao Zhang, Jonathan May, Luke Zettlemoyer, Omer Levy, Chunting Zhou
Reviewer: Hyunho Kook

{{< figure src="./Megalodon.jpg" alt="." width="600" height="600" >}}

## Introduction
Recently, Large Language Models (LLMs) have been gaining popularity. The impressive performance and versatility demonstrated by large models above a certain level have started to be utilized in various fields. However, as the size of the models grows, the size of the data that the models are expected to process is also increasing. Examples of this include processing currently open issues by inputting a GitHub repository or translating a large volume of books without losing context. In addition, the ability to maintain context and carry on a conversation for an extended period within a single chat is also sometimes required. The transformer, which is the foundation model of modern LLMs, exhibits vulnerabilities in this regard. Firstly, since it uses KV cache, memory usage increases rapidly as the sequence length grows, and it has a computational complexity proportional to the square of the sequence length.

To address this problem, the authors propose a method that inherits and advances MEGA (exponential moving average with gated attention), the predecessor of this paper. The overall contributions are as follows:

1. **CEMA (Extending Multi-dimensional Damped EMA to Complex Domain)**, an extension of Exponential Moving Average (EMA) to the complex domain, is proposed.
2. **Timestep Normalization, an extension of Group Norm to the timestep domain**, is proposed as an alternative to Layer Norm.
3. **Normalized Attention**, which performs normalization during attention computation, is proposed.
4. **2-hop Residual Connection**, which composes residual connections in 2-hop units, is proposed.

By employing these methods, the authors have created a transformer architecture that is linear with respect to context length. They have also addressed the issues encountered in the previous research, MEGA, which were (i) low performance and (ii) the need for different architecture structures for each data type or task.

## MEGA (exponential Moving avErage with Gated Attention)
