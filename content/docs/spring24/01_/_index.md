---
type: docs
bookToc: True
weight: 1
---
# **Is Bigger Edit Batch Size Always Better? - An Empirical Study on Model Editing with Llama-3**
*Authors: Junsang Yoon, Akshat Gupta, Gopala Anumanchipalli*

## Background
### What is __model editing__?

<p align="center">
  <img src="./model_editing.PNG" alt="." width="500" height="300" > 
</p>

<p align="center">
  Fig 1. Concept of model editing.[Editing Large Language Models: Problems, Methods, and Opportunities](https://arxiv.org/pdf/2305.13172)
</p>

The rapidly evolving field of artificial intelligence faces the challenge of keeping large language models (LLMs) up-to-date with new information, as traditional retraining methods are time-consuming and resource-intensive. As shown in figure, an alternative is __model editing__ proposed in [(Sinitsin et al., 2020)](https://openreview.net/pdf?id=HJedXaEtvS). It enables data-efficient alterations to the behavior of models.

<p align="center">
  <img src="./memit_concept.PNG" alt="." width="450" height="220" >
</p>

<p align="center">
  Fig 2. Example of model editing in case of MEMIT.
</p>

Model editing modifies stored facts within a model and corrects inaccuracies without retraining. Techniques such as __ROME__ (Rank-One Model Editing) [(Meng et al., 2022a)](https://arxiv.org/pdf/2202.05262), __MEMIT__ (Mass Editing Memory in Transformer) [(Meng et al., 2022b)](https://arxiv.org/pdf/2210.07229), and __EMMET__ (Equality-constrained Mass Model Editing algorithm for Transformers) [(Gupta et al., 2024)](https://arxiv.org/pdf/2401.07453), known as "locate-and-edit" algorithms, have emerged to optimize the preservation-memorization (PM) objective. These methods __directly modify__ specific areas of the model and are applicable to any transformer-based LLMs, offering a more efficient way to update models without retraining.

### How model editing works?
For a relation __(s, r, o)__ expressed as a tuple in the form of __(subject, relation, object)__. In model editing, we aim to update the memory of the existing model with new facts by learning about a new object __(s, r, o*)__. Model editing directly reform the weight by objective function, called the preservation-memorization objective. This objective consists of two parts, a __preservation term__ and a __memorization term__. Below equation shows how ROME works with preservation term and memorization term.

<p align="center">
  {{< katex display=true >}}
    \operatorname{argmin}_{\hat{W}} =\alpha \cdot X W
  {{< /katex >}} 
</p>
### How model editing performance is estimated?

This work presents a detailed guide for using PM-objective-based model editing methods on the newly released Llama-3. The process involves making edits on all Llama-3-8b layers to identify the optimal layer for balancing editing accuracy and preserving existing knowledge. Once identified, single-layer editing experiments using ROME, MEMIT, and EMMET are performed. The study explores three types of edits: singular edits (editing one fact at a time), batched edits (updating multiple facts simultaneously), and sequential-batched edits (updating batches of facts sequentially on the same model). This approach aims to optimize the editing process without degrading the model's performance, addressing concerns from previous research about the adverse effects of increasing edit batch size.

The study compares batched model editing with sequential-batched editing and finds that for Llama-3, sequential-batched editing with a batch size of 1024 offers optimal scaling performance. This method outperforms simple batched edits or sequential-batched edits with smaller batch sizes, highlighting the importance of sequential model editing for large-scale model updates. Additionally, sequential model editing aligns with the continual learning paradigm, providing baseline experiments and transparent procedures for future research on Llama-3 models. This work establishes benchmarks and guidelines for effectively editing models while maintaining their integrity.
