---
type: docs
bookToc: True
weight: 1
---
# **Is Bigger Edit Batch Size Always Better? - An Empirical Study on Model Editing with Llama-3**
*Authors: Junsang Yoon, Akshat Gupta, Gopala Anumanchipalli*

*Posted by Jin Hyun, Gyuhyun Jung*

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
  {{< katex >}}
    \argmin_{\hat{W}} \left\| \hat{W} K_0 - W_0 K_0 \right\| \quad \text{s.t.} \quad \hat{W} k_e = v_e \\Preservation\_term=\left\| \hat{W} K_0 - W_0 K_0 \right\| \\ Memorization\_term=\hat{W} k_e = v_e 
  {{< /katex >}} 
</p>
    
Where *W* represents the **weights** of the **feedforward layer** we want to edit, *k* is a **key-vector** representative of a fact, v_e is the desired output, and K0 =[k 01 |k 02 |...| k 0N ] is a matrix consisting of facts we want to preserve. Above equation is optimized by follwing gradient.

<p align="center">
  {{< katex >}}
\hat{W} = W_0 + \Delta \quad \text{where} \\
\Delta = (v_e - W_0 k_e) \frac{k_e^T C_0^{-1}}{k_e^T C_0^{-1} k_e}
  {{< /katex >}} 
</p>

For MEMIT model editing. it optimizes same objectives with ROME, but performance memorization using a least-square constraint, which allows for a closed-form solution. It has similar form with ROME method, but it multiplies \lambda term, which is hyperparameter, to preservation term. Also, it combines memorization term for minimize target

<p align="center">
  {{< katex >}}
\argmin_{\hat{W}} \lambda\left\| \hat{W} K_0 - W_0 K_0 \right\| + \left\| \hat{W} K_E - V_E \right\|\\Preservation\_term=\lambda\left\| \hat{W} K_0 - W_0 K_0 \right\| \\ Memorization\_term=\hat{W} K_E - V_E 
  {{< /katex >}} 
</p>

VE is stacked matrix of ve vectors, and fact is represented by a pair of vectors denoted as *key* (ke) and *value* (ve). This objective has similar solution of ROME, followed by below equations.

<p align="center">
  {{< katex >}}
\hat{W} = W_0 + \Delta \quad \text{where} \\
\Delta = (V_E - W_0 K_R)K^T_E (\lambda C_0 + K_E^T K_E^T)^{-1}
  {{< /katex >}} 
</p>

In EMMET, it shows model editing is possible with batched facts. It is possible by allowing memorization happens using an equality-constraint. EMMET objective and gradient solution is followed by below equations.

<p align="center">
  {{< katex >}}
\argmin_{\hat{W}} \left\| \hat{W} K_0 - W_0 K_0 \right\|\quad \text{s.t.} \hat{W} k_i^e = v_i^e \quad \forall i \in [1, 2, \cdots, E] \\Preservation\_term=\left\| \hat{W} K_0 - W_0 K_0 \right\| \\ Memorization\_term=\hat{W} k_i^e = v_i^e \quad \forall i \in [1, 2, \cdots, E] \\ \hat{W} = W_0 + \Delta \quad \text{where} \\
\Delta = (V_E - W_0 K_R)(K_E^T C_0^{-1}K_E)^{-1}K_E^TC_0^{-1}
  {{< /katex >}} 
</p>
    
### How model editing performance is estimated?
Model performance is estimated with 4 main scores, and they are denoted as follow.
#### __Efficacy Score (ES)__ 
It measures if the new fact, we want to edit, is successfully edited to model. It is measured by percentage where P(new fact) > P(old fact) for query prompt. {{< katex >}}\mathbb{E}_i \left[ \mathbb{P}_G \left[ o_i \mid p(s_i, r_i) \right] \right] > \mathbb{P}_G \left[ o_i^c \mid p(s_i, r_i) \right] {{ /katex }}
#### __Paraphrase Score (PS)__
It measures model's ability to generalize following an edit. It is measured by where P(new fact) > P(old fact) under paraphrases of the query prompt.

#### __Neighborhood Score (NS)__
It represents the locality of model editing. It measures the impact of edit process on adjacent stored facts within the model. It quantifies the percentage of nearby facts that remain unchanged after edit.

#### __Composite Score (S)__
It combines aspect of edit success, generalization, and locality. It is calculated as the harmonic mean of Edit Success (ES), Paraphrase Score (PS), and Neighborhood Score (NS). It provies overall efficacy of model edits.
