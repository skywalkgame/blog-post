---
type: docs
bookToc: True
weight: 1
---

## Unit Scaling

Unit scaling is proposed to address the limitations of existing methods for managing scale in typical models. A model is considered unit-scaled if its activations, weights, and gradients have approximately unit variance at initialization. This is achieved by inserting scaling factors into the forward and backward passes. Unlike loss scaling, which requires an empirically determined hyperparameter or an adaptive algorithm, unit scaling determines these scales based on a set of rules for each operation, approximately preserving the variance of the inputs. This leads to global unit scaling throughout the model, ensuring tensor values are centered within the exponent range at initialization, providing headroom during training to avoid going out of range. While unit scaling does not address the adaptation of scales during training, it is expected to be sufficient to avoid numerical instability for many models, as observed in experiments. 

### A framework for scaling computational graphs

+ Computational Graphs

+ Forward and backward graphs

+ Scaled ops

+ Scaled computational graph

+ Constraint-scaled computational graphs

**Proposition 5.1**

**Theorem 5.2**

### A scaling strategy for unit variance

+ Unit scaled computational graphs

+ Selecting scaling factors

### Weighted addition

### Recipe

### Example
<p align="center">
    <img src='./Figure4.png' width="600">
</p>
<p align="center">
    Fig4. Character language modelling, showing validation bits per character over a wide range of models
</p>

## Results

+ Character language modelling

    + Experimental Setup

    + Results

+ Masked language modelling

    + Experimental Setup

    + Results

<p align="center">
    <img src='./Table2.png' width="600">
</p>
<p align="center">
    Table2. Downstream performance of regular and unit-scaled BERT models
</p>

## Related Work

**Variance scaling analysis**

**FP8 inference**

## Discussion

