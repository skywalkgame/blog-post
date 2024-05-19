---
type: docs
bookToc: True
weight: 1
---
# Spectrally Pruned Gaussian Fields with Neural Compensation (SUNDAE)
*Authors: Yang, Runyi, et al

## Summary
<p align="center">
    <img src='fig1.png' width="600">
</p>
<p align="center">
    Fig1. Comparison of 3D gaussian splatting and proposed SUNDAE
</p>
Conventional 3D Gaussian Splatting
- Pros: Superior rendering speed and quality
- Cons: High memory consumption

Proposed SUNDAE
- It constructs a memory-efficient Gaussian field using spectral pruning and neural compensation. 
- It considers the relationship between primitives, reducing memory usage while maintaining rendering quality.
- It significantly reduces memory consumption while preserving high rendering quality.
- Code: https://runyiyang.github.io/projects/SUNDAE/.

## Introduction
<p align="center">
    <img src='fig2.png' width="600">
</p>
<p align="center">
    Fig2. Conceptual illustration of vanilla 3D gaussian splatting, SUNDAE spectral pruning technique, and neural compensation.
</p>
### Spectral graph pruning
Gaussian fields utilize a collection of Gaussian primitives as the representation of the scene. As these primitives are irregularly distributed in 3D space, we propose a graph-based data structure, rather than regular structures like grids, to capture the relationship between these primitives (middle panel of Fig. 2).


### Neural compensation
To address an inevitable decrease in rendering quality, they employ a neural compensation head to compensate for this quality loss (right panel of Fig. 2).

Contributions
- A newly proposed primitive pruning framework for Gaussian fields based upon the spectrum of primitive graphs;
- A novel feature splatting and mixing module to compensate for the performance drop caused by the pruning;
- State-of-the-art results, in terms of both quality and speed, on various benchmarks with low memory footprint.

### 

### Autoencoder

## Methods



## Results

## Conclusion
