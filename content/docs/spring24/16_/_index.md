---
type: docs
bookToc: True
weight: 1
---

# Mixture-of-Depths: Dynamically allocating compute in transformer-based language models
*Authors: Dativd Raposo(Google DeepMind) and Adam Santoro(Google DeepMind) et.al*


This paper insists that all problems do not require same amount of time to solve in real world and also in language models like Transformer.  
But, Transformer models spread FLOPs **'uniformly'** across input sequences, which is inefficient!  
Therefore there are many efforts like "early exiting" to reduce total FLOPs & **'dynamically'** allocate compute budgets.
However, these methods do not work well due to hardware constraints.  
So, the methods should be sophistically addressed like harmonious with current hardware stack & known tensor sizes that are selected to maximize hardware utilization.  

<p align="center">
    <img src=./Mixture-of-Depths.png> 
</p>

<p align="center">
    (Fig 1. Description of overall MoD & comparison between Vanilla Transformer & Early Exit method)
</p>
  
Therefore, the things that authors of this paper contributed in this paper are listed as follows:  

1. Suggestion of method(Mixture-of-Depths, MoD) which limits the total FLOPs by choosing only k tokens which process into Attention + mlp layer.
2. Comparing this method with Vanilla transformer(isoFLOP)
3. Comparing this method with Mixture-of-Experts(MoE) and Combine to MoDE

## Background 

#### Early Exit method

Early Exit method is an method when model decides to end computation on a given token, 

#### What is MoE?

<p align="center">
    <img src=./moe.png> 
</p>
<p align="center">
    (Fig 2. Diagram of MoE)
</p>
MoE is an model which consists of parallel expert models which is fitted to certain domains. Like MoD will 

## Implementing Mixture-of-Depth Transformers

####

## Results & Discussion

## References
