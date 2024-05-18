---
type: docs
bookToc: True
weight: 1
---

# **Mixture-of-Depths: Dynamically allocating compute in transformer-based language models**
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
2. Comparing this method with vanilla Transformer(isoFLOP) &  Mixture-of-Experts(MoE) & Combined version of MoE + MoD = MoDE
3. With this method, MoD achieves better performance than vanilla Transformer in isoFLOPs & faster inference speed.

## **Background**

#### Early Exit method

<p align="center">
    <img src=./earlyexit.png> 
</p>
<p align="center">
    (Fig 2. Early Exit method)
</p>

Early Exit method is a method when the model decides to end computation on a given token, then model skips the remaining layers.  
Difference between MoD is, MoD can choose whether skip middle layer or not, but Early Exit method can't.

#### What is MoE?

<p align="center">
    <img src=./moe.png> 
</p>
<p align="center">
    (Fig 3. Diagram of MoE)
</p>

MoE is an model which consists of parallel expert models which is fitted to certain domains.  
Like MoD, token-level routing decisions are made across the network depth.  
Difference between MoD is, MoD chooses path to transformer or to residual connection, MoE chooses path to transformer(Expert) or to transformer(Expert) or both.

## **Implementing Mixture-of-Depth Transformers**

High-level strategy of Mixture-of-Depths is as follows:

#### **Defining a compute budget**

To manage compute budgets in transformers, we use **capacity**, which is the total number of tokens processed. In vanilla transformers, capacity ($T$) covers all tokens in self-attention and MLP layers. MoE transformers split this capacity across multiple experts, balancing the compute load. Compute budgets, measured in FLOPs, depend on token capacity rather than routing decisions. Lowering computation capacity can reduce the budget without performance loss if the model learns to prioritize important tokens. Efficient routing schemes are essential for selecting these tokens dynamically. This approach optimizes computation and maintains performance within budget constraints.

#### **Routing around transformer blocks**

#### **Routing schemes**

#### **Routing implementation**

#### **Sampling**

## **Results & Discussion**

## **References**

Fig 2. https://www.sciencedirect.com/science/article/pii/S0893608022002532  
Fig 3. https://deepgram.com/learn/mixture-of-experts-ml-model-guide  
