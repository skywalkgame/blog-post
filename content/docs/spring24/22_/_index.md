---
type: docs
bookToc: True
weight: 1
---
# Larimar: Large Language Models with Episodic Memory Control
*Authors: Payel Das (IBM AI Research), Subhajit Chaudhury (IBM AI Research) et.al*

## Background
Pre-trained Large Language Models (LLMs) have demonstrated outstanding performance on various Natural Language Processing (NLP) tasks.
However, LLMs have faced a lot of challenges at the same time. Especially, in this post, we focus on the "knowledge update" problem.

### Knowledge update in LLM research
In order to keep pre-trained LLMs fact-relevant, safe, and ethical after deploymemnt, the knowledge of the LLM needs to be constantly updated. To be specific, model editing is mandatory to remove the undesired, incorrect, or obsolete facts from the LLM's "memory", and optionally replace it with desired outcome. Firgures below shows why do we need knowledge update with detailed examples.
<p align="center">
    <img src='knowledge update.png' width="600">
</p>
<p align="center">
    (Fig1. Knowledge update: New knowledge should be injected constantly)
</p>

<p align="center">
    <img src='context length generalization.png' width="600">
</p>
<p align="center">
    (Fig2. Context length generalization: The ability to quickly update the LLM can help with "input context length generalization problem")
</p>

<p align="center">
    <img src='selective fact forgetting.png' width="600">
</p>
<p align="center">
    (Fig3. Selective fact forgetting: LLMs should forget personal & sensitive data)
</p>

### Autoencoder

<p align="center">
    <img src='auto encoder.png' width="600">
</p>
<p align="center">
    (Fig4. Autoencoder)
</p>


### Memory network

### Neocortex-Hippocampus interactions
This paper imitates the role of brain. Humans can rapidly update their knowledge after encountering the first relevant instance. In the brain, this process is facilitated through interactions between the neocortex and the hippocampus. The hippocampus is the site for storing long-term memories, while the neocortex integrates long-term and short-term memories to relay the results to the body. 
<p align="center">
    <img src=brain.png width="400"> 
</p>
The Complementary Learning Systems (CLS) theory proposes a model that combines these complementary learning systems of the hippocampus and neocortex. The interaction between the neocortex and hippocampus in the brain is known to promote adaptive behavior through memorization and generalization. Furthermore, it is suggested that memory consolidation from the hippocampus to the neocortex is facilitated by the activation synchronized with multiple exact or false replays of the encoded experience in the hippocampus. This implies that the hippocampus functions as a generative associative network.

## Contributions
1. Larimar introduces a class of memory-conditioned language models inspired by complementary learning mechanisms in the brain. This architecture facilitates real-time test-time adaptation without requiring time-intensive gradient-based learning or internal fact tracing, offering a faster method for updating LLMs.
Utility Demonstration in Knowledge Editing and Context Generalization:

2. The proposed method is demonstrated on two significant and challenging use cases: knowledge editing and generalizing to longer input contexts. Larimar exhibits fast and accurate training-free adaptation to new inputs in both scenarios, outperforming baseline editing methods and existing language models.
Selective Fact Forgetting and Information Leakage Prevention:

3. Larimar effectively supports selective fact forgetting and prevents information leakage using its one-shot memory updating mechanism.
Recursive Search-Based Solution for Long Context Generalization: A simple recursive search-based approach is provided to enable Larimar's memory to generalize to longer input contexts.


## Model architecture
Larimar consists of three main components: encoder, decoder, and adaptive memory.
1) Encoder: Transforms the input into a latent vector
2) Decoder: Generates an answer to the question conditioned on the memory
3) Memory: Stores episodes in encoded form

## Memory Operations


## Results

## Conclusion

## References

https://arxiv.org/abs/2310.16218
 -> Knowledge Editing for Large Language Models: A Survey

https://arxiv.org/abs/2207.04901
 -> Exploring Length Generalization in Large Language Models

https://arxiv.org/abs/2402.05813
 -> Selective Forgetting: Advancing Machine Unlearning Techniques and Evaluation in Language Models


https://arxiv.org/abs/2403.11901

[brain figure](https://www.rallyware.com/blog/the_neuroscience_behind_successful_talent_development)
