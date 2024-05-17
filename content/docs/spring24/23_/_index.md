---
type: docs
bookToc: True
weight: 1
---

# **Beyond Language Models: Byte Models are Digital World Simulators**

Byte models expand traditional language models to the byte level, starting from the premise that all digital data and operations are fundamentally byte-based. These models process data from various modalities such as text, audio, and images uniformly as bytes, increasing their applicability in a wide digital environment.

In this paper, bGPT is introduced. bGPT is designed to model digital data at the byte level and is optimized to effectively process byte sequences. It has demonstrated performance comparable to specialized models across various modalities, including text, audio, and images, and offers new possibilities for predicting, simulating, and diagnosing hardware operations. Designed to predict and understand bytes, bGPT provides a deeper understanding of and interaction with digital systems.
<p align="center">
    <img src=framework.JPG width="800"> 
</p>
The bGPT framework simulates digital systems using native binary data. It integrates diverse data types into a single model by treating everything as a byte sequence.

### **Exploring bGPT**
***Architecture*** 
Learning patterns in digital systems at the byte level provides a unified approach to integrating various data types, but the high resolution of bytes results in long sequences that significantly increase computational costs. This issue is especially pronounced in transformer-based models, limiting the efficiency and scalability of processing binary data.
bGPT is equipped with a hierarchical structure designed to efficiently handle entire byte sequences. This structure segments a sequence of byte 
$B = \{b_1, b_2, \ldots, b_T\}$   of length $\( T \)$ into a sequence of patches $\( P \)$, where each patch contains exactly $\( S \)$ bytes:
$\[ P = [P_1, P_2, \ldots, P_N] \]$ where %( N = \left\lceil \frac{T}{S} \right\rceil \)% is the number of patches,
