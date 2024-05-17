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
{{< katex >}}B = \{b_1, b_2, \ldots, b_T\}{{< /katex >}} of length {{< /katex >}}T{{< /katex >}} into a sequence of patches {{< /katex >}}mathcal{P}{{< /katex >}}, where each patch contains exactly {{< /katex >}}S{{< /katex >}} bytes:
{{< /katex >}}\mathcal{P} = [P_1, P_2, \ldots, P_N]{{< /katex >}} where {{< /katex >}}( N = \left\lceil \frac{T}{S} \right\rceil \){{< /katex >}} is the number of patches,
{{< /katex >}}P_i = [b_{(i-1)S+1}, \ldots, b_iS]{{< /katex >}} for {{< /katex >}}\( 1 \leq i \leq N \){{< /katex >}}, {{< /katex >}}\[ P_N = [b_{(N-1)S+1}, \ldots, b_T, e, \ldots, e] \]{{< /katex >}} where {{< /katex >}}\( e \{{< /katex >}} represents the `<eop>` (end-of-patch).

***Components***
- **Linear Projection Layer**: Each byte patch is mapped to a high-dimensional feature space through a linear projection layer. During this process, each byte is encoded into a 257-dimensional vector, which includes the 256 possible byte values and a special `<eop>` (end-of-patch) token.
- **Patch-Level Decoder**: The embedded patches are processed by a patch-level decoder. This decoder plays a role in predicting the features of the next patch from the embedding of each patch, thereby learning the structural patterns of the entire dataset.
- **Byte-Level Decoder**: Based on the predicted patch features, the byte sequence within each patch is reconstructed. The byte-level decoder uses the features of each patch to predict the next byte within that patch, processing the detailed information of the entire byte sequence.

### Model Training
**Generative Modeling**

This approach requires the model to predict the next byte in a given byte sequence. The model takes the byte sequence \( B = \{b_1, b_2, \ldots, b_T\} \) as input and utilizes all previous byte information to predict the next byte \( b_{i+1} \) at each position.

As a loss function, the negative log likelihood of the next byte at each step is minimized. This encourages the model to maximize the likelihood of the actual occurrence of the next byte.

\[ L_{\text{GEN}}(\theta) = -\sum_{i=1}^{T-1} \log p(b_{i+1} \mid b_1, b_2, \ldots, b_i; \theta) \]

**Classification**

Based on the knowledge acquired through generative modeling, bGPT can also be applied to classification tasks for labeled datasets. In this process, the model takes a byte sequence as input and predicts the category to which that sequence belongs.

For classification tasks, the loss function used is the cross-entropy loss, which ensures that the model accurately outputs the prediction probabilities for each category.

\[ L_{\text{CLF}}(\theta) = -\sum_{k=1}^{K} y_k \log p(y_k \mid B; \theta) \]

These training objectives enable bGPT to understand various byte-based data and accurately mimic digital patterns of the real world. The combination of generative approaches and classification capabilities grants the model the flexibility to tackle a diverse range of problems. Through this, the model can go beyond simple pattern recognition to play a crucial role in predicting and analyzing the operations of complex digital systems.
