
## Experiments & Results

### Whats the Optimal Layer for Model Editing?

Investigating the effectiveness of hidden states in LLMS for recalling facts using causal tracing showed thjat subject’s last token within the feed-forward networks at intermediate layer plays a significant role. (Meng et al.(2022b))

**Motivation** : However, later work showed that layers deemed important during causal tracing did not always translate to model editing performance. Therefore, this work focused on finding the optimal layer for model editing layer empirically.

**Steps for finding optimal layer**

1. Make 1000 non-sequential edits from the CounterFact(Meng et al., 2022a) dataset at each layer of the Llama-3 model.
2. Calculate various model metrics(ES, PS, NS, S) to evaluate their impact.
3. The layer that achieves the highest score is selected as the most suitable for targeted interventions.

<p align="center">
  <img src="Blog%20Post/Untitled.png" alt="." width=\textwidth > 
  <img src="Blog%20Post/Untitled%201.png" alt="." width=\textwidth > 
</p>

Evaluation results showed that layer 1 for Llama-3 outperformed on numerous metrics. Furthermore this trend was also shown in previous version, Llama-2, as seen in Figure 6.

Here, MEMIT and ROME have very similar performance for model editing across layer of a model.

→ Why? : Both algorithms optimize for the **same objective** with difference in the memorization constraints. This shows that memorization constraints plays minor effect on editing performance.

### **Optimal** way of Scaling Up model editing?

After finding the optimal layer, scaling of model editing on the same model can happen in two ways : **batch editing** & **sequential editing**.

**Batch Editing :**

A large number(batch size) of knowledge edits are performed on the model with the same update. This work stick to editing a single layer of the model.

Experiment setting

- Targeting layer1 in Llama-3 with  batch size 16, 64, 256, 1024, and 4096 for Batched editing.

<p align="center">
    <img src="Blog%20Post/Untitled%202.png" alt="." > 
</p>

**Evaluation Results of Batch Editing**

<p align="left">
    <img src="Blog%20Post/Untitled%203.png" alt="." >
<p align="right">
    <img src="Blog%20Post/Untitled%204.png" alt="." > 
</p>


For both MEMIT & EMMET editing, metrics are seen to consistently fall with larger batches, with **NS** being the most pronounced to fall. **ES** is most resilient metric to edits. **PS**, only metric to do so, seen to increase dramatically between batch sizes of 16 and 64.

The similar trend between two editing techniques reflect the similarity in their optimization objectives.

**Sequential Batch Editing :** 

**Sequential Editing** is an alternate way to scale up model editing where facts are added sequentially to a model.

This work proposes optimal way to scale model editing that strikes a balance between Batch Editing & Sequential Editing.

**Sequential-batched editing** sequentially edit many batch of facts at a time. And the experiment was conducted going from batch size of 1 up to 4096. (1, 64, 256, 1024, 4096)

![Untitled](Blog%20Post/Untitled%205.png)

Experimental results according to figures above showed that larger batch sizes are actually worse for model performance than sequential edits with smaller batches. 

In contrast, larger batch sizes seem to be better for metrics in NS : while batch edits are less successful in general, it is better in preserving locality of edits.

This results were concluded to optimal batch size of 1024 for both MEMIT and EMMET. Increasing batch-size beyond that lead to larger model degradation and better editing results can be achieved by sequential-batched editing with smaller batch sizes. 

### Conclusion

This work examines several model editing techniques in the context of the newly released Llama-3 model and there are some conclusion as follows:

- Earlier layers may be more optimal intervention points.
- Model editing techniques that share same optimization objectives shows similar trends in layer and editing.
- Smaller, frequent sequential batch size edits have a superior performance.
- Batch size of 1024 for MEMIT and EMMET is optimal batchsize with sequential-batched editing.

 The authors argue that the current trend of pushing towards bigger edit batch sizes for scaling model editing may have limitations. Instead, they propose that future research should focus on methods that combine both batched and sequential editing to optimize performance while minimizing model degradation.

Future work will include experiments on multi-layer intervention for edits, as well as experiments against other popular models and algorithms, including methods that are hyper-network based

- Provide your own perspectives and discussions, and propose a **future research direction.**

NS의 경우 layer가 뒤로 갔을 때 다시 성능이 좋아진 원인, PS에서 batch size를 증가 시켰을 때 좋아지는 이유를 분석하면 multi layer edit에서 optimal point를 찾는데 도움이 될 수도 있을 것 같음.

single layer에서 나아가 multi-layer에서 몇 개의 layer edit이 효과적인지 조사.

batch size가 증가함에 따라 전체적인 metric이 내려가는 상관관계를 emprically말고 이론적으로 밝히면 더욱 효과적으로 model editing 연구에 도움이 될 수 있을 것 같다.
