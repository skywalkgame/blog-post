###  Experiments

### 3.1 Finding Optimal Layer for Model Editing

Meng et al.(2022b) assess the effectiveness of hidden states in LLMS for recalling facts using causal tracing → subject’s last token within the feed-forward networks at intermediate layer plays a significant role.

**Motivation** : However, previous work showed that layers deemed important during causal tracing did not always translate to model editing performance. (causal tracing 중에서 중요하다고 생각된 layer들이 model editing performance에서도 좋은 것은 아니었음.)

→ Therefore, find the optimal layer for model editing layer empirically.

1. Make 1000 non-sequential edits from the CounterFact(Meng et al., 2022a) dataset at each layer of the Llama-3 model.
2. Calculate various model metrics to evaluate their impact(ES, PS, NS, S)
3. The layer that achieves the highest score is selected as the most suitable for targeted interventions.

![Untitled](Untitled.png)

![Untitled](Untitled%201.png)

→ 결론 : Llama-3에서는 layer 1이 모든 metric에서 outperform. + previous version, Llama-2에서도 동일하게 나타남 as seen in Figure 6.

+ MEMIT and ROME have very similar performance for model editing across layer of a model.

→ Why? : Both algorithms optimize for the same objective with difference in the memorization constraints. Where memorization constraints plays minor effect on editing performance.

### 3.2 Batch Editing

To perform large scale model edits on the same model after finding the optimal layer for model editing. A large number of knowledge edits are performed on the model with the same update. 

This work stick to editing a single layer of the model.

→ 한번에 한 가지 정보에 대해서만 update하는 것이 아니라 batch 형태로 여러 개의 정보에 대해서 model editing을 진행하고 여러 layer를 건들기보다는 single layer에 집중

Experiment setting

- Targeting layer1 in Llama-3 with  batch size 16, 64, 256, 1024, and 4096 for Batched editing.

![Untitled](Blog%20Post%20%E1%84%8C%E1%85%A1%E1%86%A8%E1%84%89%E1%85%A5%E1%86%BC%2004c42d1fe7304b03af409de21c8b8e1a/Untitled%202.png)

![Untitled](Blog%20Post%20%E1%84%8C%E1%85%A1%E1%86%A8%E1%84%89%E1%85%A5%E1%86%BC%2004c42d1fe7304b03af409de21c8b8e1a/Untitled%203.png)

**Evaluation Results of Batch Editing**

![Untitled](Blog%20Post%20%E1%84%8C%E1%85%A1%E1%86%A8%E1%84%89%E1%85%A5%E1%86%BC%2004c42d1fe7304b03af409de21c8b8e1a/Untitled%204.png)

→ Metrics are seen to consistently fall with larger batches, with **NS** being the most pronounced to fall. ES is most resilient metric to edits. PS, only metric to do so, seen to increase dramatically between batch sizes of 16 and 64.

![Untitled](Blog%20Post%20%E1%84%8C%E1%85%A1%E1%86%A8%E1%84%89%E1%85%A5%E1%86%BC%2004c42d1fe7304b03af409de21c8b8e1a/Untitled%205.png)

EMMET shows similar trends to MEMIT. This is reflected by the similarity in their optimization objectives.

### 3.3 Sequential Batch Editing

Alternate way to scale up model editing.

- Facts are added sequentially to a model.

→ This work proposes optimal way to scale model editing that strikes a balance between these methods.

Sequential-batched editing. →  sequentially edit many batch of facts at a time. And the experiment was conducted going from batch size of 1 up to 4096. (1, 64, 256, 1024, 4096)

![Untitled](Blog%20Post%20%E1%84%8C%E1%85%A1%E1%86%A8%E1%84%89%E1%85%A5%E1%86%BC%2004c42d1fe7304b03af409de21c8b8e1a/Untitled%206.png)

→ Experimental results showed that larger batch sizes are actually worse for model performance than sequential edits with smaller batches. 

In contrast, larger batch sizes seem to be better for metrics in NS → while batch edits are less successful in general, it is better in preserving locality of edits.

This results can be concluded to optimal batch size of 1024 for both MEMIT and EMMET. Increasing batch-size beyond that lead to larger model degradation and better editing results can be achieved by sequential-batched editing with smaller batch sizes. 

### 4. Conclusion

Examines several model editing techniques in the context of the newly released Llama-3 model.

- Earlier layers may be more optimal intervention points.
- Model editing techniques that share same optimization objectives shows similar trends in layer and editing.
- Smaller, frequent sequential batch size edits have a superior performance
- Batch size of 1024 for MEMIT and EMMET is optimal batchsize.

 The authors argue that the current trend of pushing towards bigger edit batch sizes for scaling model editing may have limitations. Instead, they propose that future research should focus on methods that combine both batched and sequential editing to optimize performance while minimizing model degradation.

Future work will include experiments on multi-layer intervention for edits, as well as experiments against other popular models and algorithms, including methods that are hyper-network based

- Provide your own perspectives and discussions, and propose a **future research direction.**

NS의 경우 layer가 뒤로 갔을 때 다시 성능이 좋아진 원인, PS에서 batch size를 증가 시켰을 때 좋아지는 이유를 분석하면 multi layer edit에서 optimal point를 찾는데 도움이 될 수도 있을 것 같음.

single layer에서 나아가 multi-layer에서 몇 개의 layer edit이 효과적인지 조사.

batch size가 증가함에 따라 전체적인 metric이 내려가는 상관관계를 emprically말고 이론적으로 밝히면 더욱 효과적으로 model editing 연구에 도움이 될 수 있을 것 같다.
