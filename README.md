# Graph Transformer Version 2

> This file is adapted from the generation of GPT-4o and Claude 3.5 Sonnet.

This repository contains an enhanced implementation of the vanilla Graph Transformer (GT) network, referred to as GT-V1. The primary focus is on improving the model's performance through advancements in the attention layers, resulting in a new architecture called GT-V2.



## Project Structure

```bash
project/
├── dataset/
│   ├── QM9/
│   └── ZINC/
├── env/
│   ├── linux.yml
│   ├── osx_arm64.yml
│   └── osx_intel.yml
├── lib/
│   ├── molecules.py
│   └── ncut.py
└── task1_regres_zinc.ipynb
```

The core implementation files, datasets, and environment setups are adapted from National University of Singapore CS5284 Graph Machine Learning [tutorial](https://github.com/xbresson/CS5284_2024/tree/main).




## Attention Mechanisms of GT-V2

GT-V2 is an optimized version of the GT-V1, which aims to enhance the modeling capabilities of graph data by making every component in the network "attentioned." The structure introduces three key types of attention mechanisms to capture interactions between nodes and edges effectively.

1. **Cross-attention: node-to-edge**
   - Update node features $h_i$ using edge features $e_{ij}$.
   
   - Mathematical expression: $h_i \leftarrow \sum\limits_{j \in \mathcal{V}} \text{softmax} \left( q_i^{\top} k_{ij} \right) v_{ij}$.
   
2. **Cross-attention: edge-to-node**
   - Update edge features $e_{ij}$ using information from its connected nodes ($h_i$, $h_j$).
   
   - Mathematical expression: $e_{ij} \leftarrow \frac{\exp \left( q_{ij}^{\top} k_i \right)}{\exp \left( q_{ij}^{\top} k_i \right) + \exp \left( q_{ij}^{\top} k_j \right)} v_i + \frac{\exp \left( q_{ij}^{\top} k_j \right)}{\exp \left( q_{ij}^{\top} k_i \right) + \exp \left( q_{ij}^{\top} k_j \right)} v_j$.
   
3. **Self-attention: node-to-node**
   - Update node features $h_i$ by directly considering the relationships with all other nodes.
   
   - Mathematical expression: $h_i \leftarrow \sum\limits_{j \in \mathcal{V}} \text{softmax} \left( q_i^{\top} k_j \right) v_j$.



## Integration Mechanisms in GT-V2

GT-V2 introduces three integration mechanisms for combining self-attention and cross-attention.

1. **Weighted integration:** Uses a fixed weight $\alpha$ to balance self-attention and cross-attention.
   
   $h_k = \alpha \cdot \text{CrossAttention}(h^{\ell}) + (1 - \alpha) \cdot \text{SelfAttention}(h^{\ell})$.
   
   > *Three variants tested*: $\alpha = 0.25, 0.5,$ and $0.75$.

2. **Gated integration:** Implements a learnable gating mechanism, and uses a sigmoid function to compute dynamic weights.

   $g = \sigma \left( W_g \cdot \text{Concat}[\text{CrossAttention}(h^{\ell}), \text{SelfAttention}(h^{\ell})] + b_g \right)$,
   
   $h_k = g \odot \text{CrossAttention}(h^{\ell}) + (1 - g) \odot \text{SelfAttention}(h^{\ell})$.

3. **Mixed integration:** Learns a linear combination of attention outputs.
   
   $h_k = W_m \cdot \text{Concat}[\text{CrossAttention}(h^{\ell}), \text{SelfAttention}(h^{\ell})] + b_m$.


4. **FiLM integration:** Learns leveraging a FiLM layer.
   
   $h_k = W_1 \cdot \text{SelfAttention}(h^{\ell}) + \lbrack W_2 \cdot \text{SelfAttention}(h^{\ell}) \rbrack \odot \text{CrossAttention}(h^{\ell}) + \text{CrossAttention}(h^{\ell})$.



## Results

### **Task 1: Regression on ZINC Dataset**

Training was conducted over 50 epochs on a small subset of the ZINC dataset (2,000 training samples and 200 testing samples) with a fixed random seed. The results presented are the mean loss (and standard deviation) from the last 10 epochs.


|             Network             | Train Loss on ZINC | Test Loss on ZINC  | Time (min) |
| :-----------------------------: | :----------------: | :----------------: | :--------: |
|              GT-V1              |   0.6090(0.0067)   |   0.5624(0.0416)   |   4.7717   |
| GT-V2 (Weighted, $\alpha$=0.25) |   0.5972(0.0068)   |   0.5343(0.0276)   |   8.0485   |
| GT-V2 (Weighted, $\alpha$=0.5)  |   0.5950(0.0089)   |   0.5294(0.0429)   |   8.1099   |
| GT-V2 (Weighted, $\alpha$=0.75) |   0.5945(0.0053)   |   0.5300(0.0399)   |   8.2462   |
|          GT-V2 (Gated)          |   0.5907(0.0079)   |   0.5146(0.0285)   |   9.0665   |
|          GT-V2 (Mixed)          |   0.5877(0.0087)   |   0.5021(0.0363)   |   8.4175   |
|          GT-V2 (FiLM)           | **0.5818**(0.0078) | **0.4995**(0.0240) |   8.6810   |



### Task 2: Generation on QM9 Dataset with Diffusion Model

Training was conducted over 100 epochs on a small subset of the QM9 dataset (1,000 training samples and 200 testing samples) with a fixed random seed. The results presented are the mean loss (and standard deviation) from the last 10 epochs, the percentage of valid molecules, the percentage of unique molecules, and the time taken to train the model.


|     Network      | Train Loss on QM9  |   Valid   | Unique | Time (min) |
| :--------------: | :----------------: | :-------: | :----: | :--------: |
|      GT-V1       |   0.0798(0.0067)   |   39.2%   | 100.0% |  11.3804   |
| GT-V2 (Weighted) |   0.0759(0.0023)   |   46.0%   | 99.8%  |  19.7241   |
|  GT-V2 (Gated)   |   0.0766(0.0026)   |   37.0%   | 100.0% |  22.0124   |
|  GT-V2 (Mixed)   | **0.0757**(0.0027) |   39.7%   | 100.0  |  20.6671   |
|   GT-V2 (FiLM)   |   0.0778(0.0034)   | **48.1%** | 100.0% |  23.1486   |



### Task 3: Generation on ZINC Dataset with Diffusion Model

Training was conducted over 50 epochs on a small subset of the ZINC dataset (2,000 training samples and 200 testing samples) with a fixed random seed. The results presented are the mean loss (and standard deviation) from the last 10 epochs, the percentage of valid molecules, the percentage of unique molecules, and the time taken to train the model.

|     Network      | Train Loss on ZINC | Valid | Unique | Time (min) |
| :--------------: | :---------------: | :---: | :----: | :--------: |
|      GT-V1       | 0.0680(0.0042) | 3.8% | 100.0% | 12.4043 |
| GT-V2 (Weighted) | 0.0622(0.0023) | **14.5%** | 100.0% | 21.6791 |
|  GT-V2 (Gated)   | 0.0595(0.0022) | 10.4% | 100.0% | 23.2297 |
|  GT-V2 (Mixed)   | 0.0623(0.0023) | 13.3% | 100.0% | 22.8696 |
|   GT-V2 (FiLM)   | **0.0575**(0.0033) | 12.8% | 100.0 | 22.9364 |



## References

1. [Vijay Prakash Dwivedi and Xavier Bresson. "A generalization of transformer networks to graphs." *arXiv preprint arXiv:2012.09699* (2020).](https://arxiv.org/abs/2012.09699)
2. [Clement Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal Frossard. "Digress: Discrete denoising diffusion for graph generation." *arXiv preprint arXiv:2209.14734* (2022).](https://arxiv.org/abs/2209.14734)
3. [Jonathan Ho, Ajay Jain and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in neural information processing systems 33 (2020): 6840-6851.](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)
4. [Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin and Aaron Courville. "Film: Visual reasoning with a general conditioning layer." In Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1. 2018.](https://ojs.aaai.org/index.php/AAAI/article/view/11671)