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

GT-V2 introduces three integration mechanisms for combining self-attention and cross-attention in the transformer layers.

1. **Weighted integration**

   - Uses a fixed weight $\alpha$ to balance self-attention and cross-attention.
   
   - Formula: $h_k = \alpha \cdot \text{CrossAttention}(h^{\ell}) + (1 - \alpha) \cdot \text{SelfAttention}(h^{\ell})$.
   
   - *Three variants tested*: $\alpha = 0.25, 0.5,$ and $0.75$.

2. **Gated integration**

   - Implements a learnable gating mechanism, and uses a sigmoid function to compute dynamic weights.

   - Formula:

     $$
     g = \sigma \left( W_g \cdot \text{Concat}[\text{CrossAttention}(h^{\ell}), \text{SelfAttention}(h^{\ell})] + b_g \right) \\
     h_k = g \odot \text{CrossAttention}(h^{\ell}) + (1 - g) \odot \text{SelfAttention}(h^{\ell}) \text{.}
     $$

3. **Mixed integration**

   - Learns a linear combination of attention outputs.
   
   - Formula: $h_k = W_m \cdot \text{Concat}[\text{CrossAttention}(h^{\ell}), \text{SelfAttention}(h^{\ell})] + b_m$.



## Results

### **Task 1: ZINC Regression**

Training was conducted over 50 epochs on a small subset of the ZINC dataset, consisting of 2,000 training samples and 200 testing samples, using a fixed random seed (set to 0). The results presented are the mean loss (and standard deviation) from the last 10 epochs.

| Network | Train Loss | Test Loss | Time (min) |
|---------|------------|-----------|------------|
| GT-V1 | 0.6090(0.0067) | 0.5624(0.0416) | 4.7717 |
| GT-V2 (Weighted, α=0.25) | 0.5995(0.0072) | 0.5218(0.0312) | 8.0890 |
| GT-V2 (Weighted, α=0.5) | 0.5944(0.0099) | 0.5308(0.0370) | 7.8275 |
| GT-V2 (Weighted, α=0.75) | 0.5923(0.0084) | 0.5247(0.0321) | 7.9889 |
| GT-V2 (Gated) | 0.5939(0.0078) | 0.5232(0.0285) | 8.7069 |
| GT-V2 (Mixed) | **0.5847**(0.0088) | **0.5068**(0.0176) | 8.4193 |

Key findings:
- All GT-V2 variants outperform the original GT-V1 architecture.
- Mixed integration achieves the best performance on both training and test sets.
- Computational overhead of GT-V2 results in approximately 2x longer training time.
- Gated and mixed integration show more stable performance (lower standard deviation).



## References

1. [Vijay Prakash Dwivedi and Xavier Bresson. "A generalization of transformer networks to graphs." *arXiv preprint arXiv:2012.09699* (2020).](https://arxiv.org/abs/2012.09699)
2. [Clement Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal Frossard. "Digress: Discrete denoising diffusion for graph generation." *arXiv preprint arXiv:2209.14734* (2022).](https://arxiv.org/abs/2209.14734)