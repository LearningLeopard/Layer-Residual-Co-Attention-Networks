# LRCN for Visual Question Answering

This repository implements the **Layer-Residual Co-Attention Networks (LRCN)** as described in the paper below. 

## Description of the Problem and Proposed Solution
  1. The primary problem is improving the performance of Visual Question Answering systems, which require deep understanding and interaction between visual content and textual information.
  2. VQA systems typically **face challenges in integrating multimodal data**, especially when the information transfer between the layers of deep models is inefficient.
  3. To overcome these challenges, the paper proposes implementing the **Layer-Residual Mechanism (LRM)**, a plug-and-play solution that optimizes the multimodal feature transfer and stabilizes training in deeper layers.
  4. The solution involves applying the LRM to the Encoder-Decoder, Pure-Stacking, and Co-Stacking structures to enhance model performance across different VQA tasks.

## Completed Work

We have successfully:

- Reimplemented the **LRCN model architecture** with support for:
  - Encoder–Decoder
  - Pure-Stacking
  - Co-Stacking variants
- Integrated training on the **VQAv2 dataset** using pre-extracted ResNeSt-152 image features
- Used pretrained GloVe embeddings for text and supported vocab generation
- Added full logging, validation, checkpointing, and a custom LR scheduler
- Implemented a binary cross-entropy loss setup for multi-label classification over the top N answers


## Model Overview

The **LRCN** architecture enhances transformer-based Visual Question Answering (VQA) models by introducing a **Layer-Residual Mechanism (LRM)** which is a residual connection between adjacent co-attention blocks. This mitigates information loss across layers and improves gradient flow during training.

### Key Features:

- **Multi-head attention** layers with tunable `num_heads`, `hidden_dim`, and `num_layers`
- Supports multiple architectural variants:
  - `pure_stacking`: Simple stack of co-attention blocks
  - `co_stacking`: Stacks visual and textual co-attentions separately then merges
  - `encoder_decoder`: Uses traditional Transformer encoder-decoder structure
- **Training Setup**:
  - Optimizer: Adam
  - LR Scheduler: Custom decay after epochs 10 and 12
  - Loss: Binary Cross-Entropy over answer vectors
  - Logging: Stream + file logger with step-wise updates
  - Evaluation: Validation loss is reported after every epoch


## How to Run

Example usage:

```bash
python train.py \
  --variant co_stacking \
  --use_cuda \
  --num_epochs 13 \
  --batch_size 64 \
  --num_heads 8 \
  --hidden_dim 512 \
  --num_layers 6 \
  --learning_rate 0.001
```

## Future Work
- Run on the CLEVR dataset to evaluate model generalization and reasoning capabilities
- Experiment with alternative image encoders (e.g., ViT or CLIP)
- Integrate attention visualization for interpretability

## Dataset Statistics:
| Dataset	| Train Images | Validation images | Test Images | Disk Space | Training  Questions, Answers | Validation Questions, Answers | Test Questions |
|---------|--------------|-------------------|-------------|------------|-------------------------------|------------------------------|-----------------|
| VQA v2 | 204,721 | 40,504 | 81,434 | 30GB | 443,757 , 4,437,570 | 214,354 , 2,143,540 | 447,793 , 4,477,930 |
| CLEVR | 70,000 | 15,000 | 15,000 | 18GB | 700,000, 700,000 | 150,000  , 150,000	| 150,000 , 150,000 | 

## LRCN Paper
> **Dezhi Han, Jingya Shi, Jiahao Zhao, Huafeng Wu, Yachao Zhou, Ling-Huey Li, Muhammad Khurram Khan, and Kuan-Ching Li.**  
> *LRCN: Layer-residual Co-Attention Networks for visual question answering.*  
> *Expert Systems with Applications*, Vol. 263, 2025, 125658.  
> [https://doi.org/10.1016/j.eswa.2024.125658](https://doi.org/10.1016/j.eswa.2024.125658)