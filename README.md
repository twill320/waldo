# Transformer-Based Accent-Robust Speech Recognition (ASR)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)]()
[![Model Architecture](https://img.shields.io/badge/Architecture-Seq2Seq_Transformer-5f827d?style=flat-square)]()
[![Status](https://img.shields.io/badge/Status-Architecture_Prototyping-orange?style=flat-square)]()

An end-to-end Automatic Speech Recognition (ASR) neural network written in PyTorch, engineered to transcribe oral speech into text with a targeted focus on non-standard and accented English dialects (specifically Nigerian/Yoruba-accented speech).

---

## Motivation & Project Goals

Many standard automatic speech recognition models exhibit performance degradation when evaluated against accented speech or non-standard regional dialects. Standard training sets are heavily weighted toward General American or British English, often resulting in higher word error rates (WER) for non-native and immigrant speakers.

This project is built from first principles to:
1. Implement a complete sequence-to-sequence Transformer architecture natively in PyTorch.
2. Develop an audio preprocessing and tokenization pipeline capable of mapping acoustic features to text tokens.
3. Train and fine-tune acoustic representations on diverse accent profiles, ensuring high transcription fidelity for Nigerian/Yoruba English speakers alongside standard dialects.

---

## Architecture Overview

The system is constructed around a sequence-to-sequence Encoder-Decoder Transformer based on the *Attention Is All You Need* framework, adapted for acoustic sequence modeling:
[Audio Waveform] -> [Log-Mel Spectrogram / Acoustic Features]
|
v
+--------------------------+
|      Acoustic Encoder    |
| (Multi-Head Attn + FFN)  | x N
+--------------------------+
|
v  (Encoder Representations)
+--------------------------+
|     Autoregressive       | <--- [Target Text Tokens]
|      Text Decoder        |
| (Self-Attn + Cross-Attn) | x N
+--------------------------+
|
v
+--------------------------+
|     Projection Layer     | -> [Softmax Vocabulary Probabilities]
+--------------------------+


---

## Technical Specifications

| Component | Implementation Details |
| :--- | :--- |
| **Framework** | PyTorch 2.0+ |
| **Model Type** | Sequence-to-Sequence (Seq2Seq) Transformer |
| **Layer Normalization** | Learnable scale ($\alpha$) and shift ($\beta$) parameters with numerical stabilization ($\epsilon = 10^{-6}$) |
| **Positional Encoding** | Fixed sinusoidal position embeddings for variable-length sequence handling |
| **Attention Mechanism** | Multi-Head Scaled Dot-Product Attention with attention-mask fill support |
| **Weight Initialization** | Xavier Uniform (`nn.init.xavier_uniform_`) across multi-dimensional parameters |
| **Baseline Attribution** | Modularized and adapted from baseline Transformer implementations by Umar Jamil |

---

## Key Modules in `model.py`

* `InputEmbeddings`: Scales learned discrete token vectors by $\sqrt{d_{\text{model}}}$.
* `PositionalEncoding`: Encodes token and feature order using alternating sinusoidal functions.
* `MultiHeadAttentionBlock`: Computes independent queries, keys, and values projected over $h$ parallel attention heads.
* `FeedForwardBlock`: Two-layer linear transformation with intermediate ReLU activation and dropout.
* `ResidualConnection`: Pre-layer normalization skip connections around all sub-blocks.
* `Encoder` / `Decoder`: Cascaded $N$-layer transformer stacks with self-attention and cross-attention routing.

---

## Installation & Environment Setup

```bash
# 1. Clone the repository
git clone [https://github.com/twill320/waldo.git
cd waldo

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install required packages
pip install torch torchaudio numpy
