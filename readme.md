# Transformer for Translation

This project implements a translation model using the Transformer architecture, based on the groundbreaking paper "Attention is All You Need" (Vaswani et al., 2017). The implementation focuses on English-to-French translation whilst offering a simple to understand implementation of the architecture in PyTorch.

## Project Overview

The Transformer architecture revolutionized natural language processing by eliminating the need for recurrent or convolutional neural networks, instead relying entirely on attention mechanisms to capture relationships between words. This implementation showcases three key innovations:

1. Multi-Head Self-Attention: Allowing the model to simultaneously attend to information from different representation subspaces
2. Encoder-Decoder Architecture: Processing the input sequence and generating the output sequence using stacked attention layers
3. Positional Encoding: Incorporating sequence order information without recurrence

## Installation

```bash
git clone https://github.com/yourusername/transformer-translation.git
cd transformer-translation
pip install -r requirements.txt
```

## Implementation Details

### Core Components

1. **Tokenization** (`tokenisers.py`):
   - Word-level tokenization with special tokens (PAD, UNK, START, END)
   - Vocabulary creation with frequency-based filtering
   - Text encoding and decoding utilities

2. **Transformer Architecture** (`model.py`):
   - Multi-head attention implementation with separate query, key, and value projections
   - Encoder and decoder stacks with residual connections
   - Position-wise feed-forward networks
   - Positional encoding implementation

3. **Training Pipeline** (`train.py`):
   - Custom dataset class for handling parallel text data
   - Training loop with learning rate scheduling
   - Validation and model checkpointing
   - Generation utilities for inference

We have also provided a notebook `Transformer_Translation.ipynb` which describes how the model works and how training works.

### Model Configuration

The default model configuration includes:

- 6 encoder layers
- 3 pre-cross-attention decoder layers
- 3 cross-attention decoder layers
- 8 attention heads
- 256 embedding dimensions
- Dropout rate of 0.1

## Training

To train the model:

```bash
python train.py
```

The training script includes:
- Dynamic learning rate adjustment
- Gradient clipping
- Model checkpointing
- Validation monitoring

## Data Preparation

The model expects parallel text data in CSV format with columns for source (English) and target (French) sentences. The data should be preprocessed to:

1. Convert text to lowercase
2. Add appropriate spacing around punctuation
3. Remove special characters
4. Normalize whitespace

An example dataset that is appropriate can be found at https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench. 

## Inference

To translate text using a trained model:

```python
from model import Transformer, TransformerConfig
import torch

# Load model and tokenizers
model = Transformer(config).to(device)
model.load_state_dict(torch.load('models/best_model.pt')['model_state_dict'])

# Generate translation
translated_ids = model.generate(
    src_ids=encoded_input,
    max_new_tokens=128,
    temperature=1.0,
    top_k=50
)
```

## Performance Considerations

The implementation includes several optimizations:

- Parallel computation in multi-head attention
- Efficient batch processing of sequences
- Memory-efficient attention masking
- Gradient clipping for stable training

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- tqdm

Additional dependencies can be found in `requirements.txt`.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```