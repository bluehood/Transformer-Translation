# Transformer for Language Translation (English to French)

This project utilizes a Transformer model for language translation between English and French languages. The Transformer architecture is particularly well-suited for sequence-to-sequence tasks like translation.

## Dataset
The project uses the "massive_translation_dataset" available through the Hugging Face datasets library. The dataset contains parallel text data for English and French languages.

### Dataset Preprocessing
1. The dataset is tokenized into English and French words.
2. Word-level tokenization is performed using custom tokenizers.
3. The tokenized datasets are padded to a maximum sequence length of 512 tokens.
4. Datasets are split into train, validation, and test sets.
5. PyTorch `Dataset` and `DataLoader` classes are defined for efficient data loading during training.

## Model Architecture
The core of the project is the Transformer model for sequence-to-sequence translation.

### Transformer Architecture
- **Encoder**: Consists of multiple encoder layers, each containing a multi-head self-attention mechanism and a position-wise feedforward neural network.
- **Decoder**: Similar to the encoder but includes an additional cross-attention mechanism to attend to the encoder's output.
- **Positional Encoding**: Adds positional information to the input sequences to preserve their order.

## Training
The model is trained using a Cross-Entropy loss function and optimized using the Adam optimizer. Training is performed on a GPU if available.

### Training Loop
The training loop iterates over the dataset for multiple epochs. During each epoch:
1. Input sequences are passed through the model to obtain predicted outputs.
2. Cross-Entropy loss is calculated between predicted and target sequences.
3. Gradients are computed and model parameters are updated using backpropagation.
4. Loss statistics are printed for monitoring training progress.

## Usage
To train the model:

```bash
python train.py
```
