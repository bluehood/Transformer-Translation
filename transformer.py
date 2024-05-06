import math
import copy
import pandas as pd

from transformer_components_gpu import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

# Download dataset 
dataset = load_dataset("Amani27/massive_translation_dataset")

train_dataset = dataset["train"].to_pandas()
validation_dataset = dataset["validation"].to_pandas()
test_dataset = dataset["test"].to_pandas()

desired_columns = ['en_US', 'fr_FR']
train_dataset = train_dataset.drop(columns=[col for col in train_dataset.columns if col not in desired_columns])
validation_dataset = validation_dataset.drop(columns=[col for col in validation_dataset.columns if col not in desired_columns])
test_dataset = test_dataset.drop(columns=[col for col in test_dataset.columns if col not in desired_columns])

total_dataset = pd.concat([train_dataset, validation_dataset, test_dataset])

# Define tokenizers
text_english = "" 
text_french = ""
maximum_length = 512

for index, row in total_dataset.iterrows():
    text_english += row["en_US"] + " "
    text_french += row["fr_FR"] + " "

english_words = sorted(set(text_english.split()))
french_words = sorted(set(text_french.split()))

english_tokeniser_stoi = {word: i + 1 for i, word in enumerate(english_words)}
english_tokeniser_itos = {i + 1: word for i, word in enumerate(english_words)}
french_tokeniser_stoi = {word: i + 1 for i, word in enumerate(french_words)}
french_tokeniser_itos = {i + 1: word for i, word in enumerate(french_words)}

english_tokeniser_stoi["<unt>"] = 0
english_tokeniser_itos[0] = "<unt>"
french_tokeniser_stoi["<unt>"] = 0
french_tokeniser_itos[0] = "<unt>"


print(f'[*] Defined English Word-Level tokeniser with {len(english_words)} tokens.')
print(f'[*] Defined French Word-Level tokeniser with {len(french_words)} tokens.')
print(f'[*] Tokenising datasets.')

def tokenise_dataset(dataset):
    dataset['en_US_tokenised'] = None
    dataset['fr_FR_tokenised'] = None

    for index, row in dataset.iterrows():
        tokens_english = [english_tokeniser_stoi[word] for word in row['en_US'].split()]
        tokens_english_padded = tokens_english[:maximum_length] + [0] * (maximum_length - len(tokens_english))
        dataset.at[index, 'en_US_tokenised'] = tokens_english_padded

        tokens_french = [french_tokeniser_stoi[word] for word in row['fr_FR'].split()]
        tokens_french_padded = tokens_french[:maximum_length] + [0] * (maximum_length - len(tokens_french))
        dataset.at[index, 'fr_FR_tokenised'] = tokens_french_padded

    return dataset

train_dataset = tokenise_dataset(train_dataset)
validation_dataset = tokenise_dataset(validation_dataset)
test_dataset = tokenise_dataset(test_dataset)

print(f'[*] Defining dataset and dataloader.')

class TranslationDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'en_tokens': torch.tensor(self.data.iloc[idx]['en_US_tokenised'], dtype=torch.long),
            'fr_tokens': torch.tensor(self.data.iloc[idx]['fr_FR_tokenised'], dtype=torch.long)
        }
    

train_dataset = TranslationDataset(train_dataset)
validation_dataset = TranslationDataset(validation_dataset)
test_dataset = TranslationDataset(test_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

print(f'[*] Defining Transformer model.')

# Define Hyperparameters
src_vocab_size = len(english_words)
tgt_vocab_size = len(french_words)
d_model = 512
num_heads = 4
num_layers = 3
d_ff = 1024
max_seq_length = 512
dropout = 0.1

# Define transformer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
model = model.to(device)
print(f'[*] Created Transformer model with {sum(p.numel() for p in model.parameters())} parameters.')

# Training loop
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

model.train()

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0  # Track total loss for the epoch
    num_batches = len(train_dataloader)
    
    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        en_tokens = batch['en_tokens'].to(device)
        fr_tokens = batch['fr_tokens'].to(device)
        
        # Forward pass
        outputs = model(src=en_tokens, tgt=fr_tokens[:, :-1])
        
        # Compute loss
        loss = criterion(outputs.contiguous().view(-1, tgt_vocab_size), fr_tokens[:, 1:].contiguous().view(-1))
        
        # Check for NaN or infinity loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or infinity loss encountered in epoch {epoch+1}, batch {batch_idx+1}. Skipping batch...")
            continue
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 1 == 0:  # Print loss every 100 batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{num_batches}], Loss: {loss.item():.4f}")
    
    # Calculate average loss for the epoch
    average_loss = total_loss / num_batches
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}")

# for epoch in range(100):
#     optimizer.zero_grad()
#     output = model(src_data, tgt_data[:, :-1])
#     loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch: {epoch+1}, Loss: {loss.item()}")