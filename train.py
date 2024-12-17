import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import json
import csv
from pathlib import Path
import numpy as np
from torch.nn import functional as F

from model import TransformerConfig, Transformer

def tokenize_and_pad(texts, tokenizer, max_length):
    """Tokenize and pad a list of texts in one batch operation"""
    encoded_texts = []
    for text in texts:
        # Encode text
        encoded = [tokenizer['<START>']] + \
                 [tokenizer.get(token, tokenizer['<UNK>']) 
                  for token in text.split()] + \
                 [tokenizer['<END>']]
        
        # Truncate if necessary
        encoded = encoded[:max_length]
        
        # Pad sequence
        encoded += [tokenizer['<PAD>']] * (max_length - len(encoded))
        encoded_texts.append(encoded)
    
    return torch.tensor(encoded_texts)

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, max_length=128):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.max_length = max_length

        # Pre-tokenize all texts
        print("Pre-tokenizing source texts...")
        self.src_encoded = tokenize_and_pad(src_texts, src_tokenizer, max_length)
        print("Pre-tokenizing target texts...")
        self.tgt_encoded = tokenize_and_pad(tgt_texts, tgt_tokenizer, max_length)

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        return {
            'src_ids': self.src_encoded[idx],
            'tgt_ids': self.tgt_encoded[idx],
            'src_text': self.src_texts[idx],
            'tgt_text': self.tgt_texts[idx]
        }

def create_masks(src_ids, tgt_ids):
    # Source mask (padding mask)
    src_mask = (src_ids != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, src_len)

    # Target mask (combination of padding and subsequent mask)
    tgt_pad_mask = (tgt_ids != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, tgt_len)
    
    tgt_len = tgt_ids.size(1)
    subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt_ids.device, dtype=torch.bool))
    subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, tgt_len, tgt_len)

    tgt_mask = tgt_pad_mask & subsequent_mask

    return src_mask.to(torch.bool), tgt_mask.to(torch.bool)

def load_tokenizer(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_dataset(path):
    english_sentences = []
    french_sentences = []
    
    with open(path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            if len(row) == 2:
                english_sentences.append(row[0].lower())  # Convert to lowercase
                french_sentences.append(row[1].lower())  # Convert to lowercase
    
    return english_sentences, french_sentences

def reshuffle_training_data(dataset):
    """Reshuffle the training data while keeping pairs aligned"""
    indices = list(range(len(dataset.src_texts)))
    np.random.shuffle(indices)
    
    # Reorder all dataset attributes using the shuffled indices
    dataset.src_texts = [dataset.src_texts[i] for i in indices]
    dataset.tgt_texts = [dataset.tgt_texts[i] for i in indices]
    dataset.src_encoded = dataset.src_encoded[indices]
    dataset.tgt_encoded = dataset.tgt_encoded[indices]

def train_model(model, train_dataloader, val_dataloader, train_dataset, num_epochs, device, learning_rate=3e-4):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Reshuffle training data at the start of each epoch
        reshuffle_training_data(train_dataset)
        
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_dataloader, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=False)
        
        for batch in train_pbar:
            # Move batch to device
            src_ids = batch['src_ids'].to(device)  # [batch_size, seq_len]
            tgt_ids = batch['tgt_ids'].to(device)  # [batch_size, seq_len]
            
            # Create attention masks
            src_mask = (src_ids != 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]
            tgt_mask = (tgt_ids != 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, tgt_len]
            
            # Forward pass
            logits = model(
                src_ids=src_ids,
                tgt_ids=tgt_ids[:, :-1],  # Remove last token from target input
                src_mask=src_mask,
                tgt_mask=tgt_mask[:, :, :, :-1]  # Adjust mask accordingly
            )
            
            # Calculate loss
            loss = F.cross_entropy(
                logits.contiguous().view(-1, logits.size(-1)),
                tgt_ids[:, 1:].contiguous().view(-1),  # Shift target ids right by 1
                ignore_index=0  # Ignore padding token
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc='Validation', leave=False):
                src_ids = batch['src_ids'].to(device)
                tgt_ids = batch['tgt_ids'].to(device)
                
                src_mask = (src_ids != 0).unsqueeze(1).unsqueeze(2)
                tgt_mask = (tgt_ids != 0).unsqueeze(1).unsqueeze(2)
                
                logits = model(
                    src_ids=src_ids,
                    tgt_ids=tgt_ids[:, :-1],
                    src_mask=src_mask,
                    tgt_mask=tgt_mask[:, :, :, :-1]
                )
                
                loss = F.cross_entropy(
                    logits.contiguous().view(-1, logits.size(-1)),
                    tgt_ids[:, 1:].contiguous().view(-1),
                    ignore_index=0
                )
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss = {avg_train_loss:.4f} | Validation Loss = {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, './models/best_model.pt')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizers
    eng_tokenizer = load_tokenizer('./models/english_tokeniser.json')
    fr_tokenizer = load_tokenizer('./models/french_tokeniser.json')

    # Load dataset
    eng_sentences, fr_sentences = load_dataset('./data/eng_french.csv')

    # # debugging
    # eng_sentences = eng_sentences[0:10000]
    # fr_sentences = fr_sentences[0:10000]
    
    # Create shuffled indices for the full dataset
    indices = list(range(len(eng_sentences)))
    np.random.shuffle(indices)
    
    # Use the shuffled indices to reorder both sentence lists simultaneously
    eng_sentences = [eng_sentences[i] for i in indices]
    fr_sentences = [fr_sentences[i] for i in indices]
    
    # Create train/val split
    split_idx = int(len(eng_sentences) * 0.9)
    
    train_eng = eng_sentences[:split_idx]
    train_fr = fr_sentences[:split_idx]
    val_eng = eng_sentences[split_idx:]
    val_fr = fr_sentences[split_idx:]

    # Create datasets
    train_dataset = TranslationDataset(
        train_eng, train_fr, eng_tokenizer, fr_tokenizer
    )
    val_dataset = TranslationDataset(
        val_eng, val_fr, eng_tokenizer, fr_tokenizer
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=128, 
        shuffle=True,
        num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=128, 
        shuffle=False,
        num_workers=4
    )

    # Initialize model
    config = TransformerConfig(
        src_vocab_size=len(eng_tokenizer),
        tgt_vocab_size=len(fr_tokenizer),
        block_size=128,
        n_layer=6,
        n_pre_cross_layer=3,
        n_cross_layer=3,
        n_embd=256,
        num_heads=8,
        dropout=0.1
    )

    model = Transformer(config).to(device)

    # Train the model
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_dataset=train_dataset,  # Pass the training dataset for reshuffling
        num_epochs=20,
        device=device,
        learning_rate=3e-4
    )

if __name__ == "__main__":
    main()