import torch
import json
from model import Transformer, TransformerConfig

def load_tokenizer(path):
    """Load tokenizer from json file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def encode_text(text, tokenizer):
    """Encode text using the tokenizer"""
    # Convert to lowercase and split
    tokens = text.lower().split()
    
    # Add special tokens and encode
    encoded = [tokenizer['<START>']] + \
              [tokenizer.get(token, tokenizer['<UNK>']) for token in tokens] + \
              [tokenizer['<END>']]
              
    return torch.tensor(encoded).unsqueeze(0)  # Add batch dimension

def decode_text(token_ids, tokenizer):
    """Decode token ids back to text"""
    # Create a reverse lookup dictionary
    id_to_token = {v: k for k, v in tokenizer.items()}
    
    # Convert ids to tokens, ignore special tokens
    tokens = [id_to_token[id.item()] for id in token_ids if id.item() not in 
             [tokenizer['<PAD>'], tokenizer['<START>'], tokenizer['<END>'], tokenizer['<UNK>']]]
    
    return ' '.join(tokens)

def translate(text, model, src_tokenizer, tgt_tokenizer, device, max_length=128):
    """Translate a single piece of text"""
    model.eval()
    
    # Encode input text
    src_ids = encode_text(text, src_tokenizer)
    src_ids = src_ids.to(device)
    
    # Generate translation
    with torch.no_grad():
        output_ids = model.generate(
            src_ids=src_ids,
            max_new_tokens=max_length,
            temperature=0.7,
            top_k=50
        )
    
    # Decode the generated tokens
    translation = decode_text(output_ids[0], tgt_tokenizer)
    
    return translation

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizers
    print("Loading tokenizers...")
    eng_tokenizer = load_tokenizer('./models/english_tokeniser.json')
    fr_tokenizer = load_tokenizer('./models/french_tokeniser.json')
    
    # Initialize model configuration
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
    
    # Initialize and load the trained model
    print("Loading model...")
    model = Transformer(config).to(device)
    checkpoint = torch.load('./models/best_model_10k.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\nTranslation Interface")
    print("Enter 'q' to quit")
    
    while True:
        # Get input from user
        text = input("\nEnter English text to translate: ")
        
        # Check for quit command
        if text.lower() == 'q':
            break
            
        # Skip empty input
        if not text.strip():
            continue
        
        try:
            # Translate the text
            translation = translate(text, model, eng_tokenizer, fr_tokenizer, device)
            print(f"French translation: {translation}")
            
        except Exception as e:
            print(f"Error occurred during translation: {str(e)}")
            continue

if __name__ == "__main__":
    main()