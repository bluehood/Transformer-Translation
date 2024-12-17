from collections import defaultdict
import re
import json
import csv

def clean_dataset(dataset):
    dataset = dataset.lower()
    dataset = re.sub(r'\n', ' ', dataset)
    dataset = re.sub(r'[^a-z0-9.,!?;\'\" ]', ' ', dataset)
    return dataset

def basic_tokenize(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)

    text = re.sub(r'([.,!?;])', r' \1 ', text)
    text = re.sub(r'(["\'])', r' \1 ', text)
    
    text = re.sub(r'[^a-z0-9.,!?;\'\" ]', ' ', text)
    
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    tokens = text.split()
    return tokens

def create_vocabulary(tokens, min_frequency=2):
    token_counts = defaultdict(int)
    for token in tokens:
        token_counts[token] += 1
    
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<START>': 2,
        '<END>': 3,
    }
    
    token_idx = len(vocab)
    for token, count in token_counts.items():
        if count >= min_frequency:
            vocab[token] = token_idx
            token_idx += 1
    
    return vocab

def encode_text(text, vocab):
    tokens = basic_tokenize(text)
    encoded = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    return encoded

def decode_text(encoded, vocab):
    tokens = [vocab.get(idx, '<UNK>') for idx in encoded]
    text = ' '.join(tokens)
    return text

def save_tokeniser(vocab, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

def load_tokeniser(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_dataset(path):
    english_sentences = []
    french_sentences = []

    # If you have the data in a file named 'sentences.csv':
    with open(path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            if len(row) == 2:
                english_sentences.append(row[0])
                french_sentences.append(row[1])

    return english_sentences, french_sentences

def main():
    # download dataset from Kaggle 
    english_sentences, french_sentences = load_dataset('./data/eng_french.csv')

    english_text = ' '.join(english_sentences)
    french_text = ' '.join(french_sentences)
    
    for dataset_object in [[english_text, 'english'], [french_text, 'french']]:
        dataset = clean_dataset(dataset_object[0])
        tokens = basic_tokenize(dataset)
        vocab = create_vocabulary(tokens)
        save_tokeniser(vocab, f"./models/{dataset_object[1]}_tokeniser.json")

def debug():
    import debugpy
    debugpy.listen(('127.0.0.1', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    return

if __name__ == "__main__":
    debug()
    main()