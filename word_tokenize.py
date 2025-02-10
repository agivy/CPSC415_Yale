import requests
from transformers import AutoTokenizer
from collections import defaultdict
import nltk
from typing import List, Dict, Set
import time

# def download_documents(url1: str, url2: str) -> List[str]:
#     """Download documents from URLs and create corpus"""
#     texts = []
#     for url in [url1, url2]:
#         print(f"\nDownloading document from {url}...")
#         try:
#             response = requests.get(url)
#             response.raise_for_status()
#             texts.append(response.text)
#         except requests.RequestException as e:
#             print(f"Error downloading from {url}: {e}")
#             return None
#     return texts

def download_documents(url1: str, file2_path: str) -> List[str]:
    """Download the first document from URL and load the second document from a file"""
    texts = []
    
    # Download the first document from the URL
    print(f"\nDownloading document from {url1}...")
    try:
        response = requests.get(url1)
        response.raise_for_status()
        texts.append(response.text)
    except requests.RequestException as e:
        print(f"Error downloading from {url1}: {e}")
        return None
    
    # Load the second document from the file
    print(f"\nLoading document from {file2_path}...")
    try:
        with open(file2_path, 'r', encoding='utf-8') as file:
            texts.append(file.read())
    except FileNotFoundError as e:
        print(f"Error loading file {file2_path}: {e}")
        return None
    except Exception as e:
        print(f"Error reading file {file2_path}: {e}")
        return None
    
    return texts

def create_corpus_from_texts(texts: List[str]) -> List[str]:
    """Create corpus from downloaded texts using sentence tokenization"""
    corpus = []
    for text in texts:
        sentences = nltk.sent_tokenize(text)
        corpus.extend([sent for sent in sentences if sent.strip()])
    return corpus

def compute_word_frequencies(corpus: List[str], tokenizer) -> Dict[str, int]:
    """Compute word frequencies from corpus using BERT pre-tokenization"""
    word_freqs = defaultdict(int)
    for text in corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1
    return word_freqs

def create_alphabet(word_freqs: Dict[str, int]) -> List[str]:
    """Create alphabet from word frequencies"""
    alphabet = []
    for word in word_freqs.keys():
        if word[0] not in alphabet:
            alphabet.append(word[0])
        for letter in word[1:]:
            if f"##{letter}" not in alphabet:
                alphabet.append(f"##{letter}")
    alphabet.sort()
    return alphabet

def create_splits(word_freqs: Dict[str, int]) -> Dict[str, List[str]]:
    """Create initial splits for words"""
    return {
        word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
        for word in word_freqs.keys()
    }

def compute_pair_scores(splits: Dict[str, List[str]], word_freqs: Dict[str, int]) -> Dict[tuple, float]:
    """Compute scores for pairs of tokens"""
    letter_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)

    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            letter_freqs[split[0]] += freq
            continue

        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            letter_freqs[split[i]] += freq
            pair_freqs[pair] += freq
        letter_freqs[split[-1]] += freq

    scores = {
        pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }
    return scores

def merge_pair(a: str, b: str, splits: Dict[str, List[str]], word_freqs: Dict[str, int]) -> Dict[str, List[str]]:
    """Merge a pair of tokens in all splits"""
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                merge = a + b[2:] if b.startswith("##") else a + b
                split = split[:i] + [merge] + split[i + 2:]
            else:
                i += 1
        splits[word] = split
    return splits

def train_vocab(word_freqs: Dict[str, int], vocab_size: int = 70) -> List[str]:
    """Train vocabulary using WordPiece algorithm"""
    # Initialize vocabulary with special tokens and alphabet
    alphabet = create_alphabet(word_freqs)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()

    # Initialize splits
    splits = create_splits(word_freqs)

    # Iteratively merge best pairs until reaching desired vocabulary size
    while len(vocab) < vocab_size:
        scores = compute_pair_scores(splits, word_freqs)
        if not scores:  # If no more pairs to merge
            break

        best_pair = max(scores.items(), key=lambda x: x[1])[0]
        splits = merge_pair(*best_pair, splits, word_freqs)
        new_token = (
            best_pair[0] + best_pair[1][2:]
            if best_pair[1].startswith("##")
            else best_pair[0] + best_pair[1]
        )
        if new_token not in vocab:  # Avoid duplicates
            vocab.append(new_token)

    return vocab

def train_tokenizer(texts: List[str], vocab_size: int = 70):
    """Train tokenizer on provided texts and return the vocabulary"""
    print("\nTraining tokenizer on provided texts...")
    start_time = time.time()

    # Create corpus from provided texts
    corpus = create_corpus_from_texts(texts)
    print(f"Created corpus with {len(corpus)} sentences")

    # Initialize BERT tokenizer for pre-tokenization
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Compute word frequencies
    word_freqs = compute_word_frequencies(corpus, tokenizer)
    print(f"Computed frequencies for {len(word_freqs)} unique words")

    # Train vocabulary
    vocab = train_vocab(word_freqs, vocab_size)
    print(f"Trained vocabulary of size: {len(vocab)}")

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    return vocab, tokenizer

def encode_word(word: str, vocab: List[str]) -> List[str]:
    """Encode a single word using the vocabulary"""
    tokens = []
    while len(word) > 0:
        i = len(word)
        while i > 0 and word[:i] not in vocab:
            i -= 1
        if i == 0:
            return ["[UNK]"]
        tokens.append(word[:i])
        word = word[i:]
        if len(word) > 0:
            word = f"##{word}"
    return tokens

def tokenize_text(text: str, vocab: List[str], tokenizer) -> List[str]:
    """Tokenize text using the trained vocabulary"""
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    encoded_words = [encode_word(word, vocab) for word in pre_tokenized_text]
    return sum(encoded_words, [])

def process_document(text: str, vocab: List[str], tokenizer) -> Dict:
    """Process a single document using the trained vocabulary"""
    # Tokenize the text
    tokenized_text = tokenize_text(text, vocab, tokenizer)
    
    return {
        'tokenized_text': tokenized_text,
        'text_length': len(text),
        'token_count': len(tokenized_text)
    }

# def analyze_documents(url1: str, url2: str, vocab_size: int = 70) -> Dict:
#     """Process documents and analyze their tokenization"""
#     # Download documents
#     texts = download_documents(url1, url2)
#     if not texts:
#         return None
        
#     # Train tokenizer on both documents
#     vocab, tokenizer = train_tokenizer(texts, vocab_size)
    
#     # Process each document
#     results = []
#     for i, text in enumerate(texts, 1):
#         print(f"\nProcessing document {i}...")
#         result = process_document(text, vocab, tokenizer)
#         results.append(result)
#         print(f"Tokenized document {i} into {result['token_count']} tokens")

#     # Compute comparative statistics
#     text1_set = set(results[0]['tokenized_text'])
#     text2_set = set(results[1]['tokenized_text'])
#     tokenized_intersection = text1_set.intersection(text2_set)
#     exclusive_doc1_tokenized = text1_set - text2_set
#     exclusive_doc2_tokenized = text2_set - text1_set

#     return {
#         'vocab': set(vocab),
#         'doc1_results': results[0],
#         'doc2_results': results[1],
#         'tokenized_intersection': tokenized_intersection,
#         'exclusive_doc1_tokenized': exclusive_doc1_tokenized,
#         'exclusive_doc2_tokenized': exclusive_doc2_tokenized,
#         'comparative_stats': {
#             'common_tokenized': len(tokenized_intersection),
#             'exclusive_doc1_tokenized': len(exclusive_doc1_tokenized),
#             'exclusive_doc2_tokenized': len(exclusive_doc2_tokenized)
#         }
#     }

def analyze_documents(url1: str, file2_path: str, vocab_size: int = 70) -> Dict:
    """Process documents and analyze their tokenization"""
    # Download documents
    texts = download_documents(url1, file2_path)
    if not texts:
        return None
        
    # Train tokenizer on both documents
    vocab, tokenizer = train_tokenizer(texts, vocab_size)
    
    # Process each document
    results = []
    for i, text in enumerate(texts, 1):
        print(f"\nProcessing document {i}...")
        result = process_document(text, vocab, tokenizer)
        results.append(result)
        print(f"Tokenized document {i} into {result['token_count']} tokens")

    # Compute comparative statistics
    text1_set = set(results[0]['tokenized_text'])
    text2_set = set(results[1]['tokenized_text'])
    tokenized_intersection = text1_set.intersection(text2_set)
    exclusive_doc1_tokenized = text1_set - text2_set
    exclusive_doc2_tokenized = text2_set - text1_set

    return {
        'vocab': set(vocab),
        'doc1_results': results[0],
        'doc2_results': results[1],
        'tokenized_intersection': tokenized_intersection,
        'exclusive_doc1_tokenized': exclusive_doc1_tokenized,
        'exclusive_doc2_tokenized': exclusive_doc2_tokenized,
        'comparative_stats': {
            'common_tokenized': len(tokenized_intersection),
            'exclusive_doc1_tokenized': len(exclusive_doc1_tokenized),
            'exclusive_doc2_tokenized': len(exclusive_doc2_tokenized)
        }
    }

def print_detailed_results(results: Dict):
    """Print detailed token analysis results"""
    print("\n" + "="*50)
    print("TOKEN ANALYSIS RESULTS")
    print("="*50)

    # Document 1 unique tokens
    doc1_unique = len(set(results['doc1_results']['tokenized_text']))
    print(f"\nDocument 1 (Legal):")
    print(f"Number of unique tokens: {doc1_unique}")

    # Document 2 unique tokens
    doc2_unique = len(set(results['doc2_results']['tokenized_text']))
    print(f"\nDocument 2 (Literature):")
    print(f"Number of unique tokens: {doc2_unique}")

    # Common tokens (intersection)
    common_tokens = len(results['tokenized_intersection'])
    print(f"\nIntersection:")
    print(f"Number of common tokens between documents: {common_tokens}")

    # Exclusive tokens
    print(f"\nExclusive Tokens:")
    print(f"Tokens appearing only in Document 1: {results['comparative_stats']['exclusive_doc1_tokenized']}")
    print(f"Tokens appearing only in Document 2: {results['comparative_stats']['exclusive_doc2_tokenized']}")


url_legal = "https://media.cadc.uscourts.gov/opinions/docs/2001/09/00-5016a.txt"
#original website did not provide entire txt file
#copied contents from https://www.gutenberg.org/files/55/55-h/55-h.htm
#uploaded the entire content as .txt to Github
url_literature = "https://github.com/agivy/CPSC415_Yale/blob/main/The_Wizard_of_Oz.txt"
literature_path = 'The_Wizard_of_Oz.txt'
vocab_size = 10000

# results = analyze_documents(url_legal, url_literature, vocab_size)
results = analyze_documents(url_legal, literature_path, vocab_size)

# Write tokenized outputs to files

# Write tokenized outputs to files
if results:
    print_detailed_results(results)
    
    with open("document1_tokenized.txt", "w") as f:
        f.write(" ".join(results['doc1_results']['tokenized_text']))
    
    with open("document1_exclusive_tokenized.txt", "w") as f:
        f.write(" ".join(results['exclusive_doc1_tokenized']))
    
    with open("document2_tokenized.txt", "w") as f:
        f.write(" ".join(results['doc2_results']['tokenized_text']))
    
    with open("document2_exclusive_tokenized.txt", "w") as f:
        f.write(" ".join(results['exclusive_doc2_tokenized']))
    
    with open("tokenized_intersection.txt", "w") as f:
        f.write(" ".join(results['tokenized_intersection']))

    print("Results saved successfully.")
else:
    print("Failed to process documents.")