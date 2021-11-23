"""
Functions for providing visualizations for neural character language model comparisons.
"""
import string

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


def get_char_class(char):
    """Returns 'digit', 'Latin upper', 'Latin lower', 'Cyrillic upper', 'Cyrillic lower', 'punct' or 'special'
    """
    cyrillic_upper = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
    cyrillic_lower = "абвгдежзийклмнопрстуфхцчшщьюя"
    if char.isdigit():
        return 'digit'
    if char in ["<unk>", "<pad>", "<special_token>", "<eow>", "<sow>", "<eos>", "<sos>"]:
        return 'special'
    if char in cyrillic_upper:
        return "Cyrillic upper"
    if char in cyrillic_lower:
        return "Cyrillic lower"
    if char.isupper():
        return 'Latin upper'
    if char.islower():
        return 'Latin lower'
    if char in string.punctuation:
        return 'punctuation'
    else:
        return 'symbol'


def build_visualization_dataframe(char_embedding_weights, char_vocab):
    """Creates TSNE and PCA projections for the character embeddings,
    then returns a dataframe with PCA and TSNE coordinates and character identity info.
    :param char_embed_weights: numpy array
    """
    char_pca = PCA(2)
    char_pca_proj = char_pca.fit_transform(char_embedding_weights)
    tsne = TSNE(2, init='pca')
    char_tsne_proj = tsne.fit_transform(char_embedding_weights)
    entries = []
    for k, char_idx in char_vocab.char_to_index.items():
        char_type = get_char_class(k)
        entries.append((k, char_idx, char_type, char_pca_proj[char_idx][0], char_pca_proj[char_idx][1], char_tsne_proj[char_idx][0], char_tsne_proj[char_idx][1]))

    char_plot_df = pd.DataFrame.from_records(entries, columns=["char", "embedding_index", "char_class", "pca_x", "pca_y", "tsne_x", "tsne_y"])

    return char_plot_df, char_pca_proj, char_tsne_proj

def get_word_embedding_input_batch(word_vocab, char_vocab, model, word_embedding_size):
    """Returns the embeddings as (word_vocab_size, embedding_size) np array, where the
    embedding for a word is at the same index assigned to the word in the word vocab.
    The returned array can be fed to distance metrics up nearest neighbors in the learned word vocabulary.
    """
    lengths = [1, 1, 1] # Lengths for padding, start and end of sentence
    print("Word to index size:", len(word_vocab.word_to_index))
    index_to_word = {v:k for k, v in word_vocab.word_to_index.items()}
    print("Index to word size:", len(index_to_word))
    print("Those two numbers should match")
    max_length = max([len(v) for v in word_vocab.word_to_char_embedding.values()])
    words_in = np.zeros((word_vocab.vocab_size(), max_length))
    print(words_in.shape)
    # Start and end of sentence
    words_in[1][0:1] = np.array(char_vocab.char_to_index["<sos>"])
    words_in[2][0:1] = np.array(char_vocab.char_to_index["<eos>"])
    for w, idx in word_vocab.word_to_index.items():
        embedding = word_vocab.word_to_char_embedding[w]
        emb_len = len(embedding)
        lengths.append(emb_len)
        words_in[idx][0:emb_len] = np.array(embedding)
    print(len(words_in))

    model.eval()
    word_embedding_batches = []
    # Break up words in vocab into smaller batches so there aren't any memory issues
    word_char_batches = np.array_split(words_in, word_embedding_size)
    start = 0
    for b in word_char_batches:
        end = start+ b.shape[0]
        if start % 50==0: print("Batch from", start, "to", end)
        batch_emb = model.get_word_embeddings(b, np.array(lengths[start:end])).detach().numpy()
        if batch_emb.shape[0] != (b.shape[0]):
            print("Batch doesn't match, start:", start, "end:", end)
        word_embedding_batches.append(batch_emb)
        start = end

    word_embeddings = np.vstack(word_embedding_batches)
    print("Embedding shape:", word_embeddings.shape)
    return word_embeddings

def get_most_similar_words(word, word_embeddings, word_vocab, n=10):
    """Given a word, looks up its index in the word_vocab, then returns a list
    of the n most similar words using cosine similarity of the word embeddings
    produced by the model.
    """
    word_idx = word_vocab.word_to_index[word]
    print("Word:", word, "Index:", word_idx)
    this_embedding = word_embeddings[word_idx].reshape(1, -1)
    similarities = cosine_similarity(this_embedding, word_embeddings)
    # Note that this sorts in increasing order, so the most similar are at the end
    most_similar_idx = np.argsort(similarities, axis=1).reshape(-1)
    idx_to_word = {v:k for k,v in word_vocab.word_to_index.items()}
    word_list = []
    for i in range(n):
        word_list.append(idx_to_word[most_similar_idx[-(i+2)]])

    print(word_list)
    return word_list
