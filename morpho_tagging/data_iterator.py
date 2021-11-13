import codecs
from collections import Counter
import numpy as np
import math
import os

batching_seed = np.random.RandomState(1234)


class WordVocab:
    """Index all the words in the corpus for next word predictions and caching char embeddings
    """
    def __init__(self, char_vocab):
        # Reserved index for special words
        self.unknown_index = 0
        self.start_sentence_index = 1
        self.end_sentence_index = 2
        # starts at 3 for remaining vocab words
        self.indexer = 3
        self.word_to_index = {}
        # Cache training and prediction time word embeddings
        self.word_to_char_embedding = {}
        self.char_vocab = char_vocab

    def lookup_word(self, word, is_train=False):
        """If a word is missing from the vocab, its added.
        Return (index for word, its character embedding) and increment its count in the corpus.
        """
        if word not in self.word_to_index:
            # Unseen words at train time get added
            if is_train:
                self.word_to_index[word] = self.indexer
                word_index = self.indexer
                self.indexer += 1
            # Unseen words at prediction time
            else:
                word_index = self.unknown_index
        else:
            word_index = self.word_to_index[word]

        # Word's character representation isn't cached
        if word not in self.word_to_char_embedding:
            if is_train:
                char_repr = self.char_vocab.add_string(word, add_eow=True, add_sow=True)
                self.word_to_char_embedding[word] = char_repr
            else:
                char_repr = self.char_vocab.string_to_index(word, add_eow=True, add_sow=True)
                self.word_to_char_embedding[word] = char_repr
        else:
            char_repr = self.word_to_char_embedding[word]

        return word_index, char_repr


class CharacterGramVocab:
    def __init__(self,gram=1):

        self.unk_string = "<unk>"
        self.unk_index = 1
        self.pad_string = "<pad>"
        self.pad_index = 0
        self.special_token_string = "<special_token>"
        self.special_token_index = 2
        self.char_to_index = {self.pad_string: self.pad_index,self.unk_string: self.unk_index,self.special_token_string:self.special_token_index}
        self.vocab = [self.pad_string,self.unk_string,self.special_token_string]
        self.index_to_count = Counter()

        # end of word and start of word chars
        self.eow_string = "<eow>"
        self.sow_string = "<sow>"

        self.gram = gram

        # end of sentence and start of sentence chars (also their own 'tokens')
        self.end_sent_marker = "<eos>"
        self.start_sent_marker = "<sos>"

    def get_char_index_and_increment(self, c):
        """Checks the vocab index for char's index and adds it if missing.
        Increment the count for that character.
        Returns the characters index.
        """
        if c not in self.vocab:
            char_index = len(self.vocab)
            self.char_to_index[c] = char_index
            self.vocab.append(c)
        else:
            char_index = self.char_to_index[c]

        self.index_to_count[char_index]+=1
        return char_index

    def add_string(self, string_value, add_eow=False, add_sow=False):
        result = []
        string_chars = list(string_value)
        if add_sow:
            string_chars.insert(0, self.sow_string)
        if add_eow:
            string_chars.append(self.eow_string)

        for i in range(len(string_chars)-self.gram+1):
            c = "".join(string_chars[i:i+self.gram])
            result.append(self.get_char_index_and_increment(c))

        return result

    def string_to_index(self, string_value, add_eow=False, add_sow=False):
        result = []

        string_chars = list(string_value)
        if add_sow:
            string_chars.insert(0, self.sow_string)
        if add_eow:
            string_chars.append(self.eow_string)


        for i in range(len(string_chars)-self.gram+1):

            c = "".join(string_chars[i:i+self.gram])
            if c not in self.vocab:
                result.append(self.char_to_index[self.unk_string])
            else:
                result.append(self.char_to_index[c])

        return result

    def index_to_char(self, index):
        if isinstance(index, int):
            if index < len(self.vocab):
                return self.vocab[index]
            else:
                return None
        elif isinstance(index, list):
            result = []
            for i in index:
                if i >= len(self.vocab):
                    result.append(None)
                else:
                    result.append(self.vocab[i])
            return result
        else:
            return None

    def get_special_token_index(self):
        return self.char_to_index[self.special_token_string]

    def mark_sentence(self, start=True):
        """
        Return sentence marking indices (as a list, just like adding a word).
        :param start: boolean - True to mark start of sentence, false to mark end
        """
        if start:
            marker = self.start_sent_marker
        else:
            marker = self.end_sent_marker

        return [self.get_char_index_and_increment(marker)]


class Tag:
    # Note that this assumes _na_ will always be the 0th index in the tags file
    def __init__(self, name, index, values):
        self.name = name
        self.index = index
        self.values = values
        self.counts = np.zeros((len(self.values),))

    def get_tag_index(self, value):
        if (value in self.values):
            return self.values.index(value)
        else:
            return self.values.index('_na_')

    def add(self, name):
        self.counts[self.get_tag_index(name)] += 1


def read_tags(path, lower=False):
    tag_dict = {}
    f = codecs.open(path, 'r', 'utf-8')
    for index, tags in enumerate(f.readlines()):

        parts = tags.strip().split("\t")
        tagname = parts[0]
        tag_values = parts[1:]

        if lower:
            tagname = tagname.lower()
            tag_values = [t.lower() for t in tag_values]

        tag_dict[tagname] = Tag(tagname, index, tag_values)
    f.close()
    return tag_dict


def load_morphdata_ud(paras, tag_path="../data/", char_vocab=None,  use_sentence_markers=True):
    """
    :param paras: parameters passed along from argparse
    :param tag_path: directory with the '{paras.language}_tags_ud_filtered.txt' file
    :param char_vocab: optional CharacterGramVocab as a fixed vocab for the corpus (otherwise one will be created from the training data)
    :param use_sentence_markers: True to add begin and end of sentence markers to the data
    """
    all_labels = read_tags(os.path.join(tag_path, paras.language + "_tags_ud_filtered.txt"), lower=True,)  # all_labels = {'pos': iterator(num, s, a-pro, etc)}
    train_name = os.path.join(paras.data_path_ud, paras.language + "-ud-train.conllu")
    dev_name = os.path.join(paras.data_path_ud, paras.language + "-ud-dev.conllu")
    test_name = os.path.join(paras.data_path_ud, paras.language + "-ud-test.conllu")

    if char_vocab is None:
        char_vocab = CharacterGramVocab(gram=paras.char_gram)

    # Used for caching representations of words
    word_vocab = WordVocab(char_vocab)
    unique_pairs = {}

    # Max length of the word
    max_length = 200  # changed from 100 -> 200
    max_length_counter = [0]

    def parse_corpus(filename, name):
        x_data = []
        l_data = []
        y_data = []
        next_word_data = [] # Next word for LM predictions

        first_elem=True
        with codecs.open(filename, 'r', 'utf-8') as f:
            for line in f.readlines():
                if line.startswith("#"):
                    continue

                parts = line.strip().split("\t")

                if len(parts) > 1:
                    idx_in_sentence = int(parts[0].strip())
                    # Add end of last sentence, beginning of next sentence markers
                    if use_sentence_markers and idx_in_sentence==1:
                        # not the first sentence in data, add end sentence marker for previous sentence
                        if not first_elem:
                            # End of last sentence
                            x = np.zeros((max_length,), dtype=np.int32)
                            x[0:1] = np.asarray(char_vocab.mark_sentence(start=False))
                            y = np.zeros((len(all_labels),), dtype=np.int32)
                            x_data.append(x)
                            y_data.append(y)
                            l_data.append(1)
                            next_word_data.append(word_vocab.end_sentence_index)

                        # Beginning of sentence
                        x = np.zeros((max_length,), dtype=np.int32)
                        x[0:1] = np.asarray(char_vocab.mark_sentence(start=True))
                        y = np.zeros((len(all_labels),), dtype=np.int32)
                        x_data.append(x)
                        y_data.append(y)
                        l_data.append(1)
                        # First entry doesn't get added to next word prediction
                        if first_elem:
                            first_elem = False
                        else:
                            next_word_data.append(word_vocab.start_sentence_index)

                    word = parts[1].strip()
                    field_line = parts[3].strip()

                    if paras.unique_words:
                        if word in unique_pairs and unique_pairs[word] == field_line:
                            # naive field match is order of fields garantueed?
                            continue
                        else:
                            unique_pairs[word] = field_line

                    fields = field_line.split("|")

                    unique_pairs[parts[1].strip()] = parts[3].strip()
                    x = np.zeros((max_length,), dtype=np.int32)
                    y = np.zeros((len(all_labels),), dtype=np.int32)  # array of length one because we have just POS

                    # Caching the representations of words and indexing them for next word predictions
                    is_train = (name=="train")
                    word_index, res = word_vocab.lookup_word(word, is_train)
                    if len(res) > max_length_counter[0]:
                        max_length_counter[0] = len(res)

                    length = len(res)
                    x[0:length] = np.asarray(res)

                    field_dict = {}
                    # for field in fields:
                    #     if "=" in field:
                    #         parts = field.split("=")
                    #         field_dict[parts[0].lower()] = parts[1].lower()

                    field_dict['pos'] = fields[0].lower() # even for ambiguous cases just take the first one -> for A|S take A

                    for tag_name, tag_element in all_labels.items():
                        if tag_name in field_dict:
                            tag_value_index = tag_element.get_tag_index(field_dict[tag_name])
                            y[tag_element.index] = tag_value_index
                            if name == "train":
                                tag_element.add(field_dict[tag_name])

                        elif name == "train":
                            tag_element.add("_na_")

                    x_data.append(x)
                    l_data.append(length)
                    y_data.append(y)
                    # First entry doesn't get added to next word prediction
                    if first_elem:
                        first_elem = False
                    else:
                        next_word_data.append(word_index)

        # add last end of sentence marker
        if use_sentence_markers:
            x = np.zeros((max_length,), dtype=np.int32)
            x[0:1] = np.asarray(char_vocab.mark_sentence(start=False))
            y = np.zeros((len(all_labels),), dtype=np.int32)
            x_data.append(x)
            y_data.append(y)
            l_data.append(1)
            next_word_data.append(word_vocab.end_sentence_index)

        x = np.vstack(x_data)
        lengths = np.asarray(l_data)
        y = np.vstack(y_data)
        y_next_word = np.array(next_word_data)

        return x[:, :max_length_counter[0]], lengths, y, y_next_word



    train_x, train_lengths, train_y, train_y_next_word = parse_corpus(train_name, "train")
    dev_x, dev_lengths, dev_y, dev_y_next_word = parse_corpus(dev_name, "dev")
    test_x, test_lengths, test_y, test_y_next_word = parse_corpus(test_name, "test")

    return train_x, train_lengths, train_y, train_y_next_word, dev_x, dev_lengths, dev_y, dev_y_next_word, test_x, test_lengths, test_y, test_y_next_word, char_vocab, all_labels, word_vocab


class DataIterator:
    def __init__(self, x, lengths, y, batch_size, train=False):

        self.x = x # Words
        self.lengths = lengths # Lengths of words
        self.y = y # Labels

        self.batch_size = batch_size

        # Actually number of words!
        self.number_of_sentences = x.shape[0]
        if train:
            self.n_batches = self.number_of_sentences // self.batch_size
        else:
            self.n_batches = math.ceil(self.number_of_sentences / self.batch_size)
        self.train = train

    def __iter__(self):
        if self.train:
            indexes = batching_seed.permutation(np.arange(self.number_of_sentences))
        else:
            indexes = np.arange(self.number_of_sentences)

        for i in range(self.n_batches):
            lengths = self.lengths[indexes[i * self.batch_size:(i + 1) * self.batch_size]]
            y_batch = self.y[indexes[i * self.batch_size:(i + 1) * self.batch_size]]

            x_batch = self.x[indexes[i * self.batch_size:(i + 1) * self.batch_size],
                      :]

            yield x_batch, y_batch, lengths