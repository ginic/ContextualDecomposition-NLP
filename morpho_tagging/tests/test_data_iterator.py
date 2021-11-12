from ..data_iterator import *
from ..train import parser

TEST_PARAMS = parser.parse_args(["--language", "en", "--data_path_ud", "tests/tiny_test_data", "--save_dir", "noop"])

def test_load_morphdata_us():
    train_x, train_lengths, train_y, train_next_word, dev_x, dev_lengths, dev_y, dev_next_word, test_x, test_lengths, test_y, test_next_word, char_vocab, all_labels = load_morphdata_ud(TEST_PARAMS, use_sentence_markers=False)

    # 86 words in training set
    assert train_x.shape[0] == 86
    assert train_y.shape[0] == 86
    assert train_lengths.shape == (86,)
    assert train_next_word.shape == (85,)

    assert dev_x.shape[0] == 15
    assert dev_y.shape[0] == 15
    assert dev_lengths.shape == (15,)
    assert dev_next_word.shape == (14,)

    assert test_x.shape[0] == 18
    assert test_y.shape[0] == 18
    assert test_lengths.shape == (18,)
    assert test_next_word.shape == (17,)

    assert list(all_labels.keys()) == ["pos"]


def test_load_morphdata_with_sentence_markers():
    train_x, train_lengths, train_y, train_next_word, dev_x, dev_lengths, dev_y, dev_next_word, test_x, test_lengths, test_y, test_next_word, char_vocab, all_labels = load_morphdata_ud(TEST_PARAMS, use_sentence_markers=True)

    # 86 words in training set
    assert train_x.shape[0] == 98
    assert train_y.shape[0] == 98
    assert train_lengths.shape == (98,)
    assert train_next_word.shape == (97,)

    assert dev_x.shape[0] == 17
    assert dev_y.shape[0] == 17
    assert dev_lengths.shape == (17,)
    assert dev_next_word.shape == (16,)

    assert test_x.shape[0] == 20
    assert test_y.shape[0] == 20
    assert test_lengths.shape == (20,)
    assert test_next_word.shape == (19,)


def test_char_vocab():
    char_vocab = CharacterGramVocab()
    # add characters to vocab
    word_res = char_vocab.add_string("word", add_eow=True, add_sow=True)
    assert word_res == [3, 4, 5, 6, 7, 8]

    # Check vocab, including unknown chars
    assert char_vocab.string_to_index("words") == [4, 5, 6, 7, 1]



