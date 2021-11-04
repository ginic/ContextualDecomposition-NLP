"""Offers a naive baseline model that just predicts POS tag based on the most common tag for a wordform in the training data.
"""
import argparse
import codecs
from collections import Counter, defaultdict
import os

from sklearn import metrics

import data_iterator as data_iterator

def classify_and_report(x, y, overall_pos_counts, wordform_pos_counter, labels):
    """Classify POS tags for x data using the naive strategy and
    report metrics
    """
    y_pred = naive_pos_labeller(x, overall_pos_counts, wordform_pos_counter)
    print("Accuracy", metrics.accuracy_score(y, y_pred))
    print(metrics.classification_report(y, y_pred, labels = labels, zero_division=0))


def count_training_labels(x_train, y_train):
    """Counts up the most common labels for each wordform in training, returning mapping for each word in the corpus, {wordform -> Counter[POS]=count} and overall POS counts in the corpus, Counter[POS]=count
    """
    overall_pos_counts = Counter()
    wordform_pos_counter = defaultdict(Counter)
    for word, tag in zip(x_train, y_train):
        overall_pos_counts[tag] += 1
        wordform_pos_counter[word][tag] += 1
    return overall_pos_counts, wordform_pos_counter


def naive_pos_labeller(x, overall_pos_counts, wordform_pos_counter):
    """Returns POS predictions for x as a list, given the counts from the training data.
    If the wordform is present in the training data, assign the most common label for that word. Otherwise, assign the most common label overall.
    :param x: list of words
    :param overall_post_counts: counts of POS tags from training
    :param wordform_pos_counter: counts of POS tags for each word from training
    """
    most_common_label,_ = overall_pos_counts.most_common(1)[0]
    predictions = []
    for word in x:
        if word in wordform_pos_counter:
            label,_ = wordform_pos_counter[word].most_common(1)[0]
            predictions.append(label)
        else:
            predictions.append(most_common_label)
    return predictions


def parse_corpus(conllu_path, valid_tags):
    """Parses a corpus, returning x_data with words and y_data with POS tags
    :param conllu_path: Path to data file
    :param valid_tags:
    """
    x_data = []
    y_data = []
    # This mostly matches the data_iterator parse_corpus method, but returns words as elements rather than characters
    with codecs.open(conllu_path, 'r', 'utf-8') as f:
        for line in f.readlines():
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")

            if len(parts) > 1:
                word = parts[1].strip()
                field_line = parts[3].strip()
                fields = field_line.split("|")
                parsed_pos = fields[0].lower() # even for ambiguous cases just take the first one -> for A|S take A
                if parsed_pos in valid_tags:
                    pos = parsed_pos
                else:
                    pos = "_na_"
                x_data.append(word)
                y_data.append(pos)

    return x_data, y_data


def load_data(data_path_ud, language, valid_tags):
    """Returns x_train, y_train, x_dev, y_dev, x_test, y_test
    """
    train_name = os.path.join(data_path_ud, language + "-ud-train.conllu")
    x_train, y_train = parse_corpus(train_name, valid_tags)
    dev_name = os.path.join(data_path_ud, language + "-ud-dev.conllu")
    x_dev, y_dev = parse_corpus(dev_name, valid_tags)
    test_name = os.path.join(data_path_ud, language + "-ud-test.conllu")
    x_test, y_test = parse_corpus(test_name, valid_tags)
    return x_train, y_train, x_dev, y_dev, x_test, y_test


def main(data_path_ud, language):
    labels = data_iterator.read_tags(os.path.join("../data/", language + "_tags_ud_filtered.txt"), lower=True)
    pos_labels = labels['pos'].values
    print("Loaded part of speech tags:", pos_labels)
    x_train, y_train, x_dev, y_dev, x_test, y_test = load_data(data_path_ud, language, pos_labels)
    overall_pos_counts, wordform_pos_counter = count_training_labels(x_train, y_train)
    print("Overall label counts in training data:", overall_pos_counts)
    print("Sample of POS tags for individual wordforms:")
    flag = 0
    for word, counts in wordform_pos_counter.items():
        print("\tWord:", word, "Label counts:", counts)
        flag += 1
        if flag > 4:
            break

    print("Classification results on dev set:")
    classify_and_report(x_dev, y_dev, overall_pos_counts, wordform_pos_counter, pos_labels)

    print("Classification results on test set:")
    classify_and_report(x_test, y_test, overall_pos_counts, wordform_pos_counter, pos_labels)


parser = argparse.ArgumentParser(description="Naive POS tagging baseline")

parser.add_argument("--data_path_ud", type=str, required=True,
                    help="Where can I find the datafiles of UD1.4: *-ud-train.conllu, "
                         "*-ud-dev.conllu and *-ud-test.conllu")
parser.add_argument("--language", type=str, default="ru", help="Russian (ru)")

if __name__=="__main__":
    args = parser.parse_args()
    main(args.data_path_ud, args.language)



