import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sklearn.metrics
import argparse
from datetime import datetime
import numpy as np
import os
import time
import codecs
import pickle

import networks
import data_iterator

CLF_MODEL = "label"
LANG_MODEL = "lm"


np.random.seed(2345)

#  SETTINGS
parser = argparse.ArgumentParser(description='Morpho tagging Pytorch version.')

# which type of network for characters
parser.add_argument("--char_type", type=str, default="conv", help="Character 'bilstm', 'conv' or 'sum'")

# input
parser.add_argument("--char_embedding_size", type=int, default=50, help="Character embedding size")
parser.add_argument("--char_gram", type=int, default=1, help="Character gram")
# bilstm char
parser.add_argument("--char_rec_num_units", type=int, default=100, help="Size of hidden layers using characters")
# conv char
parser.add_argument("--char_filter_sizes", type=int, nargs='+', default=[1,2,3,4,5,6], help="Width of each filter")
parser.add_argument("--char_number_of_filters", type=int, nargs='+', default=[25,50,75,100,125,150],
                    help="Total number of filters")
parser.add_argument("--char_conv_act", type=str, default="relu", help="Default is relu, tanh is the other option")

# training
parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to run")
parser.add_argument("--dropout_frac", type=float, default=0., help="Optional dropout for embeddings")


# dataset
parser.add_argument("--language", type=str, default="ru", help="Russian (ru)")
parser.add_argument("--unique_words", type=int, default=0, help="Use unique words rather than all words.")
parser.add_argument("--data_path_ud", type=str, required=True,
                    help="Where can I find the datafiles of UD1.4: *-ud-train.conllu, "
                         "*-ud-dev.conllu and *-ud-test.conllu")
parser.add_argument("--save_dir", type=str, required=False,help="Directory to save models")
parser.add_argument("--save_file", type=str, default="tagger_")

# Added arguments for pre-training & fine-tuning
parser.add_argument("--training_type", type=str, choices=[CLF_MODEL, LANG_MODEL], help="To pre-train a language model, use 'lm'. To fine-tune a pre-trained model or train a new model for for word-level labeling, use 'label'" )
parser.add_argument("--pretrained_model", type=str, help="Path to a pre-trained model when you are fine-tuning a for POS tagging.")
parser.add_argument("--lm_hidden_size", type=int, default=100, help="Number of hidden units in language model LSTM for next word prediction")
parser.add_argument("--lm_num_layers", type=int, default=1, help="Number of LSTM layers in the language model layer")

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history.
    See https://github.com/pytorch/examples/blob/3970e068c7f18d2d54db2afee6ddd81ef3f93c24/word_language_model/main.py#L112 and https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426 for more details.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def check_new_best(validation_result, prev_best_score, prev_best_path, training_type, save_dir, save_file_model, model):
    """Given the type of training and result on the validation set, decide if this is a new best model or not. If it is, save the model.
    Returns True/False if this is the new best score, the best score overall, the path to the best model

    :param validation_result: the result from the prediction function
    :param prev_best_score: the best score thus far
    :param prev_best_path: path to the best model thus far
    :param training_type: "lm" or "label"
    :param save_dir: path to save directory for model
    :param save_file_model: prefix for model's file name
    :param model: pytorch model with state to save
    """
    is_new_best = False
    best_valid = prev_best_score
    best_path = prev_best_path
    if training_type==CLF_MODEL and validation_result > prev_best_score:
        # In labeling, check raw number of correct labels
        best_valid = validation_result
        is_new_best = True
    elif paras.training_type == LANG_MODEL and validation_result < prev_best_score:
        # In languag modeling, check that the loss/perplexity has decreased
        best_valid = validation_result
        is_new_best = True

    if is_new_best:
        best_path = os.path.join(paras.save_dir, save_file_model + "_best")
        torch.save(model.state_dict(), best_path)
        print("New best")

    return is_new_best, best_valid, best_path

def next_word_prediction(model, data_iterator, device):
    """Uses the given model and parameters to predict the next word in each sentence.
    Prints probability and perplexity of data set under the model.
    :param model: pytorch model
    :param paras: parameters passed via argparse
    :param is_cuda_available: boolean, whether or not to use GPU
    """
    model.eval()
    model.zero_grad()
    prediction_probabilities = []
    actual_next_words = []
    hidden = model.init_lm_hidden()
    for words, tags, lengths in data_iterator:
        actual_next_words.extend(tags)
        word_scores, hidden = model(words, lengths, hidden)
        hidden = repackage_hidden(hidden)
        prediction_probabilities.append(word_scores[0])

    all_predictions = torch.stack(prediction_probabilities).squeeze()
    truth = torch.LongTensor(actual_next_words).to(device).detach()
    total_loss = nn.CrossEntropyLoss()(all_predictions, truth).detach().numpy()
    data_loss = total_loss / len(actual_next_words)

    print("Loss:", data_loss, "Perplexity:", np.exp(data_loss))

    return data_loss


def predict(model, data_iterator, device, is_verbose, paras, labels=None):
    """Uses the given model and parameters to predict labels.
    Prints accuracy, count of correct labels and classification metrics for predications on the data iterator.
    Returns list of accuracies (for each tag) and list of counts of correct labels.

    :param model: pytorch model
    :param paras: parameters passed via argparse
    :param is_cuda_available: boolean, whether or not to use GPU
    :param is_verbose: boolean, True to print full classification report rather than just accuracy
    :param data_iterator: data_iterator.DataIterator object that yields sentences, tags and lengths of sentences
    :param labels: optional list of lists of tagset labels correpsonding to the classes for each tagset, used for classification report
    """
    model.eval()

    # Track total number of valid tags of all types in the entire dataset
    total_valid = 0
    correct_valid = [0 for _ in range(len(paras.tagset_size))]
    hidden = model.init_lm_hidden()
    # Use to store predictions and labels for each tag set, then compute metrics later using all sentences
    all_predictions = [[] * len(paras.tagset_size)]
    gold_labels = [[] * len(paras.tagset_size)]
    for words, tags, lengths in data_iterator:
        # set gradients zero
        model.zero_grad()
        # run model
        tag_scores = model(words, lengths, hidden)
        hidden = repackage_hidden(hidden)
        # calculate loss and backprop
        for tagtype_index in range(tags.shape[1]):
            gt = tags[:, tagtype_index]
            gold_labels[tagtype_index].extend(gt)
            predictions = torch.max(tag_scores[tagtype_index],dim=1)[1].to(device).data.numpy()
            all_predictions[tagtype_index].extend(predictions)

            correct_valid[tagtype_index]+=sum(np.equal(gt,predictions))
        total_valid+=tags.shape[0]

    print("Total valid tags in dataset:", total_valid)
    accuracies = []

    for i in range(len(paras.tagset_size)):
        tag_acc = sklearn.metrics.accuracy_score(gold_labels[i], all_predictions[i])
        accuracies.append(tag_acc)
        print(f"Tagset {i} accuracy:", tag_acc)
        if is_verbose:
            print(f"Tagset {i} classification report:")
            if labels:
                index_labels = list(range(len(labels[i])))
                print(sklearn.metrics.classification_report(gold_labels[i], all_predictions[i], zero_division=0, labels=index_labels, target_names = labels[i]))
            else:
                print(sklearn.metrics.classification_report(gold_labels[i], all_predictions[i], zero_division=0))


    return correct_valid, accuracies

def main(paras):
    """
    :param paras: parameters passed along from argparse
    """
    # load data
    train_x, train_lengths, train_y_labels, train_next_word, valid_x, valid_lengths, valid_y_labels, valid_next_word, test_x, test_lengths, test_y_labels, test_next_word, char_vocab, tag_dict, word_vocab = data_iterator.load_morphdata_ud(paras)

    # If you're doing language modeling, the last word in the set doesn't matter
    if paras.training_type == LANG_MODEL:
        train_x = train_x[:-1]
        valid_x = valid_x[:-1]
        test_x = test_x[:-1]
        train_y = train_next_word
        valid_y = valid_next_word
        test_y = test_next_word
    else:
        train_y = train_y_labels
        valid_y = valid_y_labels
        test_y = test_y_labels

    paras.save_file += paras.language + "_"

    paras.char_vocab_size = len(char_vocab.vocab)
    paras.word_vocab_size = word_vocab.vocab_size()
    paras.tagset_size = dict([(t.index,len(t.values)) for t in tag_dict.values()])
    paras.pad_index = char_vocab.pad_index

    # iterators for data
    train_it = data_iterator.DataIterator(train_x, train_lengths, train_y, paras.batch_size, train=True)
    valid_it = data_iterator.DataIterator(valid_x, valid_lengths, valid_y, paras.batch_size)
    test_it = data_iterator.DataIterator(test_x, test_lengths, test_y, paras.batch_size)

    # make model
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = networks.Tagger(paras, device)


    if paras.pretrained_model is not None:
        # Load the pre-trained model, then update its training type parameters
        model.load_state_dict(torch.load(paras.pretrained_model))
        model.paras.training_type = paras.training_type
        # TODO What other parameters should be updateable at fine-tuning time?

    else:
        # Instantiate new model from scratch
        model = networks.Tagger(paras, device)
        model.apply(networks.init_ortho)

    model.to(device)

    hidden = model.init_lm_hidden()

    # loss function {index -> pytorch loss function}
    loss_functions = {}
    if paras.training_type==CLF_MODEL:
        for tag_name, tag_element in tag_dict.items():
                loss_functions[tag_element.index] = nn.CrossEntropyLoss()
    elif paras.training_type==LANG_MODEL:
        loss_functions = {LANG_MODEL: nn.CrossEntropyLoss()}

    # optimizer
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=paras.lr)

    # print total number of parameters
    parameters = model.parameters()
    sum_params = sum([np.prod(p.size()) for p in parameters])
    print("Number of parameters: %s " % (sum_params))

    print("Store settings")
    start_time_str = time.strftime("%d_%b_%Y_%H_%M_%S")
    save_file_model = paras.save_file + "data_" + start_time_str
    save_file_settings = paras.save_file + "settings_" + start_time_str
    save_file_vocab = paras.save_file + "vocab_" + start_time_str
    file = codecs.open(os.path.join(paras.save_dir, save_file_settings), "w")
    file.write(str(vars(paras)) + "\n")
    file.close()

    file = codecs.open(os.path.join(paras.save_dir, save_file_vocab), "wb")
    pickle.dump([char_vocab],file)
    file.close()

    # Use for tracking best model performance and save path
    if paras.training_type == CLF_MODEL:
        best_valid = 0
    elif paras.training_type == LANG_MODEL:
        best_valid = float('inf')
    # A list of validation accuracy scores by label (only for training_type=="label")
    best_val_accs = []
    best_path = ""

    print(paras)
    print("Started training")
    for epoch in range(paras.num_epochs):
        ##################
        # training       #
        ##################
        start = datetime.now()
        total_loss = 0

        model.train()
        for sentences, tags, lengths in train_it:
            # set gradients zero
            model.zero_grad()
            # run model (forward pass)
            hidden = repackage_hidden(hidden)
            tag_scores, hidden = model(sentences, lengths, hidden)
            # calculate loss and backprop
            # List of losses of each example for each label
            loss = []
            if paras.training_type == CLF_MODEL:
                for tagtype_index in range(tags.shape[1]):
                    gt = Variable(torch.LongTensor(tags[:,tagtype_index]).to(device))
                    loss.append(loss_functions[tagtype_index](tag_scores[tagtype_index], gt))
            elif paras.training_type == LANG_MODEL:
                word_scores = tag_scores[0].squeeze()
                gt = Variable(torch.LongTensor(tags).to(device))
                loss.append(loss_functions[LANG_MODEL](word_scores, gt))

            total_loss+=sum([l.data.to(device).numpy() for l in loss])

            sum(loss).backward()
            optimizer.step()
            end = datetime.now()


        print("Epoch %s: train loss %s" % (epoch + 1, total_loss / train_it.n_batches))
        print("Epoch train time in minutes:", (end - start).total_seconds()/60)

        ##################
        # validation     #
        ##################
        print("Validation results")
        if paras.training_type == CLF_MODEL:
            correct_valid, val_accs = predict(model, valid_it, device, False, paras)
            validation_result = sum(correct_valid)
        elif paras.training_type == LANG_MODEL:
            validation_result = next_word_prediction(model, valid_it, device)

        is_new_best, best_valid, best_path = check_new_best(validation_result, best_valid, best_path, paras.training_type, paras.save_dir, save_file_model, model)

        if paras.training_type == CLF_MODEL and is_new_best:
            best_val_accs = val_accs

    if paras.training_type == CLF_MODEL:
        print("Best total number of correct tags is:", best_valid)
        print("Best model's score by tag is:", best_val_accs)
    elif paras.training_type == LANG_MODEL:
        print("Best validation loss is:", best_valid)

    torch.save(model.state_dict(), os.path.join(paras.save_dir, save_file_model + "_last"))

    print("Loading best model from", best_path)
    model.load_state_dict(torch.load(best_path))
    print("Evaluating on test set")
    labels = [t.values for t in tag_dict.values()]
    if paras.training_type == CLF_MODEL:
        test_correct_valid, test_accs = predict(model, test_it, device, True, paras, labels=labels)
    elif paras.training_type == LANG_MODEL:
        test_result = next_word_prediction(model, test_it, device)


if __name__ == "__main__":
    paras = parser.parse_args()
    main(paras)
