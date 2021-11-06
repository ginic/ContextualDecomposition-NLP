import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sklearn.metrics
import argparse
import numpy as np
import os
import time
import codecs
import pickle
import sys
sys.path.append("../")
import networks as networks
import data_iterator as data_iterator

# TODO clean up arguments

np.random.seed(2345)

#  SETTINGS
parser = argparse.ArgumentParser(description='Morpho tagging Pytorch version.')

# which type of network
parser.add_argument("--char_type", type=str, default="conv", help="Character 'bilstm', 'conv' or 'sum'")

# input
parser.add_argument("--char_embedding_size", type=int, default=50, help="Character embedding size")
parser.add_argument("--char_gram", type=int, default=1, help="Character gram")
# bilstm char
parser.add_argument("--char_rec_num_units", type=int, default=100, help="Word or char")
# conv char
parser.add_argument("--char_filter_sizes", type=int, nargs='+', default=[1,2,3,4,5,6], help="Width of each filter")
parser.add_argument("--char_number_of_filters", type=int, nargs='+', default=[25,50,75,100,125,150],
                    help="Total number of filters")
parser.add_argument("--char_conv_act", type=str, default="relu", help="Default is relu, tanh is the other option")

# training
parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to run")
parser.add_argument("--dropout_frac", type=float, default=0., help="Optional dropout")


# dataset
parser.add_argument("--language", type=str, default="ru", help="Russian (ru)")
parser.add_argument("--unique_words", type=int, default=1, help="Use unique words rather than all words")
parser.add_argument("--data_path_ud", type=str, required=True,
                    help="Where can I find the datafiles of UD1.4: *-ud-train.conllu, "
                         "*-ud-dev.conllu and *-ud-test.conllu")
parser.add_argument("--save_dir", type=str, required=True,help="Directory to save models")
parser.add_argument("--save_file", type=str, default="tagger_")

paras = parser.parse_args()

def predict(model, data_iterator, is_cuda_available, is_verbose, paras):
    """Uses the given model and paramters to predict labels.
    Prints accuracy, count of correct labels and classification metrics for predications on the data iterator.
    Returns list of accuracies (for each tag) and list of counts of correct labels.

    :param model: pytorch model
    :param paras: parameters passed via argparse
    :param is_cuda_available: boolean, whether or not to use GPU
    :param is_verbose: boolean, True to print full classification report rather than just accuracy
    :param data_iterator: data_iterator.DataIterator object that yields sentences, tags and lengths of sentences
    """
    model.eval()

    # Track total number of valid tags of all types in the entire dataset
    total_valid = 0
    correct_valid = [0 for _ in range(len(paras.tagset_size))]

    # Use to store predictions and labels for each tag set, then compute metrics later using all sentences
    all_predictions = [[] * len(paras.tagset_size)]
    gold_labels = [[] * len(paras.tagset_size)]
    for sentences, tags, lengths in data_iterator:
        # set gradients zero
        model.zero_grad()
        # run model
        tag_scores = model(sentences, lengths)
        # calculate loss and backprop
        for tagtype_index in range(tags.shape[1]):
            gt = tags[:, tagtype_index]
            gold_labels[tagtype_index].extend(gt)
            if is_cuda_available:
                predictions = torch.max(tag_scores[tagtype_index],dim=1)[1].cuda().data.numpy()
            else:
                predictions = torch.max(tag_scores[tagtype_index],dim=1)[1].cpu().data.numpy()

            all_predictions[tagtype_index].extend(all_predictions)
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
            print(sklearn.metrics.classification_report(gold_labels[i], all_predictions[i]))

    return correct_valid, accuracies

def main(paras):
    """
    :param paras: parameters passed along from argparse
    """
    # load data
    train_x, train_lengths, train_y, valid_x, valid_lengths, valid_y, test_x, test_lengths, test_y, char_vocab, tag_dict \
        = data_iterator.load_morphdata_ud(paras)

    paras.save_file += paras.language + "_"

    paras.char_vocab_size = len(char_vocab.vocab)
    paras.tagset_size = dict([(t.index,len(t.values)) for t in tag_dict.values()])
    paras.pad_index = char_vocab.pad_index

    # iterators for data
    train_it = data_iterator.DataIterator(train_x, train_lengths, train_y, paras.batch_size, train=True)
    valid_it = data_iterator.DataIterator(valid_x, valid_lengths, valid_y, paras.batch_size)
    test_it = data_iterator.DataIterator(test_x, test_lengths, test_y, paras.batch_size)

    # make model
    is_cuda_available = torch.cuda.is_available()
    model = networks.Tagger(paras)
    model.apply(networks.init_ortho)
    if is_cuda_available:
        model.cuda()
    else:
        model.cpu()

    # loss function
    loss_functions = {}
    for tag_name, tag_element in tag_dict.items():
            loss_functions[tag_element.index] = nn.CrossEntropyLoss()


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
    best_valid = 0
    best_acc = []
    best_path = ""

    print(paras)
    print("Started training")
    for epoch in range(paras.num_epochs):

        ##################
        # training       #
        ##################
        total_loss = 0

        model.train()
        for sentences, tags, lengths in train_it:
            # set gradients zero
            model.zero_grad()
            # run model
            tag_scores = model(sentences, lengths)
            # calculate loss and backprop
            loss = []
            for tagtype_index in range(tags.shape[1]):
                if is_cuda_available:
                    gt = Variable(torch.LongTensor(tags[:,tagtype_index]).cuda())
                else:
                    gt = Variable(torch.LongTensor(tags[:,tagtype_index]).cpu())
                loss.append(loss_functions[tagtype_index](tag_scores[tagtype_index], gt))

            if is_cuda_available:
                total_loss+=sum([l.data.cuda().numpy() for l in loss])
            else:
                total_loss+=sum([l.data.cpu().numpy() for l in loss])

            sum(loss).backward()
            optimizer.step()

        print("Epoch %s: train loss %s" % (epoch + 1, total_loss / train_it.n_batches))

        ##################
        # validation     #
        ##################
        print("Validation results")
        correct_valid, val_accs = predict(model, valid_it, is_cuda_available, False, paras)

        if sum(correct_valid) > best_valid:
            best_valid = sum(correct_valid)
            best_path = os.path.join(paras.save_dir, save_file_model + "_best")
            torch.save(model.state_dict(), best_path)
            best_acc = val_accs
            print("New best")

    print("Best total number of correct tags is: %s" % (best_valid))
    print("Best model's accuracy by tag is:", best_acc)
    torch.save(model.state_dict(), os.path.join(paras.save_dir, save_file_model + "_last"))

    print("Loading best model from", best_path)
    model.load_state_dict(torch.load(best_path))
    print("Evaluating on test set")
    test_correct_valid, test_accs = predict(model, test_it, is_cuda_available, True, paras)


if __name__ == "__main__":
    main(paras)
