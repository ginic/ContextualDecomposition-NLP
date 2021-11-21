# Introduction
This repository is a component of our final project for CS 682: Neural Networks: A Modern Introduction at UMass Amherst in Fall 2021. The code from ["Explaining Character-Aware Neural Networks for Word-Level Prediction:
Do They Discover Linguistic Rules?"](http://aclweb.org/anthology/D18-1365) served as the basis for the project and this fork their code modified for our purposes. See [our final project's main repository](https://github.com/ginic/CS682_NNs_Nerual_Char_Models_Russian) for more details.

# Framework
All code was implemented in Python 3.7. We used Pytorch 1.10.0.

# Data
Our training data was from the [Russian National Corpus](https://ruscorpora.ru/new/en/index.html) and [OpenCorpora](https://www.kaggle.com/rtatman/opencorpora-russian) converted to CONLL-U format. It should also work with any [Universal Dependency Treebank](https://universaldependencies.org) data.

All files of type *-ud-train.conllu, *-ud-dev.conllu and *-ud-test.conllu, should be place in a single data folder DATA_PATH_UD.

All other necessary files are provided in the `data` folder of this project.

# Training
Navigate to the root of this repo (the ContextualDecomposition-NLP folder.)
The following command can be used to train a language model on the CONLL-U (Universal Dependency Treebank) format dataset.
The script is in `morpho_tagging/train.py`, but you should run it as a module for the imports to work correctly.

A full overview of all the parameters can be obtained using:
```
python -m morpho_tagging.train --help
```

## Pre-training a model
Run the script with the `--training_type lm` option ('lm' for 'language model'). This trains the character embeddings by learning to predict the next word. You can also still use `--training_type label` to directly target the POS tags.
```
python -m morpho_tagging.train --training_type lm --data_path_ud DATA_PATH_UD --save_dir SAVE_DIR --language LANGUAGE --char_type CONV_OR_BILSTM --batch_size BATCH_SIZE
```

The most important values to fill in:
- DATA_PATH_UD: the path of the Universal Dependencies Dataset
- SAVE_DIR: where to save the model and metadata
- LANGUAGE: defaults to 'ru' for Russian, must match the language code in the .conllu file names
- CONV_OR_BILSTM: either train a CNN ('conv') or a BiLSTM ('bilstm') model for the characters
- BATCH_SIZE: each batch is a sequence of words, so batch size is the length of word sequences when training models, defaults to 20


## Using pre-trained character embeddings to train a POS tagger
Run the script with the `--training_type label` option and give `--pretrained_model PRETRAINED_MODEL` path and `--pretrained_settings PRETRAINED_SETTINGS` file path to the model to load.

```
python -m morpho_tagging.train --data_path_ud DATA_UD_OATH --save_dir SAVE_DIR --language en --training_type label --pretrained_model PRETRAINED_MODEL --pretrained_settings PRETRAINED_SETTINGS
```

Note that these parameters are overridden by the pre-trained model, even if you include them in arguments (otherwise the dimensions might not match): `['char_type', 'char_embedding_size', 'char_gram', 'char_rec_num_units', 'char_filter_sizes', 'char_number_of_filters', 'char_conv_act']`


# Evaluation
Evaluation on the test set is done as the final step in training.
For the `--training_type lm`, perplexity is reported.
For `--training_type label`, accuracy and a detailed classification report (precision, recall, F1) by tag value is reported.

# Unit tests
Unit tests can be run using `python -m pytest` from within the root directory.

# Citation
Our code is based on following publication's:

```
@InProceedings{D18-1365,
  author = 	"Godin, Fr{\'e}deric
		and Demuynck, Kris
		and Dambre, Joni
		and De Neve, Wesley
		and Demeester, Thomas",
  title = 	"Explaining Character-Aware Neural Networks for Word-Level Prediction: Do They Discover Linguistic Rules?",
  booktitle = 	"Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"3275--3284",
  location = 	"Brussels, Belgium",
  url = 	"http://aclweb.org/anthology/D18-1365"
}
```