# Introduction
This repository is a component of our final project for CS 682: Neural Networks: A Modern Introduction at UMass Amherst in Fall 2021. The code from ["Explaining Character-Aware Neural Networks for Word-Level Prediction:
Do They Discover Linguistic Rules?"](http://aclweb.org/anthology/D18-1365) served as the basis for the project and this fork their code modified for our purposes. See [our final project's main repository](https://github.com/ginic/CS682_NNs_Nerual_Char_Models_Russian) for more details.

# Framework
All code was implemented in Python 3.7. We used Pytorch 1.10.0.

# Data
Our training data was from the [Russian National Corpus](https://ruscorpora.ru/new/en/index.html) and [OpenCorpora](https://www.kaggle.com/rtatman/opencorpora-russian) converted to CONLL-U format. It should also work with any [Universal Dependency Treebank](https://universaldependencies.org) data.

All files of type *-ud-train.conllu, *-ud-dev.conllu and *-ud-test.conllu, should be place in a single data folder DATA_PATH_UD.

All other necessary files are provided in the data-folder of this project.

# Training
- TODO Add details about pretraining vs fine tuning
- TODO Test set evaluation is done at the end of tuning

Navigate to the root of this repo (the ContextualDecomposition-NLP folder.)
The following command can be used to train a model on the UD dataset.

```
CUDA_VISIBLE_DEVICES=0 python3 -m morpho_tagging.train --data_path_ud DATA_PATH_UD --save_dir SAVE_DIR --language LANGUAGE --char_type CONV_OR_BILSTM
```

The most important values to fill in:
- DATA_PATH_UD: the path of the Universal Dependencies Dataset
- SAVE_DIR: where to save the model and metadata
- LANGUAGE: 'fi' for Finnish, 'es' for Spanish or 'sv' for Swedish. Note that models for all UD languages can be trained and decomposed.
- CONV_OR_BILSTM: either train a CNN ('conv') or a BiLSTM ('bilstm') model

A full overview of all the parameters can be obtained using:
```
CUDA_VISIBLE_DEVICES=0 python3 train.py --help
```
# Evaluation
- TODO Update evalution instructions for our work

Move into the folder 'contextual_decomposition'.
Run the following command to evaluate the contextual decomposition for CNNs.
```
python3 evaluate_segmentation.py --settings_name tagger_LANGUAGE_settings_DATE --model_folder SAVE_DIR
```

The parameters to provide are:
- settings_name: name of the model you have trained of the format tagger_LANGUAGE_settings_DATE
- model_folder: SAVE_DIR value from during training

The output will be an overview of the correct/incorrect predictions and attributions of the trained model.
This is the same algorithm as used in the paper for evaluation.

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