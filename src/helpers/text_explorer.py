import re
import pickle
import nltk
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from def_text_summ import ROOT_DIR


def explore_text_and_summaries(text_path, summaries_path, 
                               text_word_tok=False, text_word_tok_path=None):

    print("-"*10)
    print('Basic parameters of texts and summaries:')
    print("-"*10)

    # load files
    with open(ROOT_DIR + text_path, 'rb') as file:
        text = pickle.load(file)

    with open(ROOT_DIR + summaries_path, 'rb') as file:
        summaries = pickle.load(file)

    # ensure number of samples match each other
    assert len(text) == len(summaries), "Number of text samples doesn't match number of summaries samples."
    print("Length of text is {}".format(len(text)))
    print("-"*10)

    # visually check that summary relates to the text
    check_number = np.random.randint(0, len(text))
    print('Comparison of text to its summary:')
    print('#'*10+'TEXT'+'#'*10)
    print(text[check_number])
    print('#'*10+'SUMMARY'+'#'*10)
    print(summaries[check_number])
    print("-"*10)

    # Explore number of tokens in samples of text and summaries
    if not text_word_tok:
        text_tokenized = [nltk.word_tokenize(sample) for sample in text]
    else: 
        with open(ROOT_DIR + text_word_tok_path, 'rb') as file:
            text_tokenized = pickle.load(file)
    
    summaries_tokenized = [nltk.word_tokenize(sample) for sample in summaries]

    text_tokens_count = [len(sample) for sample in text_tokenized]
    summaries_tokens_count = [len(sample) for sample in summaries_tokenized]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
    sns.distplot(text_tokens_count, bins=100, ax=axes[0]).set_title('Distribution of number of words in a text')
    sns.distplot(summaries_tokens_count, bins=500, ax=axes[1]).set_title('Distribution of number of words in summaries of a text')
    axes[1].set_xlim(0,300)
    plt.show()

    print("-"*10)

    # Explore number of sentences in samples of training set
    text_sent_tokenized = [nltk.sent_tokenize(sample) for sample in text]
    summaries_sent_tokenized = [nltk.sent_tokenize(sample) for sample in summaries]

    text_tokens_sent_count = [len(sample) for sample in text_sent_tokenized]
    summaries_tokens_sent_count = [len(sample) for sample in summaries_sent_tokenized]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
    axes = axes.flatten()
    sns.distplot(text_tokens_sent_count, bins=100, ax=axes[0]).set_title('Distribution of number of sentences in a text')
    sns.distplot(summaries_tokens_sent_count, bins=500, ax=axes[1]).set_title('Distribution of number of sentences in summaries of a text')
    axes[1].set_xlim(0,20)
    plt.show()
