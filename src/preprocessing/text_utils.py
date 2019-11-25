
import pickle
import nltk
import numpy as np
import tensorflow as tf
from def_text_summ import ROOT_DIR

    
def fit_tokenizer(path_to_text_list, vocab_size=20000, save=False, path_to_tokenizer=None):

    """
    https://github.com/keras-team/keras/issues/8092#issuecomment-372833486
    """
    
    """Function that wraps Keras Tokenizer fit on text and its saving for further use
    
    :path_to_text_list column: path to text 
    :type column: string
    :param vocab_size: size of vocabulary, defaults to 20000
    :type vocab_size: int, optional
    :param save: whether to save fitted tokenizer, defaults to False
    :type save: bool, optional
    :param path_to_tokenizer: file path for saving trained tokenizer, defaults to None
    :type path_to_tokenizer: string, optional
    :return tokenizer: fitted tokenizer
    """

    with open(ROOT_DIR + path_to_text_list, 'rb') as file:
        train_text = pickle.load(file)
    
    # vocab_size+2 because: index starts from 1 and addition of oov_token takes index 1
    # thus, effectively vocabulary starts from index 2
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size+2, oov_token='UNK', filters='!"#$%&()*+-/:;<=>?@[\\]^`{|}~\t\n', lower=True) # '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    tokenizer.fit_on_texts(train_text)
    tokenizer.word_index = {key:value for key, value in tokenizer.word_index.items() if value < vocab_size+2}
    # addition required for point-generation approach only
    tokenizer.word_index['samplestart'] = len(tokenizer.word_index) + 1
    tokenizer.word_index['sampleend'] = len(tokenizer.word_index) + 1
    tokenizer.index_word = {value:key for key, value in tokenizer.word_index.items()}

    if save:
        with open(ROOT_DIR + path_to_tokenizer, 'wb') as file:
            pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        return tokenizer


def tokenize_and_pad_text(path_to_tokenizer, path_to_text_list, path_to_tokenized_text, padding_length):

    """Wrapper for tokenizer that accepts as inputs path to tokenizer and path to text and save tokenized text to specified path
    
    :param path_to_tokenizer: path to tokenizer fitted on train text corpus
    :type path_to_tokenizer: string
    :param path_to_text_list: path to text that needs to be tokenized
    :type path_to_text_list: string
    :param path_to_tokenized_text: path which should be used for saving tokenized text
    :type path_to_tokenized_text: string
    """

    # load tokenizer
    with open(ROOT_DIR + path_to_tokenizer, 'rb') as file:
        tokenizer = pickle.load(file)

    # load preprocessed text
    with open(ROOT_DIR + path_to_text_list, 'rb') as file:
        text_list = pickle.load(file)

    # tokenize text
    tokenized_text = tokenizer.texts_to_sequences(text_list)
    tokenized_text = tf.keras.preprocessing.sequence.pad_sequences(tokenized_text, maxlen=padding_length, padding='pre', truncating='pre')

    # save tokenized text to disk
    with open(ROOT_DIR + path_to_tokenized_text, 'wb') as file:
        pickle.dump(tokenized_text, file)