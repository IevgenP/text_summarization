import re
import pickle
import tensorflow as tf
from def_text_summ import ROOT_DIR, VOCAB_SIZE, TEXT_LEN, SUMMARY_LEN
from src.preprocessing.text_utils import fit_tokenizer, tokenize_and_pad_text

# load data, preprocess and save results
def clean_text(load_path, save_path, start_end_marks=False):
    with open(ROOT_DIR + load_path, 'r') as file:
        text_list = file.readlines()
    
    prep_text_list = []
    for sample in text_list:
        sample = re.sub(r"`©", "'", sample)
        sample = re.sub(r'[\n\t]|(</t>)|(<t>)', " ", sample)
        sample = re.sub(r'"', '', sample)
        sample = re.sub(r"'", '', sample)
        if start_end_marks:
            sample = ' samplestart ' + sample + ' sampleend '
        sample = re.sub(r' +', ' ', sample)
        prep_text_list.append(sample)
    
    with open(ROOT_DIR + save_path, 'wb') as file:
        pickle.dump(prep_text_list, file)


if __name__ == '__main__':

    # TEXT
    # load and preprocess texts
    path_dict = {
        '/raw_data/train.txt.src': '/prep_data/train_data_prep',
        '/raw_data/val.txt.src': '/prep_data/val_data_prep',
        '/raw_data/test.txt.src': '/prep_data/test_data_prep',
    }
    for key, value in path_dict.items():
        clean_text(key, value)

    # fit tokenizer on train text
    fit_tokenizer(path_to_text_list='/prep_data/train_data_prep',
                  vocab_size=VOCAB_SIZE, 
                  save=True, 
                  path_to_tokenizer='/pickled/trained_text_tokenizer')


    # transform texts into matrix of tokens
    text_savepath_len = {
        '/prep_data/train_data_prep': ['/prep_data/tokenized/train_data_tokenized', TEXT_LEN],
        '/prep_data/val_data_prep': ['/prep_data/tokenized/val_data_tokenized', TEXT_LEN],
        '/prep_data/test_data_prep': ['/prep_data/tokenized/test_data_tokenized', TEXT_LEN],
    }

    for key, value in text_savepath_len.items():
        tokenize_and_pad_text(path_to_tokenizer='/pickled/trained_text_tokenizer',
                              path_to_text_list=key, 
                              path_to_tokenized_text=value[0], 
                              padding_length=value[1])


    # SUMMARIES
    path_dict_summaries = {
        '/raw_data/train.txt.tgt.tagged': '/prep_data/train_labels_prep',
        '/raw_data/val.txt.tgt.tagged': '/prep_data/val_labels_prep',
        '/raw_data/test.txt.tgt.tagged': '/prep_data/test_labels_prep',
    }
    for key, value in path_dict_summaries.items():
        clean_text(key, value, start_end_marks=True)


    # # fit tokenizer on summaries 
    # # this step is required at least because there are samplestart and sampleend tokens in summaries
    # # skipped for pointer-generator
    # fit_tokenizer(path_to_text_list='/prep_data/train_labels_prep',
    #               vocab_size=VOCAB_SIZE, 
    #               save=True, 
    #               path_to_tokenizer='/pickled/trained_summaries_tokenizer')


    # transform texts into matrix of tokens
    summary_savepath_len = {
        '/prep_data/train_labels_prep': ['/prep_data/tokenized/train_labels_tokenized', SUMMARY_LEN],
        '/prep_data/val_labels_prep': ['/prep_data/tokenized/val_labels_tokenized', SUMMARY_LEN],
        '/prep_data/test_labels_prep': ['/prep_data/tokenized/test_labels_prep', SUMMARY_LEN]
    }

    for key, value in summary_savepath_len.items():
        tokenize_and_pad_text(path_to_tokenizer='/pickled/trained_text_tokenizer',
                              path_to_text_list=key, 
                              path_to_tokenized_text=value[0], 
                              padding_length=value[1])