import re
import pickle
from def_text_summ import ROOT_DIR

# load data, preprocess and save results
def clean_text(load_path, save_path):
    with open(ROOT_DIR + load_path, 'r') as file:
        text_list = file.readlines()
    
    prep_text_list = []
    for sample in text_list:
        sample = re.sub(r"`", "'", sample)
        sample = re.sub(r'[\n\t]|(</t>)|(<t>)', " ", sample)
        sample = sample.strip()
        prep_text_list.append(sample)
    
    with open(ROOT_DIR + save_path, 'wb') as file:
        pickle.dump(prep_text_list, file)


if __name__ == '__main__':

    path_dict = {
        '/raw_data/train.txt.src': '/prep_data/train_data_prep',
        '/raw_data/train.txt.tgt.tagged': '/prep_data/train_labels_prep',
        '/raw_data/val.txt.src': '/prep_data/val_data_prep',
        '/raw_data/val.txt.tgt.tagged': '/prep_data/val_labels_prep',
        '/raw_data/test.txt.src': '/prep_data/test_data_prep',
        '/raw_data/test.txt.tgt.tagged': '/prep_data/test_labels_prep',
    }

    for key, value in path_dict.items():
        clean_text(key, value)
