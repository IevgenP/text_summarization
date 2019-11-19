import pickle
import tensorflow as tf
import numpy as np
from def_text_summ import ROOT_DIR, VOCAB_SIZE, TEXT_LEN, SUMMARY_LEN, LSTM_HIDDEN_UNITS
from src.nn.encoder_decoder import BahdanauAttention, EncoderOnly, InferenceDecoder
from src.inferencer.inferencer import decode_sequence, sequence_to_summary, sequence_to_text

# load tokenizer for extracting word indices
with open(ROOT_DIR + '/pickled/trained_summaries_tokenizer', 'rb') as file:
    summ_tokenizer = pickle.load(file)
summ_word_index = summ_tokenizer.word_index
dec_vocab_size = len(summ_word_index) + 1
rev_summ_word_index = summ_tokenizer.index_word

with open(ROOT_DIR + '/pickled/trained_text_tokenizer', 'rb') as file:
    text_tokenizer = pickle.load(file)
text_word_index = text_tokenizer.word_index
enc_vocab_size = len(text_word_index) + 1
rev_text_word_index = text_tokenizer.index_word

# load prepared sets
with open(ROOT_DIR + '/prep_data/tokenized/val_data_tokenized', 'rb') as file:
    x_val = pickle.load(file)

with open(ROOT_DIR + '/prep_data/tokenized/val_labels_tokenized', 'rb') as file:
    y_val = pickle.load(file)

# load prepared sets
with open(ROOT_DIR + '/prep_data/tokenized/train_data_tokenized', 'rb') as file:
    x_train = pickle.load(file)

with open(ROOT_DIR + '/prep_data/tokenized/train_labels_tokenized', 'rb') as file:
    y_train = pickle.load(file)

# load pre-trained model
trained_model = tf.keras.models.load_model(
    ROOT_DIR + '/pickled/enc_dec_att_sum.h5',
    custom_objects={'BahdanauAttention': BahdanauAttention}
)
trained_model.summary()

# initialize encoder
enc = EncoderOnly(trained_enc_dec_model=trained_model)

# initialize decoder
inf_dec = InferenceDecoder(
    trained_enc_dec_model=trained_model,
    text_max_len=TEXT_LEN,
    lstm_hidden_units=LSTM_HIDDEN_UNITS
)
inf_dec.summary()

# loop through examples and make summaries
for i in range(45,50): # len(x_val)
  print("Review:", sequence_to_text(input_sequence=x_val[i], reverse_word_index=rev_text_word_index))
  print("\n")
  print("Original summary:", sequence_to_summary(input_sequence=y_val[i], reverse_word_index=rev_summ_word_index))
  print("\n")
  print(
      "Predicted summary:", 
      decode_sequence(
          input_sequence=x_val[i:i+1, :],
          encoder=enc,
          decoder_model=inf_dec,
          word_index=summ_word_index,
          reverse_word_index=rev_summ_word_index,
          summary_max_len=SUMMARY_LEN
        )
    )
  print("\n")