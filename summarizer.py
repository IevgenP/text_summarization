import pickle
import tensorflow as tf
import numpy as np
from def_text_summ import ROOT_DIR, VOCAB_SIZE, TEXT_LEN, SUMMARY_LEN
from src.nn.encoder_decoder import BahdanauAttention, EncoderOnly, InferenceDecoder
from src.inferencer.inferencer import decode_sequence, sequence_to_summary, sequence_to_text

np.set_printoptions(
    suppress=True,
    formatter={'float_kind':'{:16.10f}'.format}, 
    linewidth=130
)


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


# cheating time !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# use training texts to see how bad model is
# load prepared sets
with open(ROOT_DIR + '/prep_data/tokenized/train_data_tokenized', 'rb') as file:
    x_train = pickle.load(file)

with open(ROOT_DIR + '/prep_data/tokenized/train_labels_tokenized', 'rb') as file:
    y_train = pickle.load(file)
# -------------------------------------------------

# load pre-trained model
trained_model = tf.keras.models.load_model(
    ROOT_DIR + '/pickled/simple_enc_dec_att.h5',
    custom_objects={'BahdanauAttention': BahdanauAttention}
)
#trained_model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
trained_model.summary()

# GO THROUGH FEW EPOCHS OF TRAINING BEFORE TEST RUN ####################################
y_train_inp = y_train[:, :]
y_train_out = y_train[:, :]
y_train_out = y_train_out.reshape(y_train_out.shape[0], y_train_out.shape[1], 1)

y_val_inp = y_val[:, :]
y_val_out = y_val[:, :]
y_val_out = y_val_out.reshape(y_val_out.shape[0], y_val_out.shape[1], 1)


for i in range(15,17): # len(x_val)
    print("Review:", sequence_to_text(input_sequence=x_val[i], reverse_word_index=rev_text_word_index))
    print("\n")
    print("Original summary:", sequence_to_summary(input_sequence=y_val[i], reverse_word_index=rev_summ_word_index))
    print("\n")
    prediction = trained_model.predict([x_val[i:i+1, :], y_val_inp[i:i+1, :]])
    sentence = ''
    for n in range(prediction.shape[1]):
        sampled_token_index = np.argmax(prediction[0, n, :])
        sampled_token = rev_summ_word_index[sampled_token_index]
        sentence += ' ' + sampled_token
    print ("Created summary:", sentence)
    print("\n")


#########################################################################################

# # initialize instances of encoder and inference decoder
# enc = EncoderOnly(
#     trained_enc_dec_model=trained_model,
#     text_max_len=TEXT_LEN,
#     enc_vocab_size=enc_vocab_size,
#     embedded_dimension=128
# )


enc = EncoderOnly(trained_enc_dec_model=trained_model)
print("")
print('enc summary')
# enc.summary()


# from src.nn.encoder_decoder import EncoderFromModel
# enc_check = EncoderFromModel(trained_enc_dec_model=trained_model)
# print("")
# print('enc_check summary')
# enc_check.summary()
# print("")

# for i in range(15,17):
#     pred_enc = enc.predict(x_val[i:i+1, :])
#     pred_enc_check = enc_check.predict(x_val[i:i+1, :])
#     print("Do they give same results?", np.array_equal(pred_enc[0], pred_enc_check[0]))
#     print("Do they give same results?", np.array_equal(pred_enc[1], pred_enc_check[1]))
#     print("Do they give same results?", np.array_equal(pred_enc[2], pred_enc_check[2]))
#     print("")


# enc.summary()
# enc_output, enc_h_state, enc_c_state = enc.predict(x_val[120:121])
# print("Hidden states for whole input", enc_output)
# print("Hidden state after last word", enc_h_state)
# print("Hidden state for last cell", enc_c_state)

# enc_output, enc_h_state, enc_c_state = enc.predict(x_val[10:11])
# print("Hidden states for whole input", enc_output)
# print("Hidden state after last word", enc_h_state)
# print("Hidden state for last cell", enc_c_state)

inf_dec = InferenceDecoder(
    trained_enc_dec_model=trained_model,
    text_max_len=TEXT_LEN,
    summary_max_len=SUMMARY_LEN,
    dec_vocab_size=dec_vocab_size,
    embedded_dimension=128,
    attention_units=30
)
inf_dec.summary()

for i in range(15,17): # len(x_val)
  print("Review:", sequence_to_text(input_sequence=x_val[i], reverse_word_index=rev_text_word_index))
  print("\n")
  print("Original summary:", sequence_to_summary(input_sequence=y_val[i], reverse_word_index=rev_summ_word_index))
  print("\n")
#   trained_model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
#   enc = EncoderOnly(
#     trained_enc_dec_model=trained_model,
#     text_max_len=TEXT_LEN,
#     enc_vocab_size=enc_vocab_size,
#     embedded_dimension=128
# )
#   inf_dec = InferenceDecoder(
#     trained_enc_dec_model=trained_model,
#     text_max_len=TEXT_LEN,
#     summary_max_len=SUMMARY_LEN,
#     dec_vocab_size=dec_vocab_size,
#     embedded_dimension=128,
#     attention_units=30
# )
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