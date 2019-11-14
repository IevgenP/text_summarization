import pickle
from def_text_summ import ROOT_DIR
import numpy as np
import sys
from src.nn.encoder_decoder import BahdanauAttention, EncoderOnly, InferenceDecoder
from src.inferencer.inferencer import decode_sequence, sequence_to_summary, sequence_to_text
np.set_printoptions(threshold=sys.maxsize)

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


import tensorflow as tf
# from src.nn.encoder_decoder import BahdanauAttention

# check what objects are created by encoder-decoder
model = tf.keras.models.load_model(
    ROOT_DIR + '/pickled/simple_enc_dec_att.h5',
    custom_objects={'BahdanauAttention': BahdanauAttention}
)
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')

print("MODEL SUMMARY")
model.summary()

y_train_inp = y_train[:, :]
y_train_out = y_train[:, :]
y_train_out = y_train_out.reshape(y_train_out.shape[0], y_train_out.shape[1], 1)

y_val_inp = y_val[:, :]
y_val_out = y_val[:, :]
y_val_out = y_val_out.reshape(y_val_out.shape[0], y_val_out.shape[1], 1)


for i in range(15,25): # len(x_val)
    print("Review:", sequence_to_text(input_sequence=x_val[i], reverse_word_index=rev_text_word_index))
    print("\n")
    print("Original summary:", sequence_to_summary(input_sequence=y_val[i], reverse_word_index=rev_summ_word_index))
    print("\n")
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
    prediction = model.predict([x_val[i:i+1, :], y_val_inp[i:i+1, :]])
    sentence = ''
    for n in range(prediction.shape[1]):
        sampled_token_index = np.argmax(prediction[0, n, :])
        sampled_token = rev_summ_word_index[sampled_token_index]
        sentence += ' ' + sampled_token
    print ("Created summary:", sentence)
    print("\n")




# print("Review:", sequence_to_text(input_sequence=x_train[i], reverse_word_index=rev_text_word_index))
# print("\n")
# print("Original summary:", sequence_to_summary(input_sequence=y_train[i], reverse_word_index=rev_summ_word_index))
# print("\n")
# print(sentence)


# for num, layer in enumerate(model.layers):
#     print(num, layer)
    

# #print(model.get_config())
# print(len(model.inputs))
# print(len(model.layers[7].output))


# encoder_model = tf.keras.models.Model(
#     inputs=model.layers[0].input,
#     outputs=[
#         model.layers[3].output[0], 
#         model.layers[5].output, 
#         model.layers[6].output
#     ]
# )

# print("ENCODER MODEL SUMMARY 1")
# encoder_model.summary()


# encoder_model = tf.keras.models.Model(
#     inputs=model.get_layer('encoder_input').input,
#     outputs=[
#         model.get_layer('encoder_bidirectional_lstm').output[0], 
#         model.get_layer('encoder_hidden_state_addition').output, 
#         model.get_layer('encoder_cell_state_addition').output
#     ]
# )

# print("ENCODER MODEL SUMMARY 2")
# encoder_model.summary()


# import tensorflow as tf
# from src.nn.encoder_decoder import InferenceDecoder

# inf_dec = InferenceDecoder(
#     text_max_len=1500,
#     summary_max_len=100,
#     vocabulary_size=30000, 
#     embedded_dimension=128,
#     lstm_hidden_units=128,
#     attention_units=100
# )

# inf_dec.summary()