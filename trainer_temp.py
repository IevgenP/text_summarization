import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from src.nn.encoder_decoder import MainModel
from def_text_summ import ROOT_DIR, VOCAB_SIZE, TEXT_LEN, SUMMARY_LEN, BATCH_SIZE, EPOCHS, EMBEDDING_DIM, ATT_U, DROPOUT
from src.nn.encoder_decoder import BahdanauAttention, EncoderOnly, InferenceDecoder
from src.inferencer.inferencer import decode_sequence, sequence_to_summary, sequence_to_text

earlystopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.05,
    patience=2,
    mode='min',
    restore_best_weights=True
)

if __name__ == '__main__':

    # load prepared training
    with open(ROOT_DIR + '/prep_data/tokenized/train_data_tokenized', 'rb') as file:
        x_train = pickle.load(file)

    with open(ROOT_DIR + '/prep_data/tokenized/train_labels_tokenized', 'rb') as file:
        y_train = pickle.load(file)

    # load prepared test
    with open(ROOT_DIR + '/prep_data/tokenized/val_data_tokenized', 'rb') as file:
        x_val = pickle.load(file)

    with open(ROOT_DIR + '/prep_data/tokenized/val_labels_tokenized', 'rb') as file:
        y_val = pickle.load(file)

    # load prepared tokenizers to retreive input size for embedding layers
    with open(ROOT_DIR + '/pickled/trained_text_tokenizer', 'rb') as file:
        text_tokenizer = pickle.load(file)
    enc_vocab_size = len(text_tokenizer.word_index) + 1

    with open(ROOT_DIR + '/pickled/trained_summaries_tokenizer', 'rb') as file:
        summ_tokenizer = pickle.load(file)
    dec_vocab_size = len(summ_tokenizer.word_index) + 1

    print("Shape of text and summaries:", x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    print("Encoder input vocabulary size:", enc_vocab_size)
    print("Decoder input vocabulary size:", dec_vocab_size)

    # initialize the model
    model = MainModel(
        text_max_len=TEXT_LEN,
        summary_max_len=SUMMARY_LEN,
        enc_vocab_size=enc_vocab_size,
        dec_vocab_size=dec_vocab_size,
        embedded_dimension=EMBEDDING_DIM,
        attention_units=ATT_U,
        dropout=DROPOUT
    )

    # COMPILE THE MODEL
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')

    y_train_inp = y_train[:, :-1]
    y_train_out = y_train[:, 1:]
    y_train_out = y_train_out.reshape(y_train_out.shape[0], y_train_out.shape[1], 1)

    y_val_inp = y_val[:, :-1]
    y_val_out = y_val[:, 1:]
    y_val_out = y_val_out.reshape(y_val_out.shape[0], y_val_out.shape[1], 1)

    print("Modified shape of summaries", y_train_inp.shape, y_train_out.shape, y_val_inp.shape, y_val_out.shape)
        
    model.summary()

    history = model.fit(
        x=[x_train[:200, :], y_train_inp[:200, :]],
        y=y_train_out[:200, :, :],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([x_val[:100, :], y_val_inp[:100, :]],
                        y_val_out[:100, :, :]),
        callbacks=[earlystopper]
    )