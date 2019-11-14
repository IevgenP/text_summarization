import tensorflow as tf 


class Encoder(tf.keras.Model):

    def __init__(
        self,
        batch_size, 
        text_max_len, 
        enc_vocab_size, 
        embedded_dimension, 
        lstm_hidden_units, 
        dropout
    ):

        super(Encoder, self).__init__()
        
        self.batch_size = batch_size
        self.text_max_len = text_max_len
        self.enc_vocab_size = enc_vocab_size
        self.embedded_dimension = embedded_dimension
        self.lstm_hidden_units = lstm_hidden_units
        self.dropout = dropout
        
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.enc_vocab_size,
            output_dim=self.embedded_dimension,
            input_length=self.text_max_len,
            mask_zero=True
        )
        self.lstm = tf.keras.layers.LSTM(
            units=self.lstm_hidden_units, 
            return_sequences=True,
            return_state=True,
            dropout=self.dropout,
            recurrent_dropout=self.dropout,
        )

    def call(self, x, initialized_hidden, initialized_cell): 
        x = self.embedding(x)
        enc_lstm, enc_state_h, enc_state_c = self.lstm(
            x,
            initial_state=[initialized_hidden, initialized_cell]
        )
        return enc_lstm, enc_state_h, enc_state_c

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.lstm_hidden_units))

    def initialize_cell_state(self):
        return tf.zeros((self.batch_size, self.lstm_hidden_units))


class Decoder(tf.keras.Model):

    def __init__(
        self,
        batch_size, 
        summary_max_len, 
        dec_vocab_size, 
        embedded_dimension, 
        lstm_hidden_units, 
        dropout, 
        attention_units=None
    ):

        super(Decoder, self).__init__()

        self.batch_size = batch_size
        self.summary_max_len = summary_max_len
        self.dec_vocab_size = dec_vocab_size
        self.embedded_dimension = embedded_dimension
        self.lstm_hidden_units = lstm_hidden_units
        self.dropout = dropout
        self.attention_units = attention_units

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.dec_vocab_size,
            output_dim=self.embedded_dimension,
            input_length=self.summary_max_len,
            mask_zero=True
        )
        
        self.lstm = tf.keras.layers.LSTM(
            units=self.lstm_hidden_units, 
            return_sequences=True,
            return_state=True,
            dropout=self.dropout,
            recurrent_dropout=self.dropout,
        )
        
        self.fully_connected = tf.keras.layers.Dense(
            self.dec_vocab_size, 
            activation='softmax',
            name='decoder_dense'
        )

    def call(self, x, enc_output, prepared_hidden, prepared_cell):
        x = self.embedding(x)
        dec_lstm, dec_state_h, dec_state_c = self.lstm(
            x,
            initial_state=[prepared_hidden, prepared_cell]
        )
        dec_output = self.fully_connected(dec_lstm)
        return dec_output, dec_state_h, dec_state_c
        