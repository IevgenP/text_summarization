import tensorflow as tf

class BahdanauAttention(tf.keras.layers.Layer):
    
    def __init__(self, units, **kwargs):
        self.supports_masking = True
        self.units = units
        super(BahdanauAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.W1 = tf.keras.layers.Dense(self.units, name='W1')
        self.W2 = tf.keras.layers.Dense(self.units, name='W2')
        self.V = tf.keras.layers.Dense(1, name='V')
        super(BahdanauAttention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs):
        assert isinstance(inputs, list)
        decoder_output = inputs[0]
        encoder_output = inputs[1]

        def calculate_context_vec(step_inputs, states):

            # hidden shape == (batch_size, hidden size)
            # hidden_with_time_axis shape == (batch_size, 1, hidden size)
            # we are doing this to perform addition to calculate the score
            hidden_with_time_axis = tf.expand_dims(step_inputs, 1)

            # score shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying score to self.V
            # the shape of the tensor before applying self.V is (batch_size, max_length, units)
            w_enc_output = self.W1(encoder_output)
            #print("Shape of w_enc_output", w_enc_output.shape)

            w_dec_hidden = self.W2(hidden_with_time_axis)
            #print("Shape of w_dec_hidden", w_dec_hidden.shape)

            score = self.V(
                tf.nn.tanh(
                    w_enc_output + w_dec_hidden
                )
            )

            # attention_weights shape == (batch_size, max_length, 1)
            step_attention_weights = tf.nn.softmax(score, axis=1)

            # step context_vector shape after sum == (batch_size, hidden_size)
            step_context_vector = step_attention_weights * encoder_output
            step_context_vector = tf.reduce_sum(step_context_vector, axis=1)
            return step_context_vector, [step_context_vector]


        def create_inital_state(inputs, hidden_size):
            # fake initial state to be passed to tf.keras.backend.rnn function
            # https://github.com/thushv89/attention_keras/blob/master/layers/attention.py
            fake_state = tf.keras.backend.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim)
            fake_state = tf.keras.backend.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = tf.keras.backend.expand_dims(fake_state)  # <= (batch_size, 1)
            fake_state = tf.keras.backend.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim)
            return fake_state

        fake_state = create_inital_state(encoder_output, encoder_output.shape[-1])
        _, context_vector, _ = tf.keras.backend.rnn(calculate_context_vec, decoder_output, [fake_state])

        return context_vector

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return [
            tf.TensorShape(input_shape[0][0], input_shape[0][1], input_shape[0][2]), 
            tf.TensorShape(input_shape[1][0], input_shape[1][1], input_shape[1][2]), 
        ]

    def get_config(self):
        """Defines layer configuration
        
        :return: layer configurations
        :rtype: dictionary
        """
        config = {
            "units": self.units
        }
        base_config = super(BahdanauAttention, self).get_config()
        config.update(base_config)
        return config


def MainModel(text_max_len=1500,
               summary_max_len=100,
               enc_vocab_size=20000,
               dec_vocab_size=20000,
               embedded_dimension=128,
               lstm_hidden_units=128,
               attention_units=100):

    # ENCODER -----------------------------------------------------------------------------------------------------------#
    enc_input = tf.keras.layers.Input(
        shape=(text_max_len, ), 
        dtype='int32', 
        name='encoder_input'
    )
    
    enc_embedded_sequence = tf.keras.layers.Embedding(
        input_dim=enc_vocab_size,
        output_dim=embedded_dimension,
        input_length=text_max_len,
        name='encoder_embeddings',
        mask_zero=True
    )(enc_input)
    
    enc_lstm, enc_state_h_forward, enc_state_c_forward, enc_state_h_backward, enc_state_c_backward = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=lstm_hidden_units, 
            return_sequences=True,
            return_state=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            name='encoder_lstm'
        ),
        merge_mode='sum',
        name='encoder_bidirectional_lstm'
    )(enc_embedded_sequence)

    enc_state_h = tf.keras.layers.Add(name='encoder_hidden_state_addition')([enc_state_h_forward, enc_state_h_backward])
    enc_state_c = tf.keras.layers.Add(name='encoder_cell_state_addition')([enc_state_c_forward, enc_state_c_backward])

    # DECODER -----------------------------------------------------------------------------------------------------------#
    dec_input = tf.keras.layers.Input(
        shape=(None,), 
        name='decoder_input'
    )
    dec_embedded_sequence = tf.keras.layers.Embedding(
        input_dim=dec_vocab_size,
        output_dim=embedded_dimension,
        input_length=summary_max_len,
        name='decoder_embeddings',
        mask_zero=True
    )(dec_input)

    dec_lstm, dec_state_h, dec_state_c = tf.keras.layers.LSTM(
        units=lstm_hidden_units, 
        return_sequences=True,
        return_state=True,
        dropout=0.2,
        recurrent_dropout=0.2,
        name='decoder_lstm'
    )(dec_embedded_sequence, initial_state=[enc_state_h, enc_state_c])

    # Attention layer in Decoder
    context_vec = BahdanauAttention(
        units=attention_units, 
        name='bahdanau_attention'
    )([dec_lstm, enc_lstm])

    # Concatenation of context vector with decoder output
    dec_concat = tf.keras.layers.Concatenate(name='concat_hid_layer_context_vec')([dec_lstm, context_vec])

    # Dense layer of Decoder
    dec_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            dec_vocab_size, 
            activation='softmax',
            name='decoder_dense'
        ),
        name='time_distributed_dense'
    )(dec_concat) #(dec_lstm)

    # DEFINING MAIN MODEL ------------------------------------------------------------------------------------------------#
    model = tf.keras.models.Model([enc_input, dec_input], dec_output)

    return model
    

def EncoderOnly(trained_enc_dec_model):
    return tf.keras.models.Model(
        inputs=trained_enc_dec_model.get_layer('encoder_input').input,
        outputs=[
            trained_enc_dec_model.get_layer('encoder_bidirectional_lstm').output[0],
            trained_enc_dec_model.get_layer('encoder_hidden_state_addition').output, 
            trained_enc_dec_model.get_layer('encoder_cell_state_addition').output
        ]
    )


def InferenceDecoder(trained_enc_dec_model,
                     text_max_len=1500,
                     summary_max_len=100,
                     dec_vocab_size=20000,
                     embedded_dimension=128,
                     lstm_hidden_units=128,
                     attention_units=100):

    # Decoder setup
    inf_dec_input = trained_enc_dec_model.get_layer('decoder_input').input

    # Below tensors will hold the states of the previous time step
    inf_decoder_hidden_states_input = tf.keras.layers.Input(
        shape=(text_max_len, lstm_hidden_units),
        name='inf_input_1'
    )
    inf_decoder_state_input_h = tf.keras.layers.Input(
        shape=(lstm_hidden_units,), 
        name='inf_input_2'
    )
    inf_decoder_state_input_c = tf.keras.layers.Input(
        shape=(lstm_hidden_units,), 
        name='inf_input_3'
    )

    
    # load embedding layer
    inf_dec_emb = trained_enc_dec_model.get_layer('decoder_embeddings').output

    # load weights of decoder lstm layer from trained model
    inf_lstm_layer = trained_enc_dec_model.get_layer('decoder_lstm')    
    inf_dec_lstm, inf_state_h, inf_state_c = inf_lstm_layer(
        inf_dec_emb, 
        initial_state=[inf_decoder_state_input_h, inf_decoder_state_input_c],
        training=False
    )

    # load weights and configuration of decoder attention layer from trained model
    inf_attention_layer = trained_enc_dec_model.get_layer('bahdanau_attention')
    inf_context_vec = inf_attention_layer([inf_dec_lstm, inf_decoder_hidden_states_input], training=False)

    # Concatenation of context vector with decoder output
    inf_dec_concat = tf.keras.layers.Concatenate()([inf_dec_lstm, inf_context_vec])

    # A dense softmax layer to generate prob dist. over the target vocabulary
    inf_dense_layer = trained_enc_dec_model.get_layer('time_distributed_dense') # time_distributed_dense
    inf_dec_output = inf_dense_layer(inf_dec_concat)

    # Final decoder model
    inference_decoder = tf.keras.models.Model(
        [inf_dec_input] + [inf_decoder_hidden_states_input, inf_decoder_state_input_h, inf_decoder_state_input_c],
        [inf_dec_output] + [inf_state_h, inf_state_c])

    return inference_decoder

