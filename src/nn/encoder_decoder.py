import tensorflow as tf
import numpy as np

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
        print("encoder_output shape: ", encoder_output.shape)

        def get_attention_weights(inputs, states):

            #print('#'*15)
            # hidden shape == (batch_size, hidden size)
            # hidden_with_time_axis shape == (batch_size, 1, hidden size)
            # we are doing this to perform addition to calculate the score
            #print("Shape of inputs into get_att... function: ", inputs.shape)
            hidden_with_time_axis = tf.expand_dims(inputs, 1)

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

            #print("Shape of score: ", score.shape)
            score = tf.keras.backend.reshape(score, (-1, encoder_output.shape[1])) # batch_size, en_seq_len
            #print("Shape of score: ", score.shape)

            # EXCLUDE attention_weights shape == (batch_size, max_length, 1)
            step_attention_weights = tf.nn.softmax(score, axis=1)
            #print("Shape of step_attention_weights: ", step_attention_weights.shape)

            return step_attention_weights, [step_attention_weights]

        
        def calculate_context_vec(inputs, states):

            # step context_vector shape after 
            # sum == (batch_size, hidden_size)
            #print("Shape of encoder output: ", encoder_output.shape)
            attentiont_weights_time_axis = tf.expand_dims(inputs, -1)
            #print("Shape of attentiont_weights_time_axis: ", attentiont_weights_time_axis.shape)
            step_context_vector = attentiont_weights_time_axis * encoder_output
            #print("Shape of multiplication step_att_weights * enc_output: ", step_context_vector.shape)
            step_context_vector = tf.reduce_sum(step_context_vector, axis=1)
            #print("Shape of multiplication step_att_weights * enc_output after sum reduce: ", step_context_vector.shape)
            return step_context_vector, [step_context_vector]


        def create_inital_state(inputs, hidden_size):
            # fake initial state to be passed to tf.keras.backend.rnn function
            # https://github.com/thushv89/attention_keras/blob/master/layers/attention.py
            fake_state = tf.keras.backend.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim)
            fake_state = tf.keras.backend.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = tf.keras.backend.expand_dims(fake_state)  # <= (batch_size, 1)
            fake_state = tf.keras.backend.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim)
            return fake_state

        # make fake states for rnn funiction
        fake_state_att = create_inital_state(encoder_output, encoder_output.shape[1])
        #print("Shape of fake_state_att: ", fake_state_att.shape)
        fake_state_context = create_inital_state(encoder_output, encoder_output.shape[-1])
        #print("Shape of fake_state_context: ", fake_state_context.shape)
        
        _, attention_weights, _ = tf.keras.backend.rnn(get_attention_weights, decoder_output, [fake_state_att])
        #print("PASSED FIRST STAGE")
        _, context_vector, _ = tf.keras.backend.rnn(calculate_context_vec, attention_weights, [fake_state_context])

        return context_vector, attention_weights

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
               attention_units=100,
               dropout=None):

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

    enc_lstm_0, enc_state_h_forward_0, enc_state_c_forward_0, enc_state_h_backward_0, enc_state_c_backward_0 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=lstm_hidden_units, 
            return_sequences=True,
            return_state=True,
            dropout=dropout,
            recurrent_dropout=dropout,
            name='encoder_lstm_0'
        ),
        merge_mode='ave',
        name='encoder_bidirectional_lstm_0'
    )(enc_embedded_sequence)
    
    enc_lstm, enc_state_h_forward_1, enc_state_c_forward_1, enc_state_h_backward_1, enc_state_c_backward_1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=lstm_hidden_units, 
            return_sequences=True,
            return_state=True,
            dropout=dropout,
            recurrent_dropout=dropout,
            name='encoder_lstm'
        ),
        merge_mode='ave',
        name='encoder_bidirectional_lstm'
    )(enc_lstm_0)

    enc_state_h = tf.keras.layers.Average(name='encoder_hidden_state_avg')(
        [enc_state_h_forward_0, enc_state_h_backward_0, enc_state_h_forward_1, enc_state_h_backward_1]
    )
    enc_state_c = tf.keras.layers.Average(name='encoder_cell_state_avg')(
        [enc_state_c_forward_0, enc_state_c_backward_0, enc_state_c_forward_1, enc_state_c_backward_1]
    )

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
        dropout=dropout,
        recurrent_dropout=dropout,
        name='decoder_lstm'
    )(dec_embedded_sequence, initial_state=[enc_state_h, enc_state_c])

    # Attention layer in Decoder
    context_vec, att_vector = BahdanauAttention(
        units=attention_units, 
        name='bahdanau_attention'
    )([dec_lstm, enc_lstm])

    # Concatenation of context vector with decoder output
    print("Shape of dec_lstm: ", dec_lstm.shape)
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

    # ### POINTER-GENERATOR [TO BE PRESENTED AS A LAYER]
    print("Shape of context_vec", context_vec.shape)
    print("Shape of att_vector", att_vector.shape)
    print("Shape of dec_state_h", dec_state_h)
    print("Shape of dec_embedded_sequence", dec_embedded_sequence)
    print("Shape of dec_output", dec_output)

    # Point-generator
    dense_context_vec = tf.keras.layers.Dense(units=)


    # p_gen = (
    #     W_h.T*context_vec +                 # Shape of context_vec (None, None, 128)
    #     W_s.T*dec_state_h +                 # Shape of dec_state_h Tensor("decoder_lstm/Identity_1:0", shape=(None, 128), dtype=float32)
    #     W_x.T*dec_embedded_sequence +       # Shape of dec_embedded_sequence Tensor("decoder_embeddings/Identity:0", shape=(None, None, 200), dtype=float32)
    #     b
    # )

    #  p_w = p_gen*p_vocab_w + (1-p_gen)*SUM(att_vec)  # Shape of dec_output Tensor("time_distributed_dense/Identity:0", shape=(None, None, 120002), dtype=float32)

    # DEFINING MAIN MODEL ------------------------------------------------------------------------------------------------#
    model = tf.keras.models.Model([enc_input, dec_input], dec_output)

    return model
    

def EncoderOnly(trained_enc_dec_model):
    return tf.keras.models.Model(
        inputs=trained_enc_dec_model.get_layer('encoder_input').input,
        outputs=[
            trained_enc_dec_model.get_layer('encoder_bidirectional_lstm').output[0],
            trained_enc_dec_model.get_layer('encoder_hidden_state_avg').output, 
            trained_enc_dec_model.get_layer('encoder_cell_state_avg').output
        ]
    )


def InferenceDecoder(trained_enc_dec_model,
                     text_max_len=1500,
                     lstm_hidden_units=128):

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

