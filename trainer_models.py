import os
import time
import tensorflow as tf
from def_text_summ import ROOT_DIR, VOCAB_SIZE, TEXT_LEN, SUMMARY_LEN, BATCH_SIZE, EMBEDDING_DIM, LSTM_HIDDEN_UNITS, DROPOUT, EPOCHS
from src.nn.encoder_decoder_models import Encoder, Decoder

# make optimizer and loss
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none'
)

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

# training steps function
@tf.function
def train_step(inp, targ, encoder, enc_init_h, enc_init_c, decoder, target_word_index):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_state_h, enc_state_c = encoder(inp, enc_init_h, enc_init_c)

    dec_h, dec_c = enc_state_h, enc_state_c

    dec_input = tf.expand_dims([target_word_index['samplestart']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_h, dec_c = decoder(dec_input, enc_output, dec_h, dec_c)

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss
  

if __name__ == '__main__':

    # initialize encoder and decoder
    encoder = Encoder(
        batch_size=BATCH_SIZE, 
        text_max_len=TEXT_LEN, 
        enc_vocab_size=VOCAB_SIZE, 
        embedded_dimension=EMBEDDING_DIM, 
        lstm_hidden_units=LSTM_HIDDEN_UNITS, 
        dropout=DROPOUT
    )

    decoder = Decoder(
        batch_size=BATCH_SIZE, 
        summary_max_len=SUMMARY_LEN, 
        dec_vocab_size=VOCAB_SIZE, 
        embedded_dimension=EMBEDDING_DIM, 
        lstm_hidden_units=LSTM_HIDDEN_UNITS, 
        dropout=DROPOUT
    )


    # checkpoint save
    checkpoint_dir = ROOT_DIR + '/pickled/checkpoints/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=Encoder,
                                     decoder=Decoder)

    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                            batch,
                                                            batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
