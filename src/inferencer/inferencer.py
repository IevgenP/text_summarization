import numpy as np
from def_text_summ import ROOT_DIR, VOCAB_SIZE


def decode_sequence(input_sequence, encoder, decoder_model, word_index, reverse_word_index, summary_max_len):

    # Use encoder to convert input into state vectors
    enc_output, enc_hidden_state, enc_cell_state = encoder.predict(input_sequence)
    
    # start with hidden states produced by encoder
    input_hidden_state = enc_hidden_state
    input_cell_state = enc_cell_state

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Chose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = word_index['samplestart']

    stop_condition = False
    decoded_sentence = ''
    
    # loop through decoder untill max length is reached or stop word is predicted
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [enc_output, input_hidden_state, input_cell_state])
        
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_word_index[sampled_token_index]

        decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find stop word
        if (sampled_token == 'sampleend' or len(decoded_sentence.split()) >= (summary_max_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        input_hidden_state, input_cell_state = h, c

    return decoded_sentence


def sequence_to_summary(input_sequence, reverse_word_index):
    summary_text = ''
    for token in input_sequence:
        if (
            token != 0 and
            reverse_word_index[token] != 'samplestart' and
            reverse_word_index[token] != 'sampleend'
        ):
            summary_text = summary_text + reverse_word_index[token] + ' '
    
    return summary_text


def sequence_to_text(input_sequence, reverse_word_index):
    text = ''
    for token in input_sequence:
        if token != 0:
            text = text + reverse_word_index[token] + ' '
    return text