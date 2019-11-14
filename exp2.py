import tensorflow as tf 

"""num_words = 3
tk = tf.keras.preprocessing.text.Tokenizer(oov_token='UNK', num_words=num_words+1)
texts = ["my name is far faraway asdasd", "my name is","your name is"]
tk.fit_on_texts(texts)
print(tk.word_index)
print(tk.texts_to_sequences(texts))
## **Key Step**
tk.word_index = {e:i-1 for e,i in tk.word_index.items() if i-1 <= num_words and e != 'UNK'} # <= because Tokenizer reserve index 1 for "UNK"
tk.word_index[tk.oov_token] = len(tk.word_index) + 1 # in we use vocabulary size (num_words) bigger than actual word_index length
print(tk.word_index)
print(tk.texts_to_sequences(texts))"""


# num_words = 3
# tk = tf.keras.preprocessing.text.Tokenizer(oov_token='UNK', num_words=num_words+2)
# texts = ["my name is far faraway asdasd", "my name is","your name is"]
# tk.fit_on_texts(texts)
# tk.word_index = {e:i for e,i in tk.word_index.items() if i < num_words+2}
# tk.index_word = {value:key for key, value in tk.word_index.items()}
# print(len(tk.word_index))
# print(tk.word_index)
# print(tk.texts_to_sequences(texts))

import numpy as np 

ts = np.zeros((1,1))
ts[0, 0] = 123
tr = np.zeros((1,1))
tr = np.asarray([662])
print(tr.shape)
ts = np.append(ts, tr, axis=1)
print(ts)
print(ts.shape)