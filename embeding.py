#!/usr/bin/env python
# coding: utf-8
import io
import os
import re
import shutil
import string
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization

# Download da base

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                  untar=True, cache_dir='.',
                                  cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
os.listdir(dataset_dir)
train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)


# # Tratamentos do texto

# ## Remoção de caractéres não nativos do inglês, pontuações e caractéres maiusculos
from nltk.tokenize import word_tokenize
def clean_non_english(txt):
    txt = re.sub(r'\W+', ' ', txt)
    txt = txt.lower()
    txt = txt.replace("[^a-zA-Z]", " ")
    word_tokens = word_tokenize(txt)
    filtered_word = [w for w in word_tokens if all(ord(c) < 128 for c in w)]
    filtered_word = [w + " " for w in filtered_word]
    return "".join(filtered_word)


# ## Remoção de Stop Words

## Downloads necessários - Executar somente uma vez
## nltk.download('punkt')
## nltk.download('stopwords')
## nltk.download('wordnet')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
def clean_text(english_txt):
    try:
       word_tokens = word_tokenize(english_txt)
       filtered_word = [w for w in word_tokens if not w in stop_words]
       filtered_word = [w + " " for w in filtered_word]
       return "".join(filtered_word)
    except:
       return np.nan


lm= WordNetLemmatizer()

path = ["neg/", "pos/"]
for i in path:
    for filename in os.listdir("aclImdb/train/" + i):
        with open("aclImdb/train/" + i + filename, "r", encoding="utf8") as f:
            lines = f.readlines()
            WordSet = ''.join(lines)
            
            #Remoção de pontuação, acentos e caracteres não existentes na lingua inglesa
            WordSet = clean_non_english(WordSet)
            
            #Remoção de stop words
            WordSet = clean_text(WordSet)
            
            WordSet = nltk.word_tokenize(WordSet)
            
            #Lemetização
            WordSetLem = []
            for word in WordSet:
                WordSetLem.append(lm.lemmatize(word))
#             print(WordSetLem)
            f.close()
        with open("aclImdb/train/" + i + filename, "w+") as f:
            for word in WordSetLem:
                f.write(word)
                f.write(' ')
            f.close()


# # Carrega bases de treino e validação

batch_size = 1024
seed = 123

train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)
val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed
)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Embed a 1,000 word vocabulary into 5 dimensions.
embedding_layer = tf.keras.layers.Embedding(1000, 5)

result = embedding_layer(tf.constant([1, 2, 3]))
result.numpy()

result = embedding_layer(tf.constant([[0, 1, 2], [3, 4, 5]]))
result.shape


# Create a custom standardization function to strip HTML break tags '<br />'.
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation), '')

# Vocabulary size and number of words in a sequence.
vocab_size = 10000
sequence_length = 100

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Note that the layer uses the custom standardization defined above.
# Set maximum_sequence length as all samples are not of the same length.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (no labels) and call adapt to build the vocabulary.
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)


embedding_dim=16

model = Sequential([
    vectorize_layer,
    Embedding(vocab_size, embedding_dim, name="embedding"),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1)
])



tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")



model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])



model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[tensorboard_callback])



model.summary()



#docs_infra: no_execute
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir logs')



weights = model.get_layer('embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()



out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()
