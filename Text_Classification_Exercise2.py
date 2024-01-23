#%%
#1. import packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import sklearn
import os, pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
import pickle
#%%
#2. Data Loading
URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
df= pd.read_csv(URL)
# %%
#3. Data Inspection
print("Shape of the Data", df.shape)
print("\nInfo abotu the data:\n",df.info())
print("\nDescription of the data:\n",df.describe().transpose())
print("\nExample data:\n", df.head(1))
# %%
#4. Data Cleaning
print(df.isna().sum())
print("---------------------------------------")
print(df.duplicated().sum()) 
# %%
#Remove the Duplicates
df.drop_duplicates(inplace=True)
print(df.duplicated().sum())
# %%
#5. Split Data into features and labels
features = df['review'].values
labels = df['sentiment'].values
# %%
#6. Convert categorical label into integer
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
# %%
#7. Perform train test split
from sklearn.model_selection import train_test_split
seed = 42
X_train, X_test, y_train, y_test= train_test_split(features,labels_encoded,train_size=0.2, random_state=42)
#%%
#8. Perform Tokenization
#Define parameters for the following processes
vocab_size= 16000
embedding_dim =64
max_length = 800
trunc_type = 'post'
padding_type = 'post'
oov_token = '<OOV>'

#(A)Define the Tokenizer Object
tokenizer = keras.preprocessing.text.Tokenizer(
    num_words=10000,
    split=" ",
    oov_token=oov_token
)
tokenizer.fit_on_texts(X_train)

# %%
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))

# %%
#(B) Transfrom texts into tokens
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)
# %%
#9. Perform Padding
X_train_padded = keras.utils.pad_sequences(X_train_tokens,maxlen=max_length,padding=padding_type,truncating=trunc_type)
X_test_padded = keras.utils.pad_sequences(X_test_tokens,maxlen=max_length,padding=padding_type,truncating=trunc_type)
# %%
# Define a function that can reverse the tokenization
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
def decode_tokens(tokens):
    return " ".join([reverse_word_index.get(i, "?") for i in tokens])

print(decode_tokens(X_train_padded[4]))

# %%
#10. Model development
#(A) Start with Embedding layer
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))

#(B) Build RNN model, here we are using bidirectional LSTM 
model.add(Bidirectional(LSTM(16)))

#(C) Build the classification layers with Dense layers
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
# %%
#11.Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

#%%
# Check whether it Overfitting or not
# If the training accuracy is significantly higher than the validation accuracy, it indicates overfitting.
train_loss, train_accuracy = model.evaluate(X_train_padded, y_train)
print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")

# %%
#12. Model Training
max_epoch=30
early_stopping = keras.callbacks.EarlyStopping(patience=3)
history = model.fit(X_train_padded,y_train,validation_data=(X_test_padded,y_test), epochs= max_epoch, callbacks=[early_stopping])

# %%
#13. Plot graphs for training result
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training loss', "Validation loss"])
plt.show()

#(B) Accuracy Graph
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training Accuracy', "Validation Accuracy"])
plt.show()

# %%
#14. Model deployment
#(A) Convert the text into tokens
test_string = "Awww, I love this! The Tale of the Cat and the Moon doesn't really need an synopsis, as that's what it is... the cat chasing the moon, to a Spanish poem. It's the artistry that's interesting. In fact, there was this animated short called the Fan and the Flower that was an Academy Award winner last year (2005)... yeah, almost same thing, which leads me to believe it might just be a rip off.But this is a really good short, with stark black & white shapes shifting and transitioning into beautiful motion and poetic seduction... If you believe cats are poetry in motion, see this and you'll believe it more.Also, it has such a touching end."

test_token = tokenizer.texts_to_sequences(test_string)
# %%
#(B) Remove invalid entries from the token list
def remove_space(token):
    temp = []
    for i in token:
        if i !=[]:
            temp.append(i[0])
    return temp

test_token_processed = np.expand_dims(remove_space(test_token), axis=0)
# %%
#(C) Perform padding and truncating
test_token_padded = keras.utils.pad_sequences(test_token_processed, maxlen=max_length, padding=padding_type, truncating=trunc_type)
# %%
#(D) Put the padded tokens into the model for prediction
y_pred = np.argmax(model.predict(test_token_padded))
# %%
#(E) Use the label encoder to perform inverse trasform to get the class
class_prediction = label_encoder.inverse_transform(y_pred.flatten())
print(class_prediction)
# %%
#15. Save important components so that we can deploy the NLP model elsewhere
#(A) Tokenizer
PATH = os.getcwd()
tokenizer_save_path = os.path.join(PATH,"tokenizer.pkl")
with open(tokenizer_save_path,'wb') as f:
    pickle.dump(tokenizer, f)
# %%
#(B)label encoder
label_encoder_save_path = os.path.join(PATH,"label_encoder.pkl")
with open(label_encoder_save_path,'wb') as f:
    pickle.dump(label_encoder,f)
# %%
#(C)Keras model
model_save_path = os.path.join(PATH,"nlp_model")
keras.models.save_model(model,model_save_path)
# %%
#Check if all the save components can be loaded
#(A) Load the saved tokenizer
with open(tokenizer_save_path,'rb') as f:
    saved_tokenizer = pickle.load(f)
print(type(saved_tokenizer))

#(B) load the saved label encoder
with open(label_encoder_save_path,'rb') as f:
    saved_label_encoder = pickle.load(f)
print(type(saved_label_encoder))

#(C) Load the saved model
saved_model = keras.models.load_model(model_save_path)
saved_model.summary()
# %%