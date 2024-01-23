#1. Import packages
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
import numpy  
import os,pickle,re

#2. Functions to load the pickle object and keras model
def load_pickle_file(filepath):
    with open(filepath,'rb') as f:
        pickle_object = pickle.load(f)
    return pickle_object

@st.cache_resource
def load_model(filepath):
    model_loaded = keras.models.load_model(filepath)
    return model_loaded

#3. Define the file paths towards the things we need to load
PATH = os.getcwd()
tokenizer_filepath = os.path.join(PATH,'tokenizer.pkl')
label_encoder_path = os.path.join(PATH,'label_encoder.pkl')
model_filepath = os.path.join(PATH,'nlp_model')

#4. Load the tokenizer, label encoder and model
tokenizer = load_pickle_file(tokenizer_filepath)
label_encoder = load_pickle_file(label_encoder_path)
model = load_model(model_filepath)

#5. Build the component of the streamlit app
#(A) A title text to display the app name
st.title("Text Classification of Movie Reviews")
#(B) Create a text input widget for the user to type in the news article
with st.form('input_form'):
    text_input = st.text_area("Input your review here: ")
    submitted = st.form_submit_button("Submit")

text_inputs = [text_input]

#(C) Process the input string
#Remove unwanted string
def remove_unwanted_string(text_inputs):
    for index, data in enumerate(text_inputs):
        text_inputs[index] = re.sub('<.*?>', " ", data)
        text_inputs[index] = re.sub('[^a-zA-Z]', " ", data).lower()
    return text_inputs
#a. Use the function to removed unwanted string
text_removed = remove_unwanted_string(text_inputs)
#b. Tokenize the string
text_token = tokenizer.texts_to_sequences(text_removed)
#c. Padding and truncating
text_padded = keras.utils.pad_sequences(text_token,maxlen=(200),padding="post",truncating="post")

#(D). Use the model to perfrom prediction
y_pred = np.argmax(model.predict(text_padded),axis=1)
#(E). Display the final result
label_map = {i:classes for i, classes in enumerate(label_encoder.classes_)}
result = label_map[y_pred[0]]
#(F). Write the prediction onto streamlit to display
st.write("The Review is : " + result)