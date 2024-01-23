## IMDB Sentiment Analysis
### Overview
This Python project focuses on sentiment analysis using the IMDB movie reviews dataset. The primary goal is to develop a predictive model that can classify movie reviews as positive or negative based on the sentiments expressed in the text. The project involves data loading, exploration, cleaning, tokenization, model development using Bidirectional LSTM, training, evaluation, deployment, and the creation of a Streamlit app for interactive use.

### Project Structure
The project is structured into various sections, each serving a specific purpose:

#### 1. Import Packages
The necessary Python packages are imported for data manipulation, machine learning, and visualization. Key packages include Pandas, NumPy, TensorFlow, Matplotlib, Scikit-learn, and Streamlit.

#### 2. Data Loading
The IMDB dataset is loaded from a specified URL using Pandas. This dataset contains movie reviews labeled with sentiments (positive/negative).

#### 3. Data Inspection
Basic information about the dataset is displayed, including its shape, data types, and a sample of the data. This step helps understand the structure of the dataset.

#### 4. Data Cleaning
Data cleaning involves checking for missing values and duplicates. Duplicates are removed from the dataset to ensure data integrity.

#### 5. Split Data into Features and Labels
The dataset is split into features (movie reviews) and labels (sentiments).

#### 6. Convert Categorical Label into Integer
Label encoding is performed to convert categorical sentiment labels into numerical values for model training.

#### 7. Perform Train-Test Split
The dataset is split into training and testing sets for model evaluation.

#### 8. Perform Tokenization
Text data is tokenized using TensorFlow's Tokenizer. This step converts text into sequences of integers.

#### 9. Perform Padding
Token sequences are padded to ensure uniform length, preparing the data for model input.

#### 10. Model Development
The neural network model is built using TensorFlow. It includes an Embedding layer, Bidirectional LSTM layer, and Dense layers for classification.

#### 11. Compile the Model
The model is compiled with appropriate settings, including the optimizer and loss function.

#### 12. Check for Overfitting
The training accuracy is checked to identify potential overfitting.

#### 13. Model Training
The model is trained on the training set, and training history is monitored for early stopping.

#### 14. Plot Graphs for Training Result
Graphs depicting training loss, validation loss, training accuracy, and validation accuracy are plotted for analysis.

#### 15. Model Deployment
The model is deployed by saving essential components, including the Tokenizer, Label Encoder, and the trained Keras model.

#### 16. Streamlit App
A Streamlit app is created to provide an interactive interface for users to input movie reviews and receive real-time sentiment predictions. The app uses the saved Tokenizer, Label Encoder, and Keras model for predictions.
