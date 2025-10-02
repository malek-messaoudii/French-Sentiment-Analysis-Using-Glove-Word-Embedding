# French Tweet Sentiment Analysis with GloVe Embeddings

This project explores sentiment analysis on French tweets using machine learning techniques and GloVe word embeddings. The goal is to predict whether a tweet is positive or negative based on its textual content.

## Project Description

The notebook "Word Embedding French Sentiment Analysis" provides a step-by-step approach to:

- Clean and preprocess French textual data.

- Tokenize text into sequences of words.

- Use pre-trained GloVe embeddings to represent each tweet as a vector.

- Train a classification model to predict the sentiment of tweets.

- Use Gradio to make interface and see results

## Methodology

### Data Preprocessing:

- Text cleaning (removing punctuation, stopwords, etc.).

- Tokenization of text into words.

- Vectorization with GloVe:

- Use pre-trained GloVe embeddings to get a vector for each word.

- Compute the average of word vectors to represent each tweet as a fixed-size vector.

### Classifier Training:

- Use embeddings as features to train a classification model .

### Model Evaluation:

Evaluate the model’s performance on a test dataset.

## Prerequisites

- Python 3.x

Required libraries:

- pandas
  
- Ntlk

- numpy

- gensim

- scikit-learn

- Wordcloud

- gradio (optional, for interactive demo)

## Usage

1. Clone the repository:
```
git https://github.com/malek-messaoudii/French-Sentiment-Analysis-Using-Glove-Word-Embedding.git
cd French-Sentiment-Analysis-Using-Glove-Word-Embedding
```

2. Run the notebook or script to generate embeddings and train the model.

3. (Optional) Launch the Gradio interface for interactive sentiment prediction:
````
interface.launch()
````
glove_vectors.pkl : Precomputed GloVe embeddings.
