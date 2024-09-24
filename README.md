# Comment Toxicity Detection Model

This project is a multi-class classification model designed to detect and classify toxic comments into one or more of the following categories: 
- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

The model takes user comments as input and predicts whether the comment belongs to any of these categories, providing a probability score for each class.

## Project Overview

The comment toxicity detection model was developed using **bidirectional LSTM** layers and an **embedding layer** to capture word-level contextual embeddings. Preprocessing steps included text cleaning using regular expressions and the NLTK library.

The model outputs a set of probabilities for each of the six classes, and a threshold of 0.4 is applied to determine whether the comment falls into any given category.

### Achievements
- **99% validation accuracy** achieved during model training.
- Efficient handling of text data with **preprocessing**, **word embeddings**, and a well-tuned model architecture.

## Model Architecture

The model uses:
- **Embedding Layer**: To convert text into dense vectors that represent the semantic meaning of words.
- **Bidirectional LSTM**: To capture contextual dependencies in both forward and backward directions.
- **Dense Layers**: For classification into the six categories.

## Data Preprocessing

Before training, comments are cleaned and processed using:
- Regular expressions to remove unwanted symbols, links, and non-English text.
- Tokenization using the `TextVectorization` layer.
- Padding and truncation to ensure all input sequences have uniform length.

## Usage

You can use this model to predict the toxicity of comments in real-time. Below is an example of how to use it in a **Streamlit** front end.

### Prediction Example

To predict the toxicity of a comment, you can load the model and pass your text as input. The output will be a set of probabilities for each class. If any of the class probabilities exceed 0.4, that class will be considered a predicted category for the comment.

### Streamlit Frontend

The model is integrated with a **Streamlit** app that provides a simple and interactive user interface. Users can input a comment, and the app will display:
- The original text
- Predicted classes (if any)
- A bar graph showing the probability distribution across all toxicity categories.

## How to Run

### Prerequisites
- Python 3.8+
- TensorFlow
- Streamlit
- NLTK

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/YashMasane/Comment-Toxicity-Model.git
    cd comment-toxicity-detection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Model Training

The model was trained on a dataset of user comments labeled with the six target categories. Text data was tokenized, and the vocabulary was generated using TensorFlow's `TextVectorization` layer. The model was trained using a custom bidirectional LSTM architecture, optimized with **categorical cross-entropy loss**.

## Results

- Validation accuracy: **99%**
- Precision, recall, and F1-score were evaluated for each class to ensure robust performance.

## Future Improvements

- **Expand the dataset** to include a wider variety of toxic comments and contexts.
- **Optimize the model** for deployment in low-latency environments.
- Add more advanced NLP techniques like **transformers** for better performance.


## Contact

For questions or suggestions, feel free to contact [Yash Raju Masane](mailto:masaneyash6@gmail.com).
