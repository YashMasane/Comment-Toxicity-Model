# importing libraries

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from helper import processed_text


# Load your pre-trained model 
def load_pretrained_model():
    model = tf.keras.models.load_model('models/toxic.keras')  
    return model

# Load the pre-trained model
model = load_pretrained_model()

# Define the classes
classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Function to predict classes and probabilities
def predict_toxicity(text):
    
    vectorized_text = processed_text(text)  # Modify this as per your preprocessing
    
    # Predict probabilities
    probabilities = model.predict(np.expand_dims(vectorized_text, 0))[0]
    
    # Determine predicted classes (those with probability > 0.5)
    predicted_classes = [classes[i] for i, prob in enumerate(probabilities) if prob >= 0.5]
    
    return probabilities, predicted_classes

# Streamlit Front End
st.title('Comment Toxicity Detection')

# Input field for the comment
user_input = st.text_area("Enter a comment:", "")

if st.button('Predict'):
    if user_input:
        # Make predictions
        probabilities, predicted_classes = predict_toxicity(user_input)

        # Create two columns
        col1, col2 = st.columns([2, 3])  # Adjust column width as needed

        with col2:  # Right Column
            # Display the original text
            st.subheader("Original Comment:")
            st.write(user_input)

            # Display the predicted classes
            st.subheader("Predicted Toxicity Classes:")
            if predicted_classes:
                # Capitalize, colorize, and bold each class
                for i, cls in enumerate(predicted_classes):
                    color = f'<span style="color:{plt.cm.Set1(i % 9)}; font-weight:bold;">{cls.capitalize()}</span>'
                    st.markdown(color, unsafe_allow_html=True)
            else:
                st.write("No toxic content detected.")

        with col1:  # Left Column
            # Plot the vertical bar graph of probabilities with different colors
            st.subheader("Class Probabilities:")

            # Define colors for each bar (one for each class)
            colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']

            fig, ax = plt.subplots()
            ax.bar(classes, probabilities, color=colors)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability')
            ax.set_title('Toxicity Class Probabilities')

            # Display the bar graph
            st.pyplot(fig)
    else:
        st.write("Please enter a comment to predict.")

# print('Text vectorization for input is', processed_text('Yor are a bad person'))