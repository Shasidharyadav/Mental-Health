# Chitti - AI-Powered Mental Health Chatbot 🌟

## Project Description
Mental health is a vital aspect of overall well-being, yet many individuals face barriers in accessing the support they need. Introducing Chitti, the mental health assistant that makes you feel better!

Chitti is a chatbot that uses Natural Language Processing (NLP) techniques to understand users' problems and generate responses accordingly. It is built using a Flask backend, NLTK, TensorFlow, and Keras for model training, and JavaScript for the frontend.

## Features
- **Natural Language Understanding:** Chitti can understand and respond to a wide range of mental health-related queries.
- **Context-Aware Conversations:** Maintains context over multiple interactions to provide coherent and relevant responses.
- **User-Friendly Interface:** Accessible via a simple and intuitive web interface.
- **24/7 Availability:** Always available to provide support whenever needed.
- **Privacy and Security:** Ensures user data is encrypted and protected, complying with relevant privacy regulations.

## Tech Stack
- **Backend:** Flask
- **Natural Language Processing:** NLTK
- **Machine Learning:** TensorFlow, Keras
- **Frontend:** HTML, CSS, JavaScript

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/chitti-mental-health-chatbot.git
    cd chitti-mental-health-chatbot
    ```

2. **Create a virtual environment and activate it:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Training the Model

1. **Prepare your dataset and store the intents in `intents.json`.**

2. **Train the chatbot model:**
    ```bash
    python train_chatbot.py
    ```

## Running the Application

1. **Start the Flask server:**
    ```bash
    python app.py
    ```

2. **Open your web browser and navigate to:**
    ```
    http://127.0.0.1:5000/
   ```

 ## Code Explanation
   ### Intent File (`intents.json`)
Define the various intents and responses for the chatbot.

  
## Model Training
**Model Training**
```bash
  (train_chatbot.py)
   ```

**Script to preprocess data and train the chatbot model.**


## Run in browser
  **Run it in python**
