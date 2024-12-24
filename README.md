# gpt2
gpt2 from Scratch

# Project overview

# GPT-2 from Scratch for Next-Token Prediction

## Project Overview

This project involves the development and training of a GPT-2 model from scratch to predict the next token in a sequence. The implementation demonstrates a deep understanding of transformer architectures and their application in natural language processing tasks using both raw text and structured CSV data.

## Features

- **Custom Dataset Handling**: Preprocessed raw text documents and structured CSV inputs using a custom `GPTDataset` class.
- **Dynamic Data Loading**: Created a robust data pipeline to split datasets into training and validation sets, with sanity checks to ensure data adequacy.
- **Transformer Training**: Implemented advanced training techniques, including:
  - AdamW optimizer
  - Learning rate scheduling
  - Gradient clipping
- **Validation Pipeline**: Validated the model after each epoch to monitor performance.
- **Dynamic Loss Visualization**: Developed scripts to plot and save training and validation loss, ensuring easy tracking of model convergence.
- **Model Checkpointing**: Saved trained model weights and configurations for reproducibility and future use.

## Technologies Used

- **Frameworks and Libraries**: PyTorch, Transformers, Matplotlib
- **Tokenization**: Tiktoken (GPT-2 Tokenizer)
- **Programming Language**: Python

## How to Use

### Prerequisites

1. Install Python 3.8 or later.
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt

### Project structure
```plaintext
GPT2-From-Scratch/
├── data/                # Contains raw text and CSV files for training
├── src/                 # Source code for dataset handling, model, and utilities
│   ├── create_dataset/  # Data loading and splitting scripts
│   ├── models/          # Custom GPT-2 model implementation
├── tests/               # Training, validation, and visualization scripts
├── chatbot_trained.py   # File that runs the chatbot interface
├── main.py              # Entry point for training the model
├── requirements.txt     # List of dependencies
└── README.md            # Project documentation

```
### Reference
The implementation of the GPT-2 model in this project was inspired by the concepts and methods presented in the book 'Build a Large Language Model (From Scratch)' by ### Sebastian Raschka. This book provided valuable insights into neural network architectures and training processes, which I adapted to build the GPT-2 model.
