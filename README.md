# French Sentiment Analysis

This project implements a sentiment analysis model using Long Short-Term Memory (LSTM) networks to classify customer reviews as **happy**, **sad**, or **neutral**. The model is trained on a dataset containing text reviews, and special attention has been given to address class imbalance through data resampling techniques.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data Resampling](#data-resampling)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)

## Project Overview

The main objective of this project is to develop a robust sentiment analysis model that can accurately classify sentiments in customer reviews. Given the imbalanced nature of the dataset, we employed resampling techniques to ensure all classes are adequately represented, improving the model's predictive capabilities.

## Installation

To run this project, ensure you have Python 3.6+ and the following libraries installed:

```bash
pip install numpy pandas scikit-learn imbalanced-learn tensorflow fastapi uvicorn
```

## Data Resampling

To address class imbalance in the dataset, we applied **Random Oversampling**. The distribution of sentiment classes before and after resampling is as follows:

- **Original Distribution:**
  - Happy: 205,041 samples
  - Sad: 25,637 samples
  - Neutral: 23,277 samples

- **Balanced Distribution After Oversampling:**
  - Happy: 205,041 samples
  - Sad: 205,041 samples
  - Neutral: 205,041 samples

This approach allows the model to learn effectively from all classes.

## Model Architecture

The LSTM model is structured as follows:

- **Input Layer:** Receives preprocessed text data.
- **Embedding Layer:** Converts words to dense vector representations.
- **LSTM Layer:** Captures temporal dependencies in the sequence of words.
- **Dense Layer:** Outputs the sentiment classification.

## Results

After training the model with the balanced dataset, we achieved a test accuracy of **83.84%**. The model shows promising results in classifying sentiments accurately, particularly for the minority classes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgements

- Special thanks to the contributors and libraries that made this project possible, including TensorFlow, FastAPI, and scikit-learn.
