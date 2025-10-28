
# AirPods Review Sentiment Analysis

## Overview
This project performs sentiment analysis on AirPods product reviews using NLP techniques and deep learning (PyTorch). 
It classifies reviews into sentiment categories (positive, negative, neutral) and visualizes insights through detailed EDA. 
The implementation includes Feedforward and LSTM-based models, with custom preprocessing and vectorization.

---

## How to Run the Notebook

1. **Install Dependencies**

   Open your terminal and run:
   ```bash
   pip install pandas numpy matplotlib seaborn wordcloud torch scikit-learn
   ```

2. **Run the Notebook**

   - Open `try_code.ipynb` in VS Code or Jupyter Notebook.
   - Ensure that `AirPodsPro_Reviews.csv` is in the same directory.
   - Run all cells sequentially.

---

## Libraries Used

- **pandas**: Data manipulation and cleaning
- **numpy**: Numerical computations
- **matplotlib, seaborn**: Data visualization
- **wordcloud**: Word cloud generation
- **torch (PyTorch)**: Deep learning implementation
- **scikit-learn**: Metrics, label encoding, train-test splitting

---

## Dependencies

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- wordcloud
- torch
- scikit-learn

---

## Notes

- The dataset is preprocessed (duplicates removed, reviews cleaned, tokenized, and vectorized).
- Custom tokenization and vocabulary creation have been implemented.
- Simple Feedforward Neural Network and LSTM-based models are trained and evaluated.
- Class imbalance is addressed through class weighting in loss calculation.
- Visualizations include sentiment distribution, review length distribution, and word clouds.

