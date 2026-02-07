# Credit Card Fraud Detection

A deep learning project that uses Convolutional Neural Networks (CNN) to detect fraudulent credit card transactions with high accuracy.


## 📋 Overview

This project implements a machine learning solution to identify fraudulent credit card transactions using a 1D Convolutional Neural Network. The model is trained on transaction data and achieves approximately 92% validation accuracy in detecting fraudulent activities.

Credit card fraud is a significant concern in the financial industry, causing billions of dollars in losses annually. This project aims to help financial institutions automatically flag potentially fraudulent transactions for review.

## ✨ Features

- **Deep Learning Model**: Implements a 1D CNN architecture optimized for sequential transaction data
- **High Accuracy**: Achieves ~92% validation accuracy on test data
- **Data Preprocessing**: Includes StandardScaler normalization for feature scaling
- **Visualization**: Training history plots showing model performance over epochs
- **Binary Classification**: Distinguishes between legitimate and fraudulent transactions

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow/Keras**: Deep learning framework for building the CNN model
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Scikit-learn**: Data preprocessing and train-test split

## 📊 Model Architecture

The CNN model consists of:
- **Input Layer**: Accepts transaction features
- **Conv1D Layer 1**: 32 filters, kernel size 2, ReLU activation
- **Batch Normalization**: Normalizes activations
- **Dropout (0.2)**: Prevents overfitting
- **Conv1D Layer 2**: 64 filters, kernel size 2, ReLU activation
- **Batch Normalization**: Normalizes activations
- **Dropout (0.5)**: Prevents overfitting
- **Flatten Layer**: Converts 2D features to 1D
- **Dense Layer**: 64 neurons, ReLU activation
- **Dropout (0.5)**: Prevents overfitting
- **Output Layer**: 1 neuron, Sigmoid activation for binary classification

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/anirudhch7/Credit-card-fraud-detection.git
cd Credit-card-fraud-detection
```

2. Install required dependencies:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn jupyter
```

## 🚀 Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook "Creditcards (1).ipynb"
```

2. Ensure you have the credit card dataset (`creditcard.csv`) in the appropriate location

3. Run all cells in the notebook to:
   - Load and preprocess the data
   - Build the CNN model
   - Train the model for 20 epochs
   - Visualize training history
   - Evaluate model performance

## 📈 Results

The model demonstrates strong performance:
- **Training Accuracy**: Reaches up to ~94% by epoch 20
- **Validation Accuracy**: Stabilizes around ~92%
- **Loss**: Binary crossentropy loss decreases consistently during training
- **Optimizer**: Adam optimizer with learning rate of 0.0001

## 📝 Dataset

The project uses a credit card transaction dataset containing:
- Time-series transaction data
- Multiple anonymized features (V1-V28)
- Transaction amount
- Binary class label (0 = legitimate, 1 = fraudulent)

**Note**: The dataset should be placed in the working directory as `creditcard.csv`

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is available for educational and research purposes.

## 👤 Author

**Anirudh**
- GitHub: [@anirudhch7](https://github.com/anirudhch7)

## 🙏 Acknowledgments

- Credit card dataset providers
- TensorFlow and Keras communities
- Open source contributors
