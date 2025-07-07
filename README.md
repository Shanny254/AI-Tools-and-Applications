# AI Tools and Applications 

## Project Title: Mastering the AI Toolkit – Real-World Implementation using Scikit-learn, TensorFlow, and spaCy

This project demonstrates the application of multiple AI tools and frameworks to solve real-world problems. It includes classical machine learning, deep learning, and natural language processing (NLP) components as outlined in the AI Tools and Applications assignment structure.

---

## Table of Contents

- [Overview](#overview)
- [Tools Used](#tools-used)
- [Part 1: Theoretical Questions](#part-1-theoretical-questions)
- [Part 2: Practical Implementation](#part-2-practical-implementation)
  - [Task 1: Scikit-learn (Decision Tree)](#task-1-scikit-learn-decision-tree)
  - [Task 2: TensorFlow (CNN for MNIST)](#task-2-tensorflow-cnn-for-mnist)
  - [Task 3: spaCy (NER + Sentiment)](#task-3-spacy-ner--sentiment)
- [Part 3: Ethics and Optimization](#part-3-ethics-and-optimization)
- [How to Run](#how-to-run)


---

## Overview

The goal of this assignment is to showcase our mastery of essential AI tools by applying them to real-world datasets and problems. Each task corresponds to a key area of AI:

- Classical Machine Learning with Scikit-learn
- Deep Learning with TensorFlow
- NLP with spaCy

---

## Tools Used

- Python
- Jupyter Notebook / Google Colab
- Scikit-learn
- TensorFlow / Keras
- spaCy
- Matplotlib & Seaborn
- GitHub for version control

---

## Part 1: Theoretical Questions

**Q1: TensorFlow vs PyTorch**
- TensorFlow is more production-oriented, with better deployment options (e.g., TensorFlow Lite, TensorFlow.js), while PyTorch is more research-friendly due to its dynamic computation graph.
- Choose TensorFlow for mobile/web deployment, PyTorch for experimentation and fast prototyping.

**Q2: Two use cases for Jupyter Notebooks**
- Visualizing model training metrics (loss, accuracy) interactively.
- Sharing reproducible AI experiments with documentation and outputs.

**Q3: spaCy vs basic string operations**
- spaCy provides contextual understanding, entity recognition, and syntactic parsing which plain string operations cannot do efficiently or accurately.

**Comparative: Scikit-learn vs TensorFlow**
| Feature             | Scikit-learn            | TensorFlow              |
|---------------------|-------------------------|--------------------------|
| Target Application  | Classical ML            | Deep Learning            |
| Beginner Friendly   | Yes                     | Moderate                 |
| Community Support   | Strong                  | Strong                   |

---

## Part 2: Practical Implementation

### Task 1: Scikit-learn (Decision Tree)

**Dataset**: Iris Species  
**Goal**: Predict flower species using a decision tree classifier.

**Steps:**
- Load the Iris dataset
- Encode labels and handle missing values
- Train/test split
- Train a DecisionTreeClassifier
- Evaluate using accuracy, precision, and recall

> File: `iris_decision_tree.ipynb`

---

### Task 2: TensorFlow (CNN for MNIST)

**Dataset**: MNIST Handwritten Digits  
**Goal**: Train a CNN model to classify digits with >95% accuracy

**Steps:**
- Normalize and reshape input data
- Build a CNN using Keras (Conv2D, MaxPooling, Dense)
- Train the model and evaluate performance
- Visualize predictions on sample test images

> File: `mnist_cnn_tensorflow.ipynb`

---

### Task 3: spaCy (NER + Sentiment)

**Data**: Sample Amazon product reviews  
**Goal**: Extract named entities and perform sentiment analysis

**Steps:**
- Load `en_core_web_sm` spaCy model
- Extract entities such as products and organizations
- Apply a rule-based keyword sentiment analyzer
- Print sentiment and entity output per review

> File: `spacy_nlp_analysis.ipynb`

---

## Part 3: Ethics and Optimization

### Ethical Considerations:
- **Bias in Sentiment & NER Models**: Pretrained models like `en_core_web_sm` may reflect dataset biases.
- **Fairness in ML**: Evaluate real-world impacts before deployment (e.g., consider using TensorFlow Fairness Indicators).
- **Data Privacy**: Sensitive user reviews or health records must be anonymized.

### Troubleshooting & Optimization:
- We debugged dimension mismatches in the CNN model and used appropriate loss functions (`categorical_crossentropy`).
- Optimized the rule-based sentiment analyzer to avoid false positives.

---

## How to Run

### Clone the Repository

```bash
git clone https://github.com/your-username/ai-tools-assignment.git
cd ai-tools-assignment

