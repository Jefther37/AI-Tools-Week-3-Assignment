# AI Tools and Applications Assignment

## Theme: Mastering the AI Toolkit

This repository contains the code, documentation, and analysis for the AI Tools and Applications group assignment submitted as part of the PLP Academy course. The goal of this assignment was to explore and demonstrate practical use of various AI tools, including TensorFlow, PyTorch, Scikit-learn, and spaCy, while also considering ethical implications of AI development.

The assignment was done by Group 73 which is comprised of the following members:

**Names/ Contacts**

**1. Jefther Afuyo**    **Email address:** *afuyojefther@gmail.com*

**2. Liza Bambu**     **Email address:** *lizabambu544@gmail.com*

**3. Simon Mwangi**     **Email address:**  *mwangisimone007@gmail.com*

## Project Structure

## Part 1: Theoretical Understanding

### Short Answer Questions

1. **TensorFlow vs PyTorch**  
   TensorFlow is known for production-ready scalability and deployment tools (especially via TensorFlow Serving and TFX), while PyTorch is more user-friendly for research and experimentation due to its dynamic computation graph.

2. **Use Cases for Jupyter Notebooks**  
   - Exploratory Data Analysis (EDA) and rapid prototyping  
   - Step-by-step visualization and documentation of ML workflows  

3. **spaCy vs Python String Ops**  
   spaCy provides efficient and accurate linguistic features such as Named Entity Recognition, Part-of-Speech tagging, and Dependency Parsing, which basic Python string methods cannot offer.

### Comparative Analysis: Scikit-learn vs TensorFlow

| Feature                 | Scikit-learn                   | TensorFlow                         |
|------------------------|--------------------------------|------------------------------------|
| Target Applications    | Classical ML                   | Deep learning                      |
| Beginner Friendliness  | High                           | Moderate                           |
| Community Support      | Strong, well-documented        | Very strong, extensive ecosystem   |


## Part 2: Practical Implementation

### Task 1: Classical ML with Scikit-learn

**Dataset**: Iris Species  
**Goal**: Build a decision tree classifier to predict iris species.

Steps:
- Preprocessed data and handled label encoding
- Trained a DecisionTreeClassifier
- Evaluated using accuracy, precision, and recall

### Task 2: Deep Learning with TensorFlow (Done using google colab)

**Dataset**: MNIST Handwritten Digits  
**Goal**: Build a CNN model to classify handwritten digits with >95% accuracy.

Steps:
- Preprocessed the dataset
- Built a CNN using TensorFlow/Keras
- Achieved test accuracy of 98%
- Visualized 5 sample predictions

### Task 3: NLP with spaCy

**Dataset**: Amazon Product Reviews (user reviews sample)  
**Goals**:
- Performed Named Entity Recognition (NER) to extract product names and brands
- Applied rule-based sentiment classification (positive/negative)

Tools:
- spaCy for NER
- Custom rules for sentiment tagging

## Part 3: Ethics and Optimization

### Ethical Considerations

- Identified potential dataset bias in MNIST (digit styles may differ by region and education).
- Noted bias in Amazon Reviews sentiment classification due to context ambiguity and sarcasm.
- Recommended using tools like TensorFlow Fairness Indicators and improving rule coverage in spaCy pipeline

### Troubleshooting: Debugging TensorFlow Script

- Diagnosed and fixed a buggy script containing shape mismatches and incorrect loss function for classification.
- Corrected input dimensions and used categorical cross-entropy with `softmax` output.


**Screenshots**

**All major outputs and model metrics are captured as screenshots in the final report PDF and referenced in respective notebook markdown cells.

**Authors** 

**This project was completed as a group assignment under the PLP Academy AI for Software Engineering course.**

**The authors were Group 73 with the following members who had maximum input:**

**Names/ Contacts**

**1. Jefther Afuyo**    afuyojefther@gmail.com

**2. Liza **Bambu**       lizabambu544@gmail.com

**3. Simon Mwangi**     mwangisimone007@gmail.com

## Requirements

To run the code, install the following dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow spacy
python -m spacy download en_core_web_sm
