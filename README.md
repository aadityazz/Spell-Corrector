# Spelling Correction 

`version 0.2`

This project implements a part-of-speech tagger using the Viterbi algorithm to correct spelling mistakes in text. It leverages concepts from natural language processing (NLP) and statistical language models to identify and correct errors in input text.

## Overview

The main objective of this project is to demonstrate how the Viterbi algorithm, commonly used in NLP for part-of-speech tagging tasks, can be applied to correct spelling mistakes. The algorithm utilizes probabilities of transitioning between parts of speech and observing words given their respective parts of speech to make informed corrections.

## Features

- **Part-of-Speech Tagging**: Utilizes a corpus of tagged sentences to train the Viterbi algorithm for part-of-speech tagging.
- **Spelling Correction**: Corrects spelling mistakes by identifying the most likely sequence of parts of speech for each word in the input text.
- **Edit Distance**: Implements edit distance to find the closest word in the dictionary for words not found during part-of-speech tagging.
- **Performance Metrics**: Calculates accuracy and precision metrics to evaluate the effectiveness of the spelling correction.

## Usage

1. Install the required dependencies using `pip install -r requirements.txt`.
2. Prepare your input text file with the text you want to correct (e.g., `input.txt`).
3. Run the `spell_corrector.py` script to perform spelling correction on the input text.
4. View the corrected text along with accuracy and precision metrics in the console output.

## Technology Stack

### NLTK (Natural Language Toolkit)
We utilized NLTK, a powerful Python library for working with human language data. NLTK provided access to corpora such as Treebank, Brown, and CoNLL2000, which were used to train our spelling correction model.

### Viterbi Algorithm
The Viterbi algorithm played a key role in optimizing the performance of our spelling correction system. By efficiently calculating the most probable sequence of parts of speech for each word, we reduced the processing time significantly. For instance, processing a 1000-word paragraph now takes only 20 seconds compared to the previous version, which took 100 seconds.

### Edit Distance (Dynamic Programming)
We implemented the edit distance algorithm using dynamic programming techniques. This algorithm allowed us to find the shortest distance between correct and misspelled words, leading to more accurate spelling correction.

### Exception Handling
To ensure user-friendly execution and robustness, we implemented try-except blocks to catch and handle any potential errors during program execution gracefully. This approach prevents abrupt termination and provides informative error messages to users.

### Logging
We integrated a logging mechanism into our system to track and monitor program execution. This logging feature provides regular updates and insights into the execution flow, aiding in debugging and optimization efforts.

## Caution
While the project has improved efficiency in processing time compared to previous versions, there has been a decrease in accuracy matrices. We are actively working on improving accuracy while maintaining efficiency.

#### Time Matrix: (for 1000 words)

| Metric   | Current  | Previous |
|----------|----------|----------|
| Time (s) | 20       | 120      |

#### Accuracy Matrix: (for 1000 words)

| Metric     | Current    | Previous  |
|------------|------------|-----------|
| Accuracy   | 80% - 90%  | 50% - 55% |


## Future Plans
We are currently focusing on data analysis to identify the best possible ways to increase the accuracy of the spelling correction system. Our goal is to enhance the correctness of the corrections while preserving the performance gains achieved so far.
