import time
import logging
import nltk
from collections import defaultdict

import numpy as np
from nltk.corpus import treebank, brown, conll2000

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - Spell Corrector - %(levelname)s - %(message)s')

class PartOfSpeechTagger:
    def __init__(self):
        try:
            # Load POS tagged corpora from NLTK
            self.treebank_corpus = treebank.tagged_sents()
            self.brown_corpus = brown.tagged_sents()
            self.conll_corpus = conll2000.tagged_sents()

            # Combine all corpora
            self.training_data = self.treebank_corpus + self.brown_corpus + self.conll_corpus

            # Create dictionary
            self.dictionary = set()
            for sentence in self.training_data:
                for word, _ in sentence:
                    self.dictionary.add(word)

            # Calculate transition probabilities (from tag1 to tag2)
            self.transition_counts = defaultdict(lambda: defaultdict(int))
            self.tag_counts = defaultdict(int)
            self.calculate_transition_counts()

            # Calculate emission probabilities (from word to tag)
            self.emission_counts = defaultdict(lambda: defaultdict(int))
            self.emission_probs = defaultdict(lambda: defaultdict(float))
            self.calculate_emission_counts()

        except Exception as e:
            logging.error(f"An error occurred during initialization: {str(e)}")

    def calculate_transition_counts(self):
        try:
            logging.info("Calculating transition counts")

            for sentence in self.training_data:
                prev_tag = None
                for _, tag in sentence:
                    if prev_tag is not None:
                        self.transition_counts[prev_tag][tag] += 1
                    self.tag_counts[tag] += 1
                    prev_tag = tag

            # Add start and end tags
            self.tag_counts['START'] = len(self.training_data)
            self.tag_counts['END'] = len(self.training_data)
            for sentence in self.training_data:
                self.transition_counts['START'][sentence[0][1]] += 1
                self.transition_counts[sentence[-1][1]]['END'] += 1
        except Exception as e:
            logging.error(f"An error occurred while calculating transition counts: {str(e)}")

    def calculate_emission_counts(self):
        try:
            logging.info("Calculating emission counts")
            for sentence in self.training_data:
                for word, tag in sentence:
                    self.emission_counts[tag][word] += 1

            logging.info("Calculating emission probabilities")
            for tag, word_counts in self.emission_counts.items():
                total_count = sum(word_counts.values())
                for word, count in word_counts.items():
                    self.emission_probs[tag][word] = count / total_count
        except Exception as e:
            logging.error(f"An error occurred while calculating emission counts/probabilities: {str(e)}")

    def word_not_found(self, word):
        try:
            return word not in self.dictionary
        except Exception as e:
            logging.error(f"An error occurred while searching word in dictionary: {str(e)}")

    def edit_distance(self, word1, word2):
        try:
            n = len(word1)
            m = len(word2)
            dp = np.zeros((n + 1, m + 1))
            for i in range(n + 1):
                for j in range(m + 1):
                    if i == 0:
                        dp[i][j] = j
                    elif j == 0:
                        dp[i][j] = i
                    elif word1[i - 1] == word2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        dp[i][j] = 1 + min(dp[i - 1][j],  # Deletion
                                           dp[i][j - 1],  # Insertion
                                           dp[i - 1][j - 1])  # Substitution
            return dp[n][m]
        except Exception as e:
            logging.error(f"An error occurred while calculating efficient word: {str(e)}")

    def word_corrector(self, word, initial_tag):
        try:
            min_distance = float('inf')
            closest_word = word
            for vocab_word in self.emission_counts[initial_tag]:
                distance = self.edit_distance(word, vocab_word)
                if min_distance > distance:
                    min_distance = distance
                    closest_word = vocab_word

            return closest_word
        except Exception as e:
            logging.error(f"An error occurred while correcting the word: {str(e)}")

    def tag_initializer(self):
        try:
            val = 0
            tag = ''
            for key, value in self.transition_counts.get('START').items():
                if val < value:
                    val = value
                    tag = key
            return tag
        except Exception as e:
            logging.error(f"An error occurred while initializing the tag: {str(e)}")

    def viterbi(self, sentence, initial_tag):
        try:
            no_of_word_wrong = 0

            logging.info("Viterbi function initiated")

            viterbi_probs = [defaultdict(float) for _ in range(len(sentence))]
            backpointers = [{} for _ in range(len(sentence))]

            # Initialization step
            for tag in self.tag_counts.keys():
                viterbi_probs[0][tag] = self.transition_counts['START'][tag] * self.emission_probs[tag].get(sentence[0], 0)

            # Recursion step
            for t in range(1, len(sentence)):

                if self.word_not_found(sentence[t]):
                    no_of_word_wrong += 1
                    sentence[t] = self.word_corrector(sentence[t], initial_tag)
                    logging.info(f" {no_of_word_wrong} Words found wrong, correcting it")

                for current_tag in self.tag_counts.keys():
                    if self.tag_counts[current_tag] <= 10:
                        pass
                    max_prob = 0
                    best_prev_tag = None
                    for prev_tag in self.tag_counts.keys():

                        if self.transition_counts[prev_tag][current_tag] == 0:
                            pass

                        prob = viterbi_probs[t - 1][prev_tag] * self.transition_counts[prev_tag][current_tag] * \
                               self.emission_probs[
                                   current_tag].get(sentence[t], 0)
                        if prob > max_prob:
                            max_prob = prob
                            best_prev_tag = prev_tag
                    viterbi_probs[t][current_tag] = max_prob
                    backpointers[t][current_tag] = best_prev_tag

                    for prev_tag, value in self.transition_counts[best_prev_tag].items():
                        val = 0
                        if value > val:
                            if prev_tag != 'END':
                                val = value
                                initial_tag = prev_tag

            # Termination step
            best_final_tag = max(viterbi_probs[-1], key=viterbi_probs[-1].get)

            # Backtrack to find the best sequence of tags
            best_tag_sequence = [best_final_tag]
            for t in range(len(sentence) - 1, 0, -1):
                if best_tag_sequence[-1] is not None:
                    best_tag_sequence.append(backpointers[t][best_tag_sequence[-1]])
                else:
                    break  # Exit loop if best_tag_sequence[-1] is None

            best_tag_sequence.reverse()

            corrected_sentence = ""

            for word, tag in zip(sentence, best_tag_sequence):
                corrected_sentence += word + " "

            return corrected_sentence.strip()

        except Exception as e:
            logging.error(f"Encountered an critical error while parsing viterbi algorithm for sentence: {str(e)}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Performing POS tag transition and probabilities calculation")

    tagger = PartOfSpeechTagger()

    logging.info("Initializing word correctness functionalities")

    initial_tag = tagger.tag_initializer()

    test_paragraph = "A lot of people belive that they're are too many prblems in the world today, but we must remeber that with hardwork and determination, we can make a difference. Its important to focuss on solutins rather than dwelling on the negtive aspects. We should alwayes strive to be the change we want to see in the world. Let's not let fear or doubt hold us back from acheiving our goals."
    sentences_matrix = test_paragraph.split('. ')

    corrected_result = ""

    start_time = time.time()

    for sentence in sentences_matrix:
        corrected_result += tagger.viterbi(sentence.split(" "), initial_tag) + ". "

    logging.info(corrected_result)

    end_time = time.time()
    logging.info("Execution Time: %s seconds", end_time - start_time)
