import sys
import time
import logging
from spell_corrector_package.pos_tagger import PartOfSpeechTagger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - Spell Corrector - %(levelname)s - %(message)s')

def correct_spelling(input_file, output_file):
    tagger = PartOfSpeechTagger()

    logging.info("Reading input file: %s", input_file)
    with open(input_file, 'r') as f:
        text = f.read()

    sentences = text.split('. ')
    corrected_sentences = []
    initial_tag = tagger.tag_initializer()

    logging.info("Performing spell checking and POS tagging...")
    start = time.time()

    for sentence in sentences:
        corrected_sentence = tagger.viterbi(sentence.split(" "), initial_tag)
        corrected_sentences.append(corrected_sentence)

    corrected_text = '. '.join(corrected_sentences)

    logging.info("Writing corrected text to output file: %s", output_file)

    with open(output_file, 'w') as f:
        f.write(corrected_text)

    end = time.time()
    logging.info("Process completed in %.2f seconds.", end - start)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file output_file")
        sys.exit(1)

    input_file = sys.argv[1]  # Path to input file
    output_file = sys.argv[2]  # Path to output file

    correct_spelling(input_file, output_file)

