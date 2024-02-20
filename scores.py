def main(incorrect_sentence, output_sentence, actual_correct_sentence):
    # Split sentences into words
    incorrect_words = incorrect_sentence.split()
    output_words = output_sentence.split()
    actual_correct_words = actual_correct_sentence.split()

    # Initialize counts for true positives, true negatives, false positives, false negatives
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    # Calculate the four parameters
    for i in range(len(incorrect_words)):
        if incorrect_words[i] == actual_correct_words[i]:
            if output_words[i] == actual_correct_words[i]:
                true_positives += 1  # Word was actually correct and output is correct
            else:
                false_negatives += 1  # Word was actually correct but output is incorrect
        else:
            if output_words[i] == actual_correct_words[i]:
                false_positives += 1  # Word was actually incorrect but output is correct
            else:
                true_negatives += 1  # Word was actually incorrect and output is also incorrect

    # Calculate accuracy, precision, recall, and F1 score
    total_words = len(incorrect_words)
    accuracy = (true_positives + true_negatives) / total_words
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Print the evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)


# Example usage:
incorrect_sentence = "A lot of people belive that they're are too many prblems in the world today, but we must remeber that with hardwork and determination, we can make a difference. Its important to focuss on solutins rather than dwelling on the negtive aspects. We should alwayes strive to be the change we want to see in the world. Let's not let fear or doubt hold us back from acheiving our goals."
output_sentence = "A lot of people best that they're are too many pro in the world to but we must fewer that with hardest and terminated we can make a difference. Its important to Such on such rather than dwelling on the best aspects. We should ) strive to be the change we want to see in the world. Let's not let fear or doubt hold us back from de our most."
actual_correct_sentence = "A lot of people believe that there are too many problems in the world today, but we must remember that with hard work and determination, we can make a difference. It's important to focus on solutions rather than dwelling on the negative aspects. We should always strive to be the change we want to see in the world. Let's not let fear or doubt hold us back from achieving our goals."
main(incorrect_sentence, output_sentence, actual_correct_sentence)
