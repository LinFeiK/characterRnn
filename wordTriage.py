# get the English words that contain 'qu'
# data = open('quwords-fr.txt', 'r').read()
# q_words = data.split('\n')
#
# for word in q_words:
#     for i in range(len(word)):
#         if i != 0:
#             prev = word[i - 1]
#             curr = word[i]
#             if (prev == 'q' and curr != 'u') or (curr == 'q' and i == len(word) - 1):
#                 print(word)
#                 break

import random

data = open('quwords-fr.txt', 'r').read()
qu_words = data.split('\n')

training_num_words = [100, 500, 1000, 1500, 2000]
test_num_words = 10

# take test_num_words random words from the whole population, without replacement
test_words = random.sample(qu_words, test_num_words)

test_f = open('quwords-test-10-fr.txt', 'w')


# write the given words to the specified file
def write_to_file(file, words):
    for j in range(len(words)):
        file.write(str(words[j]))
        if j != len(words) - 1:
            file.write('\n')


write_to_file(test_f, test_words)

# make sure the input words don't contain the test words
for i in range(len(test_words)):
    qu_words.remove(test_words[i])

training_f = open('quwords-training-2459-fr.txt', 'w')
write_to_file(training_f, qu_words)

# take training_num_words from the resulting words
for num in training_num_words:
    filename = "quwords-training-" + str(num) + "-fr.txt"
    training_words = random.sample(qu_words, num)
    training_f = open(filename, 'w')
    write_to_file(training_f, training_words)

