data = open('qwords.txt', 'r').read()
q_words = data.split('\n')

for i in range(len(q_words)):
    if not q_words[i].__contains__("qu"):
        print(q_words[i])

