import torch
from torch import nn
import numpy as np

# read input from simple text file
data = open('input.txt', 'r').read()

# get the list of unique characters
chars = list(set(data))

data_size, vocab_size = len(data), len(chars)
print('data has', data_size, 'characters,', vocab_size, 'unique')

# split input text into strings
sample_text = data.split('\n')

# a dictionary that maps characters to ints
char_to_idx = {"a": 1, "c": 2, "d": 3, "e": 4, "g": 5, "h": 6, "i": 7, "k": 8, "l": 9, "n": 10, "o": 11, "p": 12,
               "r": 13, "s": 14, "t": 15, "v": 16, "w": 17, "y": 18, ' ': 19}

# a dictionary that maps the ints back to characters
idx_to_char = {1: "a", 2: "c", 3: "d", 4: "e", 5: "g", 6: "h", 7: "i", 8: "k", 9: "l", 10: "n", 11: "o", 12: "p",
               13: "r", 14: "s", 15: "t", 16: "v", 17: "w", 18: "y", 19: ' '}

# find the length of the longest string in the sample text
max_len = len(max(sample_text, key=len))

# add padding to sequences so that they are all the same length as max_len
for i in range(len(sample_text)):
    while len(sample_text[i]) < max_len:
        sample_text[i] += ' '

# lists for input and output sequences
input_seq = []
output_seq = []

for i in range(len(sample_text)):
    input_seq.append(sample_text[i])
    output_seq.append(sample_text[i])

# convert chars to ints
for i in range(len(sample_text)):
    input_seq[i] = [char_to_idx[char] for char in input_seq[i]]
    output_seq[i] = [char_to_idx[char] for char in output_seq[i]]

# convert lists to Tensors
input_array = np.array(input_seq)
input_tensor = torch.from_numpy(input_array).float()
# print(input_array.shape)
# print(input_tensor.shape)

output_array = np.array(output_seq)
output_tensor = torch.Tensor(output_array)
# print(output_tensor)
# print(output_array)

# check that torch.cuda is available
is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # define parameters
        self.hidden_dim = hidden_dim  # hidden dimensions
        self.n_layers = n_layers  # hidden layers

        # define layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)  # RNN layer
        self.fc = nn.Linear(hidden_dim, 19)  # Fully connected layer - converts RNN output to desired output shape

    def forward(self, x):
        batch_size = x.size(0)
        # initialize hidden state
        hidden = self.init_hidden(batch_size)

        outputs = []
        for i in range(25):

            # print("x: ", x[:,:,i].shape)
            # pass in input and hidden state into model to get output for first layer
            out, hidden = self.rnn(x[:,:,i].reshape(3, 1, 1), hidden)

            # reshape the output to fit into fully connected layer
            # print("out ", out.shape)
            out = out.contiguous().view(-1, self.hidden_dim)
            # print("out ", out.shape)
            out = self.fc(out)
            # print("out ", out.shape)
            outputs.append(out.unsqueeze(dim=0))

        return torch.cat(outputs, dim=0).permute(1, 0, 2), hidden

    def init_hidden(self, batch_size):
        # creates first hidden state of zeros
        hidden = torch.zeros(self.n_layers, 3, self.hidden_dim)
        return hidden


# prints, for each of the 3 sentences, which sentence the model believes is the most likely to occur
# argument is a tensor of shape (3, 25, 19)
def get_output(prob_output):
    # print(prob_output)
    # print(prob_output[0])
    # print(max(prob_output[0]))
    sentence = ""
    for j in range(3):
        for k in range(25):
            # gets the index of the highest value among the 19 possible choices
            # (starts at 0, so with our dict we need to add 1)
            char_index = prob_output[j][k].tolist().index(max(prob_output[j][k]))
            sentence += idx_to_char[char_index + 1]
            # once we get to the end of the sentence, print what the model thinks is the most likely sentence
            if k == 24:
                print("input:  ", sample_text[j])
                print("output: ", str(sentence).strip('[]'))
                sentence = ""


dict_size = len(char_to_idx)  # number of unique chars in sample_text

# instantiate the model with hyperparameters
model = Model(input_size=1, output_size=dict_size, hidden_dim=12, n_layers=1)
# set the model to the device that we defined earlier (default is CPU)
model.to(device)

batch_size = len(sample_text)
# define hyperparameters
n_epochs = 200
lr = 0.01

# define loss, optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# training
input_tensor = input_tensor.to(device)
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad()  # clears gradients
    # print(input_tensor.unsqueeze(dim=0).shape)
    output, hidden = model(input_tensor.unsqueeze(dim=0))
    # output = output.to(device)
    # output_tensor = output_tensor.to(device)
    # print(output.shape)
    # print(output_tensor.shape)
    # print(output_tensor)
    loss = criterion(output.reshape(3*25,19), output_tensor.view(-1).long() - 1)
    loss.backward()  # does backprop and calculates gradients

    optimizer.step()  # updates the weights accordingly

    if epoch % 10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
    if epoch % 50 == 0:
        get_output(output)
