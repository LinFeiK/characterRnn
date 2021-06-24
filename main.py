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
char_to_idx = {ch: i for i, ch in enumerate(chars)}

# a dictionary that maps the ints back to characters
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# find the length of the longest string in the sample text
max_len = len(max(sample_text, key=len))

sample_text_length = len(sample_text)

# add padding to sequences so that they are all the same length as max_len
for i in range(sample_text_length):
    while len(sample_text[i]) < max_len:
        sample_text[i] += ' '

# lists for input and output sequences
input_seq = []
output_seq = []

for i in range(sample_text_length):
    input_seq.append(sample_text[i])
    output_seq.append(sample_text[i])

# convert chars to ints
for i in range(sample_text_length):
    input_seq[i] = [char_to_idx[char] for char in input_seq[i]]
    output_seq[i] = [char_to_idx[char] for char in output_seq[i]]

# convert lists to Tensors
input_array = np.array(input_seq)
input_tensor = torch.from_numpy(input_array).float()

output_array = np.array(output_seq)
output_tensor = torch.Tensor(output_array)

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
        self.fc = nn.Linear(hidden_dim, vocab_size)  # Fully connected layer - converts RNN output to desired output shape

    def forward(self, x):
        # initialize hidden state
        hidden = self.init_hidden()

        outputs = []
        for i in range(max_len):
            # pass in input and hidden state into model to get output for first layer
            out, hidden = self.rnn(x[:,:,i].reshape(sample_text_length, 1, 1), hidden)

            # reshape the output to fit into fully connected layer
            out = out.contiguous().view(-1, self.hidden_dim)

            out = self.fc(out)

            outputs.append(out.unsqueeze(dim=0))

        return torch.cat(outputs, dim=0).permute(1, 0, 2), hidden

    def init_hidden(self):
        # creates first hidden state of zeros
        hidden = torch.zeros(self.n_layers, sample_text_length, self.hidden_dim)
        return hidden


# prints, for each of the sample_text_length sentences, which sentence the model believes is the most likely to occur
# argument is a tensor of shape (sample_text_length, max_len, vocab_size)
def get_output(prob_output):
    sentence = ""
    for j in range(sample_text_length):
        for k in range(max_len):
            # gets the index of the highest value among the vocab_size possible choices
            char_index = prob_output[j][k].tolist().index(max(prob_output[j][k]))
            sentence += idx_to_char[char_index]
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

batch_size = sample_text_length
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
    output, hidden = model(input_tensor.unsqueeze(dim=0))
    loss = criterion(output.reshape(sample_text_length*max_len, vocab_size), output_tensor.view(-1).long())
    loss.backward()  # does backprop and calculates gradients

    optimizer.step()  # updates the weights accordingly

    if epoch % 10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
    if epoch % 50 == 0:
        get_output(output)
