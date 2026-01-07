import torch
import torch.nn as nn

class SpeechRecognitionModel(nn.Module): # nn.Module automatically handles learnable parameters: weights and biases
    def __init__(self, input_dim, hidden_dim, layer_dims, output_dim, num_classes):
        super(SpeechRecognitionModel, self).__init__()
        # ... define your layers (e.g., convolutional, recurrent, linear)
        # generate random weights and biases according to hidden_size and input_size
        # num_classes would seem to be the total number of characters in this case that can be translated to text from speech
        # will need to think about how speech is rendered as numbers and vice versa - ascii?
        self.input_dim = input_dim # amount of inputs
        self.output_dim = output_dim # amount of outputs
        self.hidden_dim = hidden_dim # number of nodes in each hidden layer
        self.layer_dims = layer_dims # number of hidden layers
        self.num_classes = num_classes # vocabulary
        self.rnn_layers = nn.ModuleList()

        # First layer connects input to the first hidden layer size
        self.rnn_layers.append(nn.RNN(input_dim, layer_dims[0], batch_first=True), nonlinearity='relu')
         
        # Subsequent layers connect the output of the previous layer to the next hidden layer size
        for i in range(1, len(hidden_dim)):
            self.rnn_layers.append(nn.RNN(layer_dims[i-1], layer_dims[i], batch_first=True, nonlinearity='relu'))
            
        # Output linear layer
        self.fc = nn.Linear(layer_dims[-1], output_dim)

        pass
    def forward(self, input):
        # ... define the forward pass of your network
        # forward propogation -> matrix multiplication of weights and biases against input, hidden layers to get output layer
        # Initialize hidden state for each RNN layer
        # In pytorch, you can't perform operations between tensors that live on different devices
        # -> .to(input.device) prevents this
        # device refers to specific processor in computer
        # input.size(0) - takes first dimension of input - batch size
        # batch -> smaller subset of training data fed to model; batch size -> number of training samples in each batch
        hiddens = [torch.zeros(1, input.size(0), hs).to(input.device) for hs in self.layer_dims]
        
        # Pass input through each RNN layer sequentially
        output = input
        for i, rnn_layer in enumerate(self.rnn_layers):
            output, hiddens[i] = rnn_layer(output, hiddens[i]) # output of each layer becomes input to the next layer
            
        # Use the hidden state of the last layer for the output
        output = self.fc(hiddens[-1].squeeze(0)) # Remove the layer dimension

        return output