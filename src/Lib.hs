
module Lib where


data Network i o a where
  OutputLayer
    :: Vec o a  -- ^ biases
    -> Network o o
  HiddenLayer
    :: Vec i a  -- ^ biases
    -> Matrix '[o, i] a -- ^ weights
    -> Network o x  -- ^ the next layer in the network
    -> Network i x
  InputLayer
    :: Matrix '[i, o] a -- ^ weights
    -> Network o x -- ^ the next layer in the network
    -> Network i x

feedforward
  :: Vec i Float -- ^ x
  -> Vec o Float -- ^ y
  -> Network i o Float -- ^ network of weights and biases
  -> 

        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
