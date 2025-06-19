import numpy as np
import pickle
import os


class Network(object):
  def __init__(self, learning_rate, input_size, output_size):
    self.learning_rate = learning_rate
    self.input_size = input_size
    self.output_size = output_size
    
    # Initialize network architecture for MNIST
    hidden_size1 = 128  # First hidden layer
    hidden_size2 = 64   # Second hidden layer
    
    # Initialize weights with Xavier/Glorot initialization
    self.weights1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2.0 / input_size)
    self.weights2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2.0 / hidden_size1)
    self.weights3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2.0 / hidden_size2)
    
    # Initialize biases
    self.bias1 = np.zeros((1, hidden_size1))
    self.bias2 = np.zeros((1, hidden_size2))
    self.bias3 = np.zeros((1, output_size))


  def forward(self, X):
    # Ensure X is 2D
    if len(X.shape) == 1:
      X = X.reshape(1, -1)
    
    # First hidden layer
    self.z1 = np.dot(X, self.weights1) + self.bias1
    self.a1 = self.activation(self.z1)
    
    # Second hidden layer
    self.z2 = np.dot(self.a1, self.weights2) + self.bias2
    self.a2 = self.activation(self.z2)
    
    # Output layer
    self.z3 = np.dot(self.a2, self.weights3) + self.bias3
    output = self.softmax(self.z3)
    
    return output


  def activation(self, s):
    # ReLU activation function
    return np.maximum(0, s)


  def activation_derivative(self, s):
    # ReLU derivative
    return (s > 0).astype(float)


  def softmax(self, s):
    # Numerical stability improvement
    exp_s = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)


  def cross_entropy_loss(self, y_true, y_pred):
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


  def backward(self, X, y_true, y_pred):
    m = X.shape[0]  # batch size
    
    # Output layer gradients
    dz3 = y_pred - y_true
    dw3 = (1/m) * np.dot(self.a2.T, dz3)
    db3 = (1/m) * np.sum(dz3, axis=0, keepdims=True)
    
    # Second hidden layer gradients
    da2 = np.dot(dz3, self.weights3.T)
    dz2 = da2 * self.activation_derivative(self.z2)
    dw2 = (1/m) * np.dot(self.a1.T, dz2)
    db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
    
    # First hidden layer gradients
    da1 = np.dot(dz2, self.weights2.T)
    dz1 = da1 * self.activation_derivative(self.z1)
    dw1 = (1/m) * np.dot(X.T, dz1)
    db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
    
    # Update weights and biases
    self.weights3 -= self.learning_rate * dw3
    self.bias3 -= self.learning_rate * db3
    self.weights2 -= self.learning_rate * dw2
    self.bias2 -= self.learning_rate * db2
    self.weights1 -= self.learning_rate * dw1
    self.bias1 -= self.learning_rate * db1


  def train_batch(self, X, y):
    # Forward pass
    output = self.forward(X)
    
    # Backward pass
    self.backward(X, y, output)
    
    # Calculate loss and accuracy
    loss = self.cross_entropy_loss(y, output)
    accuracy = self.calculate_accuracy(y, output)
    
    return loss, accuracy


  def calculate_accuracy(self, y_true, y_pred):
    predictions = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    return np.mean(predictions == true_labels)


  def predict(self, X):
    output = self.forward(X)
    return np.argmax(output, axis=1)


  def guess(self, X):
    # Legacy method for compatibility
    return self.forward(X)


  def save_model(self, filepath):
    """Save the trained model to a file."""
    model_data = {
        'weights1': self.weights1,
        'weights2': self.weights2,
        'weights3': self.weights3,
        'bias1': self.bias1,
        'bias2': self.bias2,
        'bias3': self.bias3,
        'learning_rate': self.learning_rate,
        'input_size': self.input_size,
        'output_size': self.output_size
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {filepath}")


  def load_model(self, filepath):
    """Load a trained model from a file."""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    self.weights1 = model_data['weights1']
    self.weights2 = model_data['weights2']
    self.weights3 = model_data['weights3']
    self.bias1 = model_data['bias1']
    self.bias2 = model_data['bias2']
    self.bias3 = model_data['bias3']
    self.learning_rate = model_data['learning_rate']
    self.input_size = model_data['input_size']
    self.output_size = model_data['output_size']
    
    print(f"Model loaded from {filepath}")