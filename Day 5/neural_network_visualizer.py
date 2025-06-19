import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Force interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
from network.network import Network
from network.mnist_loader import load_mnist_data, get_batch
import time
import threading
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import os

# Configure matplotlib for better display
plt.ion()  # Turn on interactive mode


class NeuralNetworkVisualizer:
    def __init__(self, network=None):
        self.network = network
        self.fig = None
        self.axes = {}
        
        # Training data storage
        self.training_losses = []
        self.training_accuracies = []
        self.test_accuracies = []
        self.epochs = []
        self.current_epoch = 0
        
        # Weight history for visualization
        self.weight_history = {'w1': [], 'w2': [], 'w3': []}
        
        # Current activations for real-time display
        self.current_activations = {'input': None, 'h1': None, 'h2': None, 'output': None}
        self.current_input_image = None
        
        # Animation control
        self.is_training = False
        self.animation = None
        
        # Colors for different layers
        self.layer_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        
    def setup_visualization(self):
        """Set up the complete visualization layout with two separate figures."""
        
        # ===== FIGURE 1: Main Analysis =====
        self.fig1 = plt.figure(figsize=(16, 10))
        self.fig1.suptitle('Neural Network Analysis - Performance & Predictions', fontsize=16, fontweight='bold')
        
        # Create grid layout for figure 1 (2x3 layout)
        gs1 = self.fig1.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Current input image
        self.axes['input_image'] = self.fig1.add_subplot(gs1[0, 0])
        self.axes['input_image'].set_title('Current Input (28×28)', fontweight='bold')
        
        # Training loss
        self.axes['loss'] = self.fig1.add_subplot(gs1[0, 1])
        self.axes['loss'].set_title('Training Loss', fontweight='bold')
        self.axes['loss'].set_xlabel('Epoch')
        self.axes['loss'].set_ylabel('Loss')
        self.axes['loss'].grid(True, alpha=0.3)
        
        # Training accuracy
        self.axes['accuracy'] = self.fig1.add_subplot(gs1[0, 2])
        self.axes['accuracy'].set_title('Training Accuracy', fontweight='bold')
        self.axes['accuracy'].set_xlabel('Epoch')
        self.axes['accuracy'].set_ylabel('Accuracy')
        self.axes['accuracy'].grid(True, alpha=0.3)
        
        # Output probabilities (larger, spanning 2 columns)
        self.axes['prediction'] = self.fig1.add_subplot(gs1[1, 0:2])
        self.axes['prediction'].set_title('Output Probabilities', fontweight='bold')
        self.axes['prediction'].set_xlabel('Digit')
        self.axes['prediction'].set_ylabel('Probability')
        
        # Add a text area for model info
        self.axes['model_info'] = self.fig1.add_subplot(gs1[1, 2])
        self.axes['model_info'].set_title('Model Information', fontweight='bold')
        self.axes['model_info'].axis('off')
        
        # Layer activations (full width)
        self.axes['activations'] = self.fig1.add_subplot(gs1[2, :])
        self.axes['activations'].set_title('Layer Activations (Current Sample)', fontweight='bold')
        self.axes['activations'].set_xlabel('Neuron Index')
        self.axes['activations'].set_ylabel('Activation Value')
        
        # ===== FIGURE 2: Weight Analysis =====
        self.fig2 = plt.figure(figsize=(16, 10))
        self.fig2.suptitle('Neural Network Analysis - Weights & Distributions', fontsize=16, fontweight='bold')
        
        # Create grid layout for figure 2
        gs2 = self.fig2.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Weight matrices heatmaps
        self.axes['weights1'] = self.fig2.add_subplot(gs2[0, 0])
        self.axes['weights1'].set_title('Weights: Input→Hidden1\n(784×128 sampled)', fontweight='bold')
        
        self.axes['weights2'] = self.fig2.add_subplot(gs2[0, 1])
        self.axes['weights2'].set_title('Weights: Hidden1→Hidden2\n(128×64)', fontweight='bold')
        
        self.axes['weights3'] = self.fig2.add_subplot(gs2[1, 0])
        self.axes['weights3'].set_title('Weights: Hidden2→Output\n(64×10)', fontweight='bold')
        
        # Weight distribution
        self.axes['weight_dist'] = self.fig2.add_subplot(gs2[1, 1])
        self.axes['weight_dist'].set_title('Weight Distributions', fontweight='bold')
        
        # Store figures for later reference
        self.fig = self.fig1  # Keep compatibility
        
        try:
            self.fig1.tight_layout()
            self.fig2.tight_layout()
        except:
            pass  # Ignore tight_layout warnings
        

    
    def update_weight_heatmaps(self):
        """Update the weight matrix heatmaps."""
        if self.network is None:
            return
            
        # Weight matrix 1: Input to Hidden1 (show a subset)
        w1_subset = self.network.weights1[::10, ::2]  # Sample every 10th input, every 2nd hidden
        self.axes['weights1'].clear()
        im1 = self.axes['weights1'].imshow(w1_subset.T, cmap='RdBu', aspect='auto', 
                                         vmin=-2, vmax=2, interpolation='nearest')
        self.axes['weights1'].set_title('Weights: Input→Hidden1\n(sampled)', fontweight='bold')
        
        # Weight matrix 2: Hidden1 to Hidden2
        self.axes['weights2'].clear()
        im2 = self.axes['weights2'].imshow(self.network.weights2.T, cmap='RdBu', aspect='auto',
                                         vmin=-2, vmax=2, interpolation='nearest')
        self.axes['weights2'].set_title('Weights: Hidden1→Hidden2\n(128×64)', fontweight='bold')
        
        # Weight matrix 3: Hidden2 to Output
        self.axes['weights3'].clear()
        im3 = self.axes['weights3'].imshow(self.network.weights3.T, cmap='RdBu', aspect='auto',
                                         vmin=-2, vmax=2, interpolation='nearest')
        self.axes['weights3'].set_title('Weights: Hidden2→Output\n(64×10)', fontweight='bold')
    
    def update_training_progress(self):
        """Update training progress plots."""
        if len(self.training_losses) == 0:
            return
            
        # Loss plot
        self.axes['loss'].clear()
        self.axes['loss'].plot(self.epochs, self.training_losses, 'b-', label='Training Loss', linewidth=2)
        self.axes['loss'].set_title('Training Loss', fontweight='bold')
        self.axes['loss'].set_xlabel('Epoch')
        self.axes['loss'].set_ylabel('Loss')
        self.axes['loss'].grid(True, alpha=0.3)
        self.axes['loss'].legend()
        
        # Accuracy plot
        self.axes['accuracy'].clear()
        if len(self.training_accuracies) > 0:
            self.axes['accuracy'].plot(self.epochs, self.training_accuracies, 'g-', 
                                     label='Training Accuracy', linewidth=2)
        if len(self.test_accuracies) > 0:
            self.axes['accuracy'].plot(self.epochs, self.test_accuracies, 'r-', 
                                     label='Test Accuracy', linewidth=2)
        self.axes['accuracy'].set_title('Accuracy', fontweight='bold')
        self.axes['accuracy'].set_xlabel('Epoch')
        self.axes['accuracy'].set_ylabel('Accuracy')
        self.axes['accuracy'].set_ylim(0, 1)
        self.axes['accuracy'].grid(True, alpha=0.3)
        self.axes['accuracy'].legend()
    
    def update_activations(self, input_data):
        """Update the activation displays."""
        if self.network is None or input_data is None:
            return
            
        # Check if axes are set up
        if not hasattr(self, 'axes') or 'input_image' not in self.axes:
            return
            
        # Forward pass to get activations
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
            
        # Store activations during forward pass
        self.current_activations['input'] = input_data[0]
        
        # Forward pass step by step
        z1 = np.dot(input_data, self.network.weights1) + self.network.bias1
        a1 = self.network.activation(z1)
        self.current_activations['h1'] = a1[0]
        
        z2 = np.dot(a1, self.network.weights2) + self.network.bias2
        a2 = self.network.activation(z2)
        self.current_activations['h2'] = a2[0]
        
        z3 = np.dot(a2, self.network.weights3) + self.network.bias3
        output = self.network.softmax(z3)
        self.current_activations['output'] = output[0]
        
        # Update input image
        self.axes['input_image'].clear()
        if self.current_input_image is not None:
            self.axes['input_image'].imshow(self.current_input_image, cmap='gray')
            self.axes['input_image'].set_title('Current Input (28×28)', fontweight='bold')
            self.axes['input_image'].axis('off')
        
        # Update activation plot
        self.axes['activations'].clear()
        layers = ['Input', 'Hidden1', 'Hidden2', 'Output']
        activations = [
            self.current_activations['input'][:50],  # Show first 50 inputs
            self.current_activations['h1'],
            self.current_activations['h2'],
            self.current_activations['output']
        ]
        
        x_offset = 0
        colors = ['blue', 'green', 'orange', 'red']
        for i, (layer, act, color) in enumerate(zip(layers, activations, colors)):
            if act is not None:
                x_range = np.arange(x_offset, x_offset + len(act))
                self.axes['activations'].bar(x_range, act, alpha=0.7, color=color, 
                                           label=f'{layer} ({len(act)})')
                x_offset += len(act) + 5
        
        self.axes['activations'].set_title('Layer Activations (Current Sample)', fontweight='bold')
        self.axes['activations'].set_xlabel('Neuron Index')
        self.axes['activations'].set_ylabel('Activation Value')
        self.axes['activations'].legend()
        
        # Update prediction probabilities
        self.axes['prediction'].clear()
        if self.current_activations['output'] is not None:
            digits = np.arange(10)
            bars = self.axes['prediction'].bar(digits, self.current_activations['output'], 
                                             color='lightcoral', alpha=0.7)
            
            # Highlight the predicted digit
            predicted_digit = np.argmax(self.current_activations['output'])
            bars[predicted_digit].set_color('red')
            
            self.axes['prediction'].set_title(f'Output Probabilities\nPredicted: {predicted_digit}', 
                                            fontweight='bold')
            self.axes['prediction'].set_xlabel('Digit')
            self.axes['prediction'].set_ylabel('Probability')
            self.axes['prediction'].set_ylim(0, 1)
    
    def update_weight_distributions(self):
        """Update weight distribution histogram."""
        if self.network is None:
            return
            
        self.axes['weight_dist'].clear()
        
        # Flatten all weights
        all_weights = np.concatenate([
            self.network.weights1.flatten(),
            self.network.weights2.flatten(),
            self.network.weights3.flatten()
        ])
        
        self.axes['weight_dist'].hist(all_weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        self.axes['weight_dist'].axvline(0, color='red', linestyle='--', alpha=0.8)
        self.axes['weight_dist'].set_title(f'Weight Distributions\nMean: {all_weights.mean():.4f}, Std: {all_weights.std():.4f}', 
                                         fontweight='bold')
        self.axes['weight_dist'].set_xlabel('Weight Value')
        self.axes['weight_dist'].set_ylabel('Frequency')
    
    def update_model_info(self):
        """Update model information display."""
        if self.network is None:
            return
            
        self.axes['model_info'].clear()
        self.axes['model_info'].axis('off')
        
        # Calculate network statistics
        total_params = (self.network.weights1.size + self.network.weights2.size + 
                       self.network.weights3.size + self.network.bias1.size + 
                       self.network.bias2.size + self.network.bias3.size)
        
        info_text = f"""
Architecture: 784→128→64→10
Total Parameters: {total_params:,}
Learning Rate: {self.network.learning_rate:.4f}

Weight Ranges:
W1: [{self.network.weights1.min():.3f}, {self.network.weights1.max():.3f}]
W2: [{self.network.weights2.min():.3f}, {self.network.weights2.max():.3f}]
W3: [{self.network.weights3.min():.3f}, {self.network.weights3.max():.3f}]
"""
        
        self.axes['model_info'].text(0.1, 0.5, info_text, transform=self.axes['model_info'].transAxes,
                                   fontsize=10, verticalalignment='center', fontfamily='monospace')
    

    
    def animate_training(self, frame):
        """Animation function for training visualization."""
        if self.network is not None:
            # Update Figure 1: Performance Analysis
            self.update_training_progress()
            self.update_model_info()
            
            # Update Figure 2: Weight Analysis
            self.update_weight_heatmaps()
            self.update_weight_distributions()
            
            # Redraw both figures
            self.fig1.canvas.draw_idle()
            self.fig2.canvas.draw_idle()
        return []
    
    def start_visualization(self):
        """Start the visualization."""
        self.setup_visualization()
        
        # Start animation
        self.animation = FuncAnimation(self.fig, self.animate_training, interval=1000, blit=False)
        
        plt.show()
    
    def add_training_data(self, epoch, loss, train_acc, test_acc=None):
        """Add training data for visualization."""
        self.epochs.append(epoch)
        self.training_losses.append(loss)
        self.training_accuracies.append(train_acc)
        if test_acc is not None:
            self.test_accuracies.append(test_acc)
    
    def set_current_input(self, image_data):
        """Set the current input image for visualization."""
        if len(image_data.shape) == 1:
            self.current_input_image = image_data.reshape(28, 28)
        else:
            self.current_input_image = image_data
        
        # Update activations with this input (only if axes are set up)
        if hasattr(self, 'axes') and 'input_image' in self.axes:
            self.update_activations(image_data.flatten())


def visualize_existing_model(model_path="models/mnist_model_final.pkl"):
    """Visualize an existing trained model."""
    print("Loading MNIST dataset...")
    (train_images, train_labels_encoded, train_labels), (test_images, test_labels_encoded, test_labels) = load_mnist_data()
    
    # Load model
    network = Network(0.001, 784, 10)
    # Try multiple possible model paths
    possible_paths = [model_path, "models/mnist_model.pkl", "models/demo_model.pkl"]
    model_loaded = False
    
    for path in possible_paths:
        if os.path.exists(path):
            network.load_model(path)
            print(f"Model loaded from {path}")
            model_loaded = True
            break
    
    if not model_loaded:
        print(f"No model found. Tried: {possible_paths}")
        print("Please train a model first.")
        return
    
    # Create visualizer
    visualizer = NeuralNetworkVisualizer(network)
    
    # Set up the visualization first
    visualizer.setup_visualization()
    
    # Add some dummy training data for display
    visualizer.add_training_data(1, 2.5, 0.1, 0.15)
    visualizer.add_training_data(2, 1.8, 0.4, 0.45)
    visualizer.add_training_data(3, 1.2, 0.7, 0.72)
    visualizer.add_training_data(4, 0.8, 0.85, 0.86)
    visualizer.add_training_data(5, 0.6, 0.9, 0.91)
    
    # Set up a demo input after axes are created
    sample_image = test_images[0]
    visualizer.set_current_input(sample_image)
    
    print("Starting visualization...")
    
    # Force initial update
    visualizer.update_training_progress()
    visualizer.update_model_info()
    visualizer.update_weight_heatmaps()
    visualizer.update_weight_distributions()
    
    # Start animation with save_count to avoid warnings
    visualizer.animation1 = FuncAnimation(visualizer.fig1, visualizer.animate_training, 
                                        interval=2000, blit=False, save_count=100, cache_frame_data=False)
    visualizer.animation2 = FuncAnimation(visualizer.fig2, visualizer.animate_training, 
                                        interval=2000, blit=False, save_count=100, cache_frame_data=False)
    
    # Show both plots
    plt.show(block=True)


if __name__ == "__main__":
    visualize_existing_model() 