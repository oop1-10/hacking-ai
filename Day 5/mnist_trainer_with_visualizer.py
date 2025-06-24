import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import threading
import time
from network.network import Network
from network.mnist_loader import load_mnist_data, get_batch
from neural_network_visualizer import NeuralNetworkVisualizer


def train_mnist_network_with_visualization(epochs=10, batch_size=32, learning_rate=0.001):
    """Train the neural network on MNIST dataset with real-time visualization."""
    print("Loading MNIST dataset...")
    (train_images, train_labels_encoded, train_labels), (test_images, test_labels_encoded, test_labels) = load_mnist_data()
    
    # Initialize network
    print("Initializing neural network...")
    network = Network(learning_rate, input_size=784, output_size=10)
    
    # Initialize visualizer - must be in main thread
    print("Setting up real-time visualization...")
    visualizer = NeuralNetworkVisualizer(network)
    visualizer.setup_visualization()
    
    # Initialize visualizer data structures
    visualizer.training_losses = []
    visualizer.training_accuracies = []
    visualizer.test_accuracies = []
    visualizer.epochs = []
    visualizer.train_losses = []  # For batch-level updates
    visualizer.train_accuracies = []  # For batch-level updates
    visualizer.epoch_train_losses = []
    visualizer.epoch_train_accuracies = []
    visualizer.epoch_test_accuracies = []
    visualizer.current_input = None
    visualizer.current_input_image = None
    visualizer.current_true_label = None
    visualizer.current_predicted_label = None
    visualizer.current_output_probs = None
    visualizer.current_epoch = 0
    visualizer.current_batch = 0
    visualizer.total_batches = 0
    
    # Initial setup
    try:
        visualizer.update_model_info()
        visualizer.update_weight_heatmaps()
        visualizer.update_weight_distributions()
        plt.pause(0.1)  # Allow plots to render
    except Exception as e:
        print(f"Warning: Initial visualization setup failed: {e}")
    
    # Training metrics
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"\nStarting training for {epochs} epochs with live visualization...")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("-" * 50)
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_accuracies = []
        
        # Shuffle training data
        indices = np.random.permutation(len(train_images))
        train_images_shuffled = train_images[indices]
        train_labels_shuffled = train_labels_encoded[indices]
        
        # Train in batches
        num_batches = len(train_images) // batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            batch_images, batch_labels = get_batch(train_images_shuffled, train_labels_shuffled, batch_size, start_idx)
            
            # Train on batch
            loss, accuracy = network.train_batch(batch_images, batch_labels)
            epoch_losses.append(loss)
            epoch_accuracies.append(accuracy)
            
            # Update visualizer with current sample (show first image of batch)
            if batch_idx % 20 == 0:  # Update visualization every 20 batches
                try:
                    current_image = batch_images[0].reshape(28, 28)
                    current_true_label = np.argmax(batch_labels[0])
                    
                    # Get current predictions and activations
                    output = network.forward(batch_images[0])
                    predicted_label = np.argmax(output)
                    
                    # Update visualizer data
                    visualizer.current_input = current_image
                    visualizer.current_input_image = current_image
                    visualizer.current_true_label = current_true_label
                    visualizer.current_predicted_label = predicted_label
                    visualizer.current_output_probs = output.flatten()
                    visualizer.current_epoch = epoch + 1
                    visualizer.current_batch = batch_idx
                    visualizer.total_batches = num_batches
                    
                    # Update training history for real-time plots
                    visualizer.train_losses.append(loss)
                    visualizer.train_accuracies.append(accuracy)
                    
                    # Keep only last 100 data points for smooth visualization
                    if len(visualizer.train_losses) > 100:
                        visualizer.train_losses = visualizer.train_losses[-100:]
                        visualizer.train_accuracies = visualizer.train_accuracies[-100:]
                    
                    # Update visualizations safely
                    update_visualizations(visualizer, current_image)
                    
                except Exception as e:
                    print(f"Warning: Visualization update failed at batch {batch_idx}: {e}")
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{num_batches}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Calculate average metrics for epoch
        avg_train_loss = np.mean(epoch_losses)
        avg_train_accuracy = np.mean(epoch_accuracies)
        
        # Test accuracy
        test_predictions = network.forward(test_images)
        test_accuracy = network.calculate_accuracy(test_labels_encoded, test_predictions)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        test_accuracies.append(test_accuracy)
        
        # Update visualizer with epoch summary
        try:
            visualizer.epoch_train_losses.append(avg_train_loss)
            visualizer.epoch_train_accuracies.append(avg_train_accuracy)
            visualizer.epoch_test_accuracies.append(test_accuracy)
            
            # Also update the main tracking lists for the visualizer
            visualizer.training_losses.append(avg_train_loss)
            visualizer.training_accuracies.append(avg_train_accuracy)
            visualizer.test_accuracies.append(test_accuracy)
            visualizer.epochs.append(epoch + 1)
            
            # Update weight matrices visualization periodically
            if epoch % 3 == 0:  # Update weights every 3 epochs
                visualizer.update_weight_heatmaps()
                visualizer.update_weight_distributions()
                visualizer.update_training_progress()
                plt.pause(0.01)  # Small pause to allow updates
        except Exception as e:
            print(f"Warning: Epoch visualization update failed: {e}")
        
        print(f"Epoch {epoch+1}/{epochs} Complete:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Train Accuracy: {avg_train_accuracy:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print("-" * 50)
    
    # Save the trained model
    model_path = "models/mnist_model.pkl"
    network.save_model(model_path)
    
    print("\n🎉 Training completed! Visualization will continue running...")
    print("📊 Check the visualization windows for final results")
    print("⚡ Final weight matrices and training curves are displayed")
    
    # Final update to visualizer
    try:
        visualizer.update_weight_heatmaps()
        visualizer.update_weight_distributions()
        visualizer.update_training_progress()
        visualizer.update_model_info()
        plt.pause(0.1)
    except Exception as e:
        print(f"Warning: Final visualization update failed: {e}")
    
    # Plot training progress (traditional plots)
    plot_training_progress(train_losses, train_accuracies, test_accuracies)
    
    # Test final model performance
    final_test_accuracy = evaluate_model(network, test_images, test_labels_encoded, test_labels)
    
    print("\n📈 Traditional training plots have been saved to 'plots/' directory")
    print("🔍 Keep the visualization windows open to explore the final model state!")
    
    return network, final_test_accuracy


def update_visualizations(visualizer, current_input):
    """Safely update all visualizations."""
    try:
        # Update current input display
        if hasattr(visualizer, 'axes') and 'input_image' in visualizer.axes:
            visualizer.axes['input_image'].clear()
            visualizer.axes['input_image'].imshow(visualizer.current_input_image, cmap='gray')
            visualizer.axes['input_image'].set_title(f'Current Input\nTrue: {visualizer.current_true_label}, Pred: {visualizer.current_predicted_label}')
            visualizer.axes['input_image'].axis('off')
        
        # Update output probabilities
        if hasattr(visualizer, 'axes') and 'prediction' in visualizer.axes and visualizer.current_output_probs is not None:
            visualizer.axes['prediction'].clear()
            bars = visualizer.axes['prediction'].bar(range(10), visualizer.current_output_probs)
            visualizer.axes['prediction'].set_title('Output Probabilities')
            visualizer.axes['prediction'].set_xlabel('Digit')
            visualizer.axes['prediction'].set_ylabel('Probability')
            visualizer.axes['prediction'].set_ylim(0, 1)
            
            # Highlight predicted digit
            if visualizer.current_predicted_label is not None:
                bars[visualizer.current_predicted_label].set_color('red')
        
        # Update training progress if we have data
        if len(visualizer.train_losses) > 1:
            # Loss plot
            if 'loss' in visualizer.axes:
                visualizer.axes['loss'].clear()
                visualizer.axes['loss'].plot(visualizer.train_losses[-50:], 'b-', linewidth=2)
                visualizer.axes['loss'].set_title('Recent Training Loss')
                visualizer.axes['loss'].set_ylabel('Loss')
                visualizer.axes['loss'].grid(True, alpha=0.3)
            
            # Accuracy plot
            if 'accuracy' in visualizer.axes:
                visualizer.axes['accuracy'].clear()
                visualizer.axes['accuracy'].plot(visualizer.train_accuracies[-50:], 'g-', linewidth=2)
                visualizer.axes['accuracy'].set_title('Recent Training Accuracy')
                visualizer.axes['accuracy'].set_ylabel('Accuracy')
                visualizer.axes['accuracy'].set_ylim(0, 1)
                visualizer.axes['accuracy'].grid(True, alpha=0.3)
        
        # Update layer activations
        if 'activations' in visualizer.axes and current_input is not None:
            # Get current activations by doing a forward pass
            input_flat = current_input.reshape(1, -1) if len(current_input.shape) > 1 else current_input.reshape(1, -1)
            
            # Forward pass to get activations
            z1 = np.dot(input_flat, visualizer.network.weights1) + visualizer.network.bias1
            a1 = visualizer.network.activation(z1)
            
            z2 = np.dot(a1, visualizer.network.weights2) + visualizer.network.bias2
            a2 = visualizer.network.activation(z2)
            
            z3 = np.dot(a2, visualizer.network.weights3) + visualizer.network.bias3
            a3 = visualizer.network.softmax(z3)
            
            # Plot activations
            visualizer.axes['activations'].clear()
            
            # Create positions for different layers
            input_pos = np.arange(0, 50)  # Show first 50 input neurons
            h1_pos = np.arange(60, 60 + len(a1[0]))  # Hidden layer 1
            h2_pos = np.arange(200, 200 + len(a2[0]))  # Hidden layer 2
            output_pos = np.arange(280, 290)  # Output layer
            
            # Plot each layer's activations
            visualizer.axes['activations'].bar(input_pos, input_flat[0][:50], alpha=0.7, label='Input (first 50)', color='lightblue')
            visualizer.axes['activations'].bar(h1_pos, a1[0], alpha=0.7, label='Hidden1 (128)', color='lightgreen')
            visualizer.axes['activations'].bar(h2_pos, a2[0], alpha=0.7, label='Hidden2 (64)', color='lightyellow')
            visualizer.axes['activations'].bar(output_pos, a3[0], alpha=0.7, label='Output (10)', color='lightcoral')
            
            visualizer.axes['activations'].set_title('Layer Activations (Current Sample)')
            visualizer.axes['activations'].set_xlabel('Neuron Index')
            visualizer.axes['activations'].set_ylabel('Activation Value')
            visualizer.axes['activations'].legend()
            visualizer.axes['activations'].grid(True, alpha=0.3)
        
        # Update model info
        if 'model_info' in visualizer.axes:
            visualizer.axes['model_info'].clear()
            visualizer.axes['model_info'].text(0.1, 0.8, f'Epoch: {visualizer.current_epoch}', fontsize=12, transform=visualizer.axes['model_info'].transAxes)
            visualizer.axes['model_info'].text(0.1, 0.6, f'Batch: {visualizer.current_batch}/{visualizer.total_batches}', fontsize=12, transform=visualizer.axes['model_info'].transAxes)
            if len(visualizer.train_losses) > 0:
                visualizer.axes['model_info'].text(0.1, 0.4, f'Loss: {visualizer.train_losses[-1]:.4f}', fontsize=12, transform=visualizer.axes['model_info'].transAxes)
                visualizer.axes['model_info'].text(0.1, 0.2, f'Accuracy: {visualizer.train_accuracies[-1]:.4f}', fontsize=12, transform=visualizer.axes['model_info'].transAxes)
            visualizer.axes['model_info'].set_title('Training Status')
            visualizer.axes['model_info'].axis('off')
        
        # Force redraw
        if hasattr(visualizer, 'fig1'):
            visualizer.fig1.canvas.draw_idle()
        if hasattr(visualizer, 'fig2'):
            visualizer.fig2.canvas.draw_idle()
        
        plt.pause(0.001)  # Very short pause to allow updates
        
    except Exception as e:
        print(f"Warning: Visualization update error: {e}")


def train_mnist_network(epochs=10, batch_size=32, learning_rate=0.001):
    """Train the neural network on MNIST dataset without visualization."""
    print("Loading MNIST dataset...")
    (train_images, train_labels_encoded, train_labels), (test_images, test_labels_encoded, test_labels) = load_mnist_data()
    
    # Initialize network
    print("Initializing neural network...")
    network = Network(learning_rate, input_size=784, output_size=10)
    
    # Training metrics
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("-" * 50)
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_accuracies = []
        
        # Shuffle training data
        indices = np.random.permutation(len(train_images))
        train_images_shuffled = train_images[indices]
        train_labels_shuffled = train_labels_encoded[indices]
        
        # Train in batches
        num_batches = len(train_images) // batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            batch_images, batch_labels = get_batch(train_images_shuffled, train_labels_shuffled, batch_size, start_idx)
            
            # Train on batch
            loss, accuracy = network.train_batch(batch_images, batch_labels)
            epoch_losses.append(loss)
            epoch_accuracies.append(accuracy)
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{num_batches}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Calculate average metrics for epoch
        avg_train_loss = np.mean(epoch_losses)
        avg_train_accuracy = np.mean(epoch_accuracies)
        
        # Test accuracy
        test_predictions = network.forward(test_images)
        test_accuracy = network.calculate_accuracy(test_labels_encoded, test_predictions)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs} Complete:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Train Accuracy: {avg_train_accuracy:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print("-" * 50)
    
    # Save the trained model
    model_path = "models/mnist_model.pkl"
    network.save_model(model_path)
    
    # Plot training progress
    plot_training_progress(train_losses, train_accuracies, test_accuracies)
    
    # Test final model performance
    final_test_accuracy = evaluate_model(network, test_images, test_labels_encoded, test_labels)
    
    return network, final_test_accuracy


def plot_training_progress(train_losses, train_accuracies, test_accuracies):
    """Plot training metrics."""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot training accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, 'r-', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot test accuracy
    plt.subplot(1, 3, 3)
    plt.plot(epochs, test_accuracies, 'g-', label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/training_progress.png', dpi=300, bbox_inches='tight')
    print("Training progress plot saved to 'plots/training_progress.png'")
    plt.show()


def evaluate_model(network, test_images, test_labels_encoded, test_labels):
    """Evaluate the trained model on test data."""
    print("\nEvaluating final model performance...")
    
    # Get predictions
    predictions = network.predict(test_images)
    
    # Calculate overall accuracy
    accuracy = np.mean(predictions == test_labels)
    print(f"Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Calculate per-class accuracy
    print("\nPer-class accuracy:")
    for digit in range(10):
        digit_mask = test_labels == digit
        digit_accuracy = np.mean(predictions[digit_mask] == test_labels[digit_mask])
        print(f"Digit {digit}: {digit_accuracy:.4f} ({digit_accuracy*100:.2f}%)")
    
    # Show some example predictions
    show_predictions(network, test_images, test_labels, num_examples=10)
    
    return accuracy


def show_predictions(network, test_images, test_labels, num_examples=10):
    """Show some example predictions with images."""
    # Select random examples
    indices = np.random.choice(len(test_images), num_examples, replace=False)
    
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices):
        image = test_images[idx].reshape(28, 28)
        true_label = test_labels[idx]
        predicted_probs = network.forward(test_images[idx])
        predicted_label = np.argmax(predicted_probs)
        confidence = np.max(predicted_probs)
        
        plt.subplot(2, 5, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'True: {true_label}\nPred: {predicted_label} ({confidence:.2f})')
        plt.axis('off')
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/example_predictions.png', dpi=300, bbox_inches='tight')
    print("Example predictions saved to 'plots/example_predictions.png'")
    plt.show()


def load_and_test_model(model_path="models/mnist_model.pkl"):
    """Load a saved model and test it."""
    print(f"Loading model from {model_path}...")
    
    # Load test data
    _, (test_images, test_labels_encoded, test_labels) = load_mnist_data()
    
    # Create network and load model
    network = Network(0.001, 784, 10)  # Parameters will be overwritten by load_model
    network.load_model(model_path)
    
    # Evaluate
    accuracy = evaluate_model(network, test_images, test_labels_encoded, test_labels)
    return network, accuracy


def main():
    """Main function to run MNIST training."""
    print("MNIST Digit Recognition Neural Network with Live Visualization")
    print("=" * 60)
    
    # Check if we should load existing model or train new one
    model_path = "models/mnist_model.pkl"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--load":
            if os.path.exists(model_path):
                load_and_test_model(model_path)
            else:
                print(f"Model file {model_path} not found. Training new model...")
                train_mnist_network_with_visualization()
        elif sys.argv[1] == "--no-viz":
            # Train without visualization
            network, accuracy = train_mnist_network(epochs=15, batch_size=64, learning_rate=0.001)
            print(f"\nTraining completed! Final accuracy: {accuracy:.4f}")
            print(f"Model saved to: {model_path}")
        else:
            print("Usage: python mnist_trainer_with_visualizer.py [--load | --no-viz]")
            print("  --load: Load and test existing model")
            print("  --no-viz: Train without visualization")
            print("  (no args): Train with real-time visualization")
    else:
        # Train new model with real-time visualization
        print("🎯 Starting training with real-time neural network visualization!")
        print("📊 Two windows will open showing network performance and weights")
        print("⚠️  Make sure to keep the visualization windows in focus for best performance")
        network, accuracy = train_mnist_network_with_visualization(epochs=15, batch_size=64, learning_rate=0.001)
        print(f"\nTraining completed! Final accuracy: {accuracy:.4f}")
        print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()