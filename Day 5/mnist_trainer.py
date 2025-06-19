import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from network.network import Network
from network.mnist_loader import load_mnist_data, get_batch


def train_mnist_network(epochs=10, batch_size=32, learning_rate=0.001):
    """Train the neural network on MNIST dataset."""
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
    print("MNIST Digit Recognition Neural Network")
    print("=" * 50)
    
    # Check if we should load existing model or train new one
    model_path = "models/mnist_model.pkl"
    
    if len(sys.argv) > 1 and sys.argv[1] == "--load":
        if os.path.exists(model_path):
            load_and_test_model(model_path)
        else:
            print(f"Model file {model_path} not found. Training new model...")
            train_mnist_network()
    else:
        # Train new model with more epochs for better performance
        network, accuracy = train_mnist_network(epochs=15, batch_size=64, learning_rate=0.001)
        print(f"\nTraining completed! Final accuracy: {accuracy:.4f}")
        print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()