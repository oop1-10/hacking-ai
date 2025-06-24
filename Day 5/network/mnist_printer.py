import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


class MNISTPredictor(object):
    """Class to visualize MNIST predictions and display results."""
    
    def __init__(self):
        self.digit_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    def show_predictions(self, images, network, save_images=True):
        """Display MNIST images with their predictions."""
        num_images = min(len(images), 10)  # Show up to 10 images
        
        for i in range(num_images):
            image = images[i]
            
            # Get prediction
            prediction_probs = network.forward(image.reshape(1, -1))
            predicted_digit = np.argmax(prediction_probs)
            confidence = np.max(prediction_probs)
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            # Show image
            ax1.imshow(image.reshape(28, 28), cmap='gray')
            ax1.set_title(f'Input Image\nPredicted: {predicted_digit} (Confidence: {confidence:.3f})')
            ax1.axis('off')
            
            # Show probability distribution
            bars = ax2.bar(range(10), prediction_probs[0], alpha=0.7)
            bars[predicted_digit].set_color('red')  # Highlight predicted digit
            ax2.set_xlabel('Digit')
            ax2.set_ylabel('Probability')
            ax2.set_title('Prediction Probabilities')
            ax2.set_xticks(range(10))
            ax2.set_xticklabels(self.digit_names)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_images:
                plt.savefig(f'predictions/prediction_{i+1}.png', dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def create_confusion_matrix(self, true_labels, predicted_labels):
        """Create and display confusion matrix."""
        from collections import defaultdict
        
        # Create confusion matrix
        confusion = np.zeros((10, 10), dtype=int)
        for true, pred in zip(true_labels, predicted_labels):
            confusion[true, pred] += 1
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add labels
        tick_marks = np.arange(10)
        plt.xticks(tick_marks, self.digit_names)
        plt.yticks(tick_marks, self.digit_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add text annotations
        thresh = confusion.max() / 2.
        for i in range(10):
            for j in range(10):
                plt.text(j, i, format(confusion[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if confusion[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig('predictions/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return confusion
    
    def analyze_errors(self, images, true_labels, predicted_labels, num_errors=5):
        """Show examples of misclassified digits."""
        # Find misclassified examples
        errors = []
        for i, (true, pred) in enumerate(zip(true_labels, predicted_labels)):
            if true != pred:
                errors.append((i, true, pred))
        
        if len(errors) == 0:
            print("No errors found!")
            return
        
        # Show first few errors
        num_errors = min(num_errors, len(errors))
        plt.figure(figsize=(15, 3))
        
        for i in range(num_errors):
            idx, true_label, pred_label = errors[i]
            image = images[idx].reshape(28, 28)
            
            plt.subplot(1, num_errors, i + 1)
            plt.imshow(image, cmap='gray')
            plt.title(f'True: {true_label}\nPred: {pred_label}')
            plt.axis('off')
        
        plt.suptitle(f'Misclassified Examples (Total errors: {len(errors)})')
        plt.tight_layout()
        plt.savefig('predictions/error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Total misclassified: {len(errors)} out of {len(true_labels)}")
        print(f"Error rate: {len(errors)/len(true_labels)*100:.2f}%")