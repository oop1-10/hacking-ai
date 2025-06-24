import numpy as np
import gzip
import pickle
import urllib.request
import os


def download_mnist():
    """Download MNIST dataset if it doesn't exist."""
    # Updated URL that should work reliably
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz", 
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    if not os.path.exists("mnist_data"):
        os.makedirs("mnist_data")
    
    for file in files:
        if not os.path.exists(f"mnist_data/{file}"):
            print(f"Downloading {file}...")
            try:
                urllib.request.urlretrieve(base_url + file, f"mnist_data/{file}")
            except Exception as e:
                print(f"Failed to download {file} from {base_url}. Trying alternative...")
                # Fallback to a different mirror
                alt_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
                try:
                    urllib.request.urlretrieve(alt_url + file, f"mnist_data/{file}")
                except Exception as e2:
                    print(f"Failed to download from alternative source. Error: {e2}")
                    print("Please download MNIST data manually or check your internet connection.")
                    raise e2
    
    print("MNIST dataset downloaded successfully!")


def load_mnist_images(filename):
    """Load MNIST images from compressed file."""
    with gzip.open(filename, 'rb') as f:
        # Skip header (magic number and dimensions)
        data = f.read()
        magic = int.from_bytes(data[:4], 'big')
        num_images = int.from_bytes(data[4:8], 'big')
        rows = int.from_bytes(data[8:12], 'big')
        cols = int.from_bytes(data[12:16], 'big')
        
        # Load image data
        images = np.frombuffer(data[16:], dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
        
        # Normalize to 0-1 range and flatten
        images = images.astype(np.float32) / 255.0
        images = images.reshape(num_images, rows * cols)
        
        return images


def load_mnist_labels(filename):
    """Load MNIST labels from compressed file."""
    with gzip.open(filename, 'rb') as f:
        data = f.read()
        magic = int.from_bytes(data[:4], 'big')
        num_labels = int.from_bytes(data[4:8], 'big')
        
        # Load labels
        labels = np.frombuffer(data[8:], dtype=np.uint8)
        return labels


def one_hot_encode(labels, num_classes=10):
    """Convert labels to one-hot encoded format."""
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def load_mnist_data():
    """Load and preprocess MNIST dataset."""
    # Download if needed
    download_mnist()
    
    # Load training data
    train_images = load_mnist_images("mnist_data/train-images-idx3-ubyte.gz")
    train_labels = load_mnist_labels("mnist_data/train-labels-idx1-ubyte.gz")
    
    # Load test data
    test_images = load_mnist_images("mnist_data/t10k-images-idx3-ubyte.gz")
    test_labels = load_mnist_labels("mnist_data/t10k-labels-idx1-ubyte.gz")
    
    # One-hot encode labels
    train_labels_encoded = one_hot_encode(train_labels)
    test_labels_encoded = one_hot_encode(test_labels)
    
    print(f"Training data: {train_images.shape[0]} samples, {train_images.shape[1]} features")
    print(f"Test data: {test_images.shape[0]} samples, {test_images.shape[1]} features")
    
    return (train_images, train_labels_encoded, train_labels), (test_images, test_labels_encoded, test_labels)


def get_batch(X, y, batch_size, start_idx):
    """Get a batch of data for training."""
    end_idx = min(start_idx + batch_size, len(X))
    return X[start_idx:end_idx], y[start_idx:end_idx]