"""
Image preprocessing utilities for the digit drawing application.
Provides advanced preprocessing functions to improve recognition accuracy.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import cv2


class ImagePreprocessor:
    """Advanced image preprocessing for digit recognition."""
    
    @staticmethod
    def preprocess_drawing(pil_image, target_size=(28, 28)):
        """
        Preprocess a drawn digit image for optimal recognition.
        
        Args:
            pil_image: PIL Image object of the drawing
            target_size: Target size for the network (default: 28x28)
            
        Returns:
            numpy array ready for network input
        """
        # Convert to grayscale if not already
        if pil_image.mode != 'L':
            pil_image = pil_image.convert('L')
        
        # Apply Gaussian blur to smooth the image
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Center the digit in the image
        pil_image = ImagePreprocessor._center_digit(pil_image)
        
        # Resize to target size with good quality
        pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(pil_image, dtype=np.float32)
        
        # Normalize to [0, 1] range
        img_array = img_array / 255.0
        
        # Apply contrast enhancement
        img_array = ImagePreprocessor._enhance_contrast(img_array)
        
        # Flatten for network input
        img_array = img_array.reshape(1, -1)
        
        return img_array
    
    @staticmethod
    def _center_digit(pil_image):
        """Center the digit in the image using bounding box detection."""
        # Convert to numpy array for processing
        img_array = np.array(pil_image)
        
        # Find bounding box of the drawn content
        rows = np.any(img_array > 0, axis=1)
        cols = np.any(img_array > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # No content found, return original
            return pil_image
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Extract the digit region
        digit_region = img_array[rmin:rmax+1, cmin:cmax+1]
        
        # Calculate the size of the bounding box
        height, width = digit_region.shape
        
        # Determine the size for centering (preserve aspect ratio)
        max_dim = max(height, width)
        
        # Create a square canvas with some padding
        padding = max_dim // 8  # 12.5% padding
        canvas_size = max_dim + 2 * padding
        
        # Create new canvas
        canvas = np.zeros((canvas_size, canvas_size), dtype=img_array.dtype)
        
        # Calculate position to center the digit
        y_offset = (canvas_size - height) // 2
        x_offset = (canvas_size - width) // 2
        
        # Place the digit in the center
        canvas[y_offset:y_offset+height, x_offset:x_offset+width] = digit_region
        
        # Convert back to PIL Image
        return Image.fromarray(canvas)
    
    @staticmethod
    def _enhance_contrast(img_array):
        """Enhance contrast of the image."""
        # Apply histogram equalization-like enhancement
        # Ensure values are in [0, 1] range
        img_array = np.clip(img_array, 0, 1)
        
        # Apply gamma correction for better contrast
        gamma = 1.2
        img_array = np.power(img_array, 1/gamma)
        
        # Normalize again
        if img_array.max() > 0:
            img_array = img_array / img_array.max()
        
        return img_array
    
    @staticmethod
    def preprocess_with_opencv(pil_image, target_size=(28, 28)):
        """
        Alternative preprocessing using OpenCV (if available).
        
        Args:
            pil_image: PIL Image object
            target_size: Target size tuple
            
        Returns:
            numpy array ready for network input
        """
        try:
            # Convert PIL to OpenCV format
            img_array = np.array(pil_image)
            
            # Apply morphological operations to clean up the image
            kernel = np.ones((2, 2), np.uint8)
            img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
            
            # Find contours to get the digit region
            contours, _ = cv2.findContours(img_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour (should be the digit)
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Extract the digit region
                digit_region = img_array[y:y+h, x:x+w]
                
                # Resize to target size
                digit_region = cv2.resize(digit_region, target_size, interpolation=cv2.INTER_AREA)
                
                # Normalize
                digit_region = digit_region.astype(np.float32) / 255.0
                
                return digit_region.reshape(1, -1)
            else:
                # Fall back to basic preprocessing
                return ImagePreprocessor.preprocess_drawing(pil_image, target_size)
                
        except ImportError:
            # OpenCV not available, use basic preprocessing
            return ImagePreprocessor.preprocess_drawing(pil_image, target_size)
        except Exception:
            # Any other error, fall back to basic preprocessing
            return ImagePreprocessor.preprocess_drawing(pil_image, target_size)
    
    @staticmethod
    def visualize_preprocessing_steps(pil_image, save_path=None):
        """
        Visualize the preprocessing steps for debugging.
        
        Args:
            pil_image: Input PIL Image
            save_path: Optional path to save the visualization
            
        Returns:
            Dictionary with images at each step
        """
        steps = {}
        
        # Original image
        steps['original'] = pil_image.copy()
        
        # Convert to grayscale
        gray_img = pil_image.convert('L') if pil_image.mode != 'L' else pil_image.copy()
        steps['grayscale'] = gray_img
        
        # Apply blur
        blurred = gray_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        steps['blurred'] = blurred
        
        # Center the digit
        centered = ImagePreprocessor._center_digit(blurred)
        steps['centered'] = centered
        
        # Resize to 28x28
        resized = centered.resize((28, 28), Image.Resampling.LANCZOS)
        steps['resized'] = resized
        
        # Apply contrast enhancement
        img_array = np.array(resized, dtype=np.float32) / 255.0
        enhanced_array = ImagePreprocessor._enhance_contrast(img_array)
        enhanced_img = Image.fromarray((enhanced_array * 255).astype(np.uint8))
        steps['enhanced'] = enhanced_img
        
        # Save visualization if requested
        if save_path:
            ImagePreprocessor._save_preprocessing_visualization(steps, save_path)
        
        return steps
    
    @staticmethod
    def _save_preprocessing_visualization(steps, save_path):
        """Save a visualization of preprocessing steps."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            fig.suptitle('Image Preprocessing Steps', fontsize=16)
            
            step_names = ['original', 'grayscale', 'blurred', 'centered', 'resized', 'enhanced']
            
            for i, step_name in enumerate(step_names):
                if step_name in steps:
                    row = i // 3
                    col = i % 3
                    axes[row, col].imshow(steps[step_name], cmap='gray')
                    axes[row, col].set_title(step_name.title())
                    axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for visualization")
        except Exception as e:
            print(f"Error saving visualization: {e}")


def test_preprocessor():
    """Test the image preprocessor with a sample image."""
    try:
        # Create a simple test image
        test_img = Image.new('L', (100, 100), 0)  # Black background
        
        # Draw a simple digit-like shape
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_img)
        draw.ellipse([30, 20, 70, 80], fill=255)  # White oval
        
        print("Testing ImagePreprocessor...")
        
        # Test basic preprocessing
        processed = ImagePreprocessor.preprocess_drawing(test_img)
        print(f"Processed image shape: {processed.shape}")
        print(f"Processed image range: {processed.min():.3f} to {processed.max():.3f}")
        
        # Test visualization
        steps = ImagePreprocessor.visualize_preprocessing_steps(test_img)
        print(f"Preprocessing steps: {list(steps.keys())}")
        
        print("✓ ImagePreprocessor test completed successfully!")
        
    except Exception as e:
        print(f"❌ ImagePreprocessor test failed: {e}")


if __name__ == "__main__":
    test_preprocessor()