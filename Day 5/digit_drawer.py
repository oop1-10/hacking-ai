"""
GUI Digit Drawing and Recognition Application
Allows users to draw digits and get predictions from the trained MNIST network.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import os
import sys
from network.network import Network


class DigitDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Recognition - Draw and Predict")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        
        # Initialize variables
        self.canvas_size = 280
        self.brush_size = 12
        self.model_loaded = False
        self.network = None
        self.drawing = False
        
        # Create GUI components
        self.setup_ui()
        
        # Try to load the trained model
        self.load_model()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="MNIST Digit Recognition", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Instructions
        instructions = ttk.Label(main_frame, 
                                text="Draw a digit (0-9) in the box below, then click 'Predict' to see what the AI thinks it is!",
                                font=("Arial", 10))
        instructions.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        
        # Left frame for canvas and controls
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=2, column=0, padx=(0, 20), sticky=(tk.W, tk.N))
        
        # Canvas for drawing
        canvas_label = ttk.Label(left_frame, text="Drawing Area:", font=("Arial", 12, "bold"))
        canvas_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.canvas = tk.Canvas(left_frame, width=self.canvas_size, height=self.canvas_size, 
                               bg='black', cursor='pencil')
        self.canvas.grid(row=1, column=0, pady=(0, 10))
        
        # Bind mouse events for drawing
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
        # Control buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=2, column=0, pady=(0, 10))
        
        self.clear_button = ttk.Button(button_frame, text="Clear Canvas", 
                                      command=self.clear_canvas)
        self.clear_button.grid(row=0, column=0, padx=(0, 5))
        
        self.predict_button = ttk.Button(button_frame, text="Predict Digit", 
                                       command=self.predict_digit)
        self.predict_button.grid(row=0, column=1, padx=(0, 5))
        
        self.save_button = ttk.Button(button_frame, text="Save Image", 
                                     command=self.save_image)
        self.save_button.grid(row=0, column=2, padx=(0, 5))
        
        self.debug_button = ttk.Button(button_frame, text="Show Processing", 
                                      command=self.show_preprocessing)
        self.debug_button.grid(row=0, column=3)
        
        # Brush size control
        brush_frame = ttk.Frame(left_frame)
        brush_frame.grid(row=3, column=0, pady=(0, 10))
        
        ttk.Label(brush_frame, text="Brush Size:").grid(row=0, column=0, padx=(0, 5))
        self.brush_scale = tk.Scale(brush_frame, from_=5, to=20, orient=tk.HORIZONTAL,
                                   command=self.update_brush_size)
        self.brush_scale.set(self.brush_size)
        self.brush_scale.grid(row=0, column=1)
        
        # Right frame for predictions
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=2, column=1, sticky=(tk.W, tk.N))
        
        # Prediction result
        pred_label = ttk.Label(right_frame, text="Prediction Results:", 
                              font=("Arial", 12, "bold"))
        pred_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Main prediction display
        self.prediction_frame = ttk.LabelFrame(right_frame, text="Predicted Digit", padding="10")
        self.prediction_frame.grid(row=1, column=0, pady=(0, 10), sticky=(tk.W, tk.E))
        
        self.prediction_label = ttk.Label(self.prediction_frame, text="?", 
                                         font=("Arial", 48, "bold"), foreground="blue")
        self.prediction_label.grid(row=0, column=0)
        
        self.confidence_label = ttk.Label(self.prediction_frame, text="Draw a digit to predict", 
                                         font=("Arial", 10))
        self.confidence_label.grid(row=1, column=0)
        
        # Confidence scores for all digits
        self.scores_frame = ttk.LabelFrame(right_frame, text="Confidence Scores", padding="10")
        self.scores_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        # Create labels for each digit's confidence
        self.score_labels = {}
        for i in range(10):
            row = i // 5
            col = i % 5
            digit_frame = ttk.Frame(self.scores_frame)
            digit_frame.grid(row=row, column=col, padx=5, pady=2, sticky=tk.W)
            
            digit_label = ttk.Label(digit_frame, text=f"{i}:", font=("Arial", 10, "bold"))
            digit_label.grid(row=0, column=0)
            
            score_label = ttk.Label(digit_frame, text="0.0%", font=("Arial", 10))
            score_label.grid(row=0, column=1, padx=(5, 0))
            
            self.score_labels[i] = score_label
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready to draw!", 
                                     font=("Arial", 9), foreground="green")
        self.status_label.grid(row=3, column=0, columnspan=2, pady=(10, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
    
    def load_model(self):
        """Load the trained MNIST model."""
        # Prioritize the newly trained model over demo model
        model_paths = ["models/mnist_model_final.pkl", "models/demo_model.pkl"]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    self.network = Network(0.001, 784, 10)
                    self.network.load_model(model_path)
                    self.model_loaded = True
                    self.status_label.config(text=f"Model loaded: {model_path}", foreground="green")
                    print(f"Successfully loaded model from {model_path}")
                    return
                except Exception as e:
                    print(f"Error loading model from {model_path}: {e}")
                    continue
        
        # No model found
        self.model_loaded = False
        self.status_label.config(text="No trained model found! Please train a model first.", 
                               foreground="red")
        self.predict_button.config(state="disabled")
        
        messagebox.showwarning("Model Not Found", 
                             "No trained MNIST model found.\n\n"
                             "Please run 'python mnist_trainer.py' or 'python quick_demo.py' "
                             "first to create a trained model.")
    
    def start_drawing(self, event):
        """Start drawing when mouse is pressed."""
        self.drawing = True
        self.draw(event)
    
    def draw(self, event):
        """Draw on the canvas."""
        if self.drawing:
            x, y = event.x, event.y
            radius = self.brush_size // 2
            
            # Draw white circle (digit)
            self.canvas.create_oval(x - radius, y - radius, 
                                  x + radius, y + radius, 
                                  fill='white', outline='white')
    
    def stop_drawing(self, event):
        """Stop drawing when mouse is released."""
        self.drawing = False
    
    def clear_canvas(self):
        """Clear the drawing canvas."""
        self.canvas.delete("all")
        self.prediction_label.config(text="?")
        self.confidence_label.config(text="Draw a digit to predict")
        
        # Reset all confidence scores
        for i in range(10):
            self.score_labels[i].config(text="0.0%", foreground="black")
        
        self.status_label.config(text="Canvas cleared. Ready to draw!", foreground="green")
    
    def update_brush_size(self, value):
        """Update brush size."""
        self.brush_size = int(value)
    
    def canvas_to_array(self):
        """Convert canvas drawing to 28x28 numpy array for the network."""
        # Create a PIL image from the canvas
        self.canvas.update()
        
        # Create PIL image and draw the canvas content
        img = Image.new('RGB', (self.canvas_size, self.canvas_size), 'black')
        draw = ImageDraw.Draw(img)
        
        # Get all canvas items and recreate the drawing
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if len(coords) == 4:  # oval
                draw.ellipse(coords, fill='white')
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Apply MNIST-style preprocessing
        try:
            img_array = self.mnist_style_preprocess(img)
            # Get a simple 28x28 version for saving
            simple_img = img.resize((28, 28), Image.Resampling.LANCZOS)
            return img_array, simple_img
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            # Fallback to basic preprocessing
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = img_array.reshape(1, 784)
            return img_array, img
    
    def mnist_style_preprocess(self, pil_image):
        """
        Preprocess image to match MNIST training data format.
        MNIST has black digits on white background, normalized to [0,1].
        """
        # Convert to numpy array
        img_array = np.array(pil_image)
        
        # Find bounding box of the drawn content (white pixels)
        rows = np.any(img_array > 0, axis=1)
        cols = np.any(img_array > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # No content found, return zeros
            return np.zeros((1, 784), dtype=np.float32)
        
        # Get bounding box coordinates
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Extract the digit region
        digit_region = img_array[rmin:rmax+1, cmin:cmax+1]
        
        # Calculate aspect ratio and size for centering
        height, width = digit_region.shape
        
        # Determine the size for the square bounding box
        max_dim = max(height, width)
        
        # Add some padding (about 10% of the max dimension)
        padding = max(int(max_dim * 0.1), 2)
        canvas_size = max_dim + 2 * padding
        
        # Create a black canvas (like MNIST background)
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
        
        # Calculate position to center the digit
        y_offset = (canvas_size - height) // 2
        x_offset = (canvas_size - width) // 2
        
        # Place the digit in the center
        canvas[y_offset:y_offset+height, x_offset:x_offset+width] = digit_region
        
        # Convert to PIL for high-quality resize
        canvas_img = Image.fromarray(canvas.astype(np.uint8))
        
        # Resize to 28x28 using high-quality resampling
        resized_img = canvas_img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert back to numpy and normalize
        final_array = np.array(resized_img, dtype=np.float32) / 255.0
        
        # Apply some smoothing to reduce pixelation artifacts
        try:
            from scipy import ndimage
            final_array = ndimage.gaussian_filter(final_array, sigma=0.5)
        except ImportError:
            # If scipy not available, skip smoothing
            pass
        
        # Ensure proper normalization
        if final_array.max() > 0:
            final_array = final_array / final_array.max()
        
        # Flatten for network input
        return final_array.reshape(1, 784)
    
    def predict_digit(self):
        """Make a prediction on the drawn digit."""
        if not self.model_loaded:
            messagebox.showerror("Error", "No model loaded. Please check the model file.")
            return
        
        # Check if anything is drawn
        if len(self.canvas.find_all()) == 0:
            messagebox.showinfo("No Drawing", "Please draw a digit first!")
            return
        
        try:
            # Convert canvas to array
            img_array, pil_img = self.canvas_to_array()
            
            # Get prediction probabilities
            probabilities = self.network.forward(img_array)[0]  # Get first (and only) result
            
            # Get predicted digit
            predicted_digit = np.argmax(probabilities)
            confidence = probabilities[predicted_digit]
            
            # Update main prediction display
            self.prediction_label.config(text=str(predicted_digit))
            self.confidence_label.config(text=f"Confidence: {confidence:.1%}")
            
            # Update all confidence scores
            for i in range(10):
                score_text = f"{probabilities[i]:.1%}"
                color = "red" if i == predicted_digit else "black"
                weight = "bold" if i == predicted_digit else "normal"
                
                self.score_labels[i].config(text=score_text, foreground=color)
                # Make the predicted digit's score bold
                current_font = self.score_labels[i].cget("font")
                if i == predicted_digit:
                    self.score_labels[i].config(font=("Arial", 10, "bold"))
                else:
                    self.score_labels[i].config(font=("Arial", 10, "normal"))
            
            self.status_label.config(text=f"Prediction: {predicted_digit} ({confidence:.1%} confidence)", 
                                   foreground="blue")
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error making prediction: {str(e)}")
            print(f"Prediction error: {e}")
    
    def save_image(self):
        """Save the drawn image."""
        if len(self.canvas.find_all()) == 0:
            messagebox.showinfo("No Drawing", "Please draw something first!")
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs("saved_drawings", exist_ok=True)
            
            # Convert canvas to image
            img_array, pil_img = self.canvas_to_array()
            
            # Save the 28x28 processed image
            import time
            timestamp = int(time.time())
            filename = f"saved_drawings/digit_{timestamp}.png"
            
            # Scale up for better visibility when saved
            save_img = pil_img.resize((280, 280), Image.Resampling.NEAREST)
            save_img.save(filename)
            
            messagebox.showinfo("Image Saved", f"Image saved as {filename}")
            self.status_label.config(text=f"Image saved: {filename}", foreground="green")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving image: {str(e)}")
    
    def show_preprocessing(self):
        """Show the preprocessing steps for the current drawing."""
        if len(self.canvas.find_all()) == 0:
            messagebox.showinfo("No Drawing", "Please draw something first!")
            return
        
        try:
            # Create the drawing image
            img = Image.new('RGB', (self.canvas_size, self.canvas_size), 'black')
            draw = ImageDraw.Draw(img)
            
            # Recreate the drawing
            for item in self.canvas.find_all():
                coords = self.canvas.coords(item)
                if len(coords) == 4:  # oval
                    draw.ellipse(coords, fill='white')
            
            # Convert to grayscale
            img = img.convert('L')
            
            # Create directories
            os.makedirs("debug_images", exist_ok=True)
            
            # Generate preprocessing visualization
            import time
            timestamp = int(time.time())
            save_path = f"debug_images/preprocessing_steps_{timestamp}.png"
            
            try:
                from image_preprocessor import ImagePreprocessor
                steps = ImagePreprocessor.visualize_preprocessing_steps(img, save_path)
                
                messagebox.showinfo("Preprocessing Visualization", 
                                  f"Preprocessing steps saved to:\n{save_path}\n\n"
                                  f"Steps shown: {', '.join(steps.keys())}")
                
                self.status_label.config(text=f"Preprocessing visualization saved: {save_path}", 
                                       foreground="green")
            except ImportError:
                messagebox.showinfo("Debug Info", "Advanced preprocessing visualization not available.")
            
        except Exception as e:
            messagebox.showerror("Visualization Error", f"Error creating visualization: {str(e)}")
            print(f"Visualization error: {e}")


def main():
    """Main function to run the application."""
    # Check if required files exist
    required_files = [
        "network/network.py",
        "network/mnist_loader.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure all network files are present.")
        return
    
    # Create and run the GUI application
    root = tk.Tk()
    app = DigitDrawer(root)
    
    # Handle window closing
    def on_closing():
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the GUI event loop
    print("Starting MNIST Digit Recognition GUI...")
    print("Draw digits in the application window and click 'Predict' to see the results!")
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
    except Exception as e:
        print(f"Application error: {e}")


if __name__ == "__main__":
    main()