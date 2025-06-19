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
import json
import threading
import time
import tempfile
import wave
from network.network import Network

# TTS imports
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame not available - TTS functionality disabled")


class DigitDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Recognition - Draw and Predict")
        self.root.geometry("950x600")
        self.root.resizable(False, False)
        
        # Initialize variables
        self.canvas_width = 420  # Wider for multi-digit numbers
        self.canvas_height = 280  # Keep height the same
        self.brush_size = 12
        self.model_loaded = False
        self.network = None
        self.drawing = False
        
        # TTS variables
        self.voice_library = {}
        self.fallback_dir = "fallback"
        self.user_dir = "user"
        self.voice_library_path = "voice_library.json"
        self.tts_enabled = PYGAME_AVAILABLE
        self.last_prediction = ""
        
        # Initialize pygame for TTS
        if self.tts_enabled:
            try:
                pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=512)
                pygame.mixer.init()
                print("TTS audio system initialized")
            except Exception as e:
                print(f"TTS audio initialization failed: {e}")
                self.tts_enabled = False
        
        # Load voice library for TTS
        if self.tts_enabled:
            self.load_voice_library()
        
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
        tts_status = " (with voice output)" if self.tts_enabled else " (no voice - install pygame for TTS)"
        instructions = ttk.Label(main_frame, 
                                text=f"Draw digits or numbers (like 1024) in the box below, then click 'Predict Digits' to recognize each digit{tts_status}",
                                font=("Arial", 10))
        instructions.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        
        # Left frame for canvas and controls
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=2, column=0, padx=(0, 20), sticky=(tk.W, tk.N))
        
        # Canvas for drawing
        canvas_label = ttk.Label(left_frame, text="Drawing Area (420×280):", font=("Arial", 12, "bold"))
        canvas_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.canvas = tk.Canvas(left_frame, width=self.canvas_width, height=self.canvas_height, 
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
        
        self.predict_button = ttk.Button(button_frame, text="Predict Digits", 
                                       command=self.predict_digits)
        self.predict_button.grid(row=0, column=1, padx=(0, 5))
        
        self.save_button = ttk.Button(button_frame, text="Save Image", 
                                     command=self.save_image)
        self.save_button.grid(row=0, column=2, padx=(0, 5))
        
        self.debug_button = ttk.Button(button_frame, text="Show Processing", 
                                      command=self.show_preprocessing)
        self.debug_button.grid(row=0, column=3, padx=(0, 5))
        
        self.single_button = ttk.Button(button_frame, text="Single Digit Mode", 
                                       command=self.predict_digit)
        self.single_button.grid(row=0, column=4)
        
        # TTS button (only show if TTS is available)
        if self.tts_enabled:
            self.tts_button = ttk.Button(button_frame, text="Repeat", 
                                        command=self.repeat_last_prediction)
            self.tts_button.grid(row=0, column=5, padx=(5, 0))
        
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
        self.prediction_frame = ttk.LabelFrame(right_frame, text="Predicted Number", padding="10")
        self.prediction_frame.grid(row=1, column=0, pady=(0, 10), sticky=(tk.W, tk.E))
        
        self.prediction_label = ttk.Label(self.prediction_frame, text="?", 
                                         font=("Arial", 32, "bold"), foreground="blue")
        self.prediction_label.grid(row=0, column=0)
        
        self.confidence_label = ttk.Label(self.prediction_frame, text="Draw digits to predict", 
                                         font=("Arial", 10))
        self.confidence_label.grid(row=1, column=0)
        
        # Individual digit predictions
        self.digits_frame = ttk.LabelFrame(right_frame, text="Individual Digits", padding="10")
        self.digits_frame.grid(row=2, column=0, pady=(0, 10), sticky=(tk.W, tk.E))
        
        self.digit_predictions = []  # Will store individual digit prediction labels
        
        # Confidence scores for all digits
        self.scores_frame = ttk.LabelFrame(right_frame, text="Current Digit Confidence", padding="10")
        self.scores_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
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
        self.status_label = ttk.Label(main_frame, text="Ready to draw numbers!", 
                                     font=("Arial", 9), foreground="green")
        self.status_label.grid(row=3, column=0, columnspan=2, pady=(10, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
    
    def load_voice_library(self):
        """Load voice library for TTS (fallback first, then user recordings)."""
        try:
            # Load fallback WAV files first
            self.load_fallback_voices()
            
            # Load user recordings (these take priority)
            self.load_user_voices()
            
            voice_count = len(self.voice_library)
            fallback_count = len([v for v in self.voice_library.values() if v.get('is_fallback')])
            user_count = voice_count - fallback_count
            
            print(f"TTS Voice Library loaded: {user_count} user voices, {fallback_count} fallback voices")
            
        except Exception as e:
            print(f"Error loading voice library: {e}")
            self.tts_enabled = False
    
    def load_fallback_voices(self):
        """Load WAV files from fallback directory."""
        if not os.path.exists(self.fallback_dir):
            print("No fallback voice directory found")
            return
        
        # Basic number words we need
        number_words = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", 
            "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", 
            "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million"
        ]
        
        loaded_count = 0
        for word in number_words:
            wav_file = os.path.join(self.fallback_dir, f"{word}.wav")
            if os.path.exists(wav_file):
                try:
                    # Load WAV file info
                    with wave.open(wav_file, 'rb') as wav_info:
                        frames = wav_info.readframes(-1)
                        sample_rate = wav_info.getframerate()
                        duration = len(frames) / (wav_info.getnframes() / wav_info.getframerate()) if wav_info.getnframes() > 0 else 1.0
                    
                    self.voice_library[word] = {
                        'wav_file': wav_file,
                        'is_fallback': True,
                        'duration': duration,
                        'sample_rate': sample_rate
                    }
                    loaded_count += 1
                except Exception as e:
                    print(f"Error loading fallback WAV {wav_file}: {e}")
        
        print(f"  Loaded {loaded_count} fallback WAV voices")
    
    def load_user_voices(self):
        """Load user recorded voices (takes priority over fallback)."""
        if not os.path.exists(self.voice_library_path):
            print("No user voice library found")
            return
        
        try:
            with open(self.voice_library_path, 'r') as f:
                metadata = json.load(f)
            
            loaded_count = 0
            for word, meta in metadata.items():
                audio_file = f"{self.user_dir}/{word}.wav"
                
                if os.path.exists(audio_file):
                    try:
                        with wave.open(audio_file, 'rb') as wav_file:
                            frames = wav_file.readframes(-1)
                            audio_data = np.frombuffer(frames, dtype=np.int16)
                        
                        # Override fallback with user recording
                        self.voice_library[word] = {
                            'audio_data': audio_data.tolist(),
                            'sample_rate': meta['sample_rate'],
                            'duration': meta['duration'],
                            'recorded_at': meta['recorded_at'],
                            'is_fallback': False
                        }
                        loaded_count += 1
                    except Exception as e:
                        print(f"Could not load user voice {audio_file}: {e}")
            
            print(f"  Loaded {loaded_count} user recorded voices")
            
        except Exception as e:
            print(f"Error loading user voices: {e}")
    
    def number_to_words(self, number):
        """Convert number to words using basic individual words."""
        if number == 0:
            return ['zero']
        
        # Basic number words
        ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 
                'seventeen', 'eighteen', 'nineteen']
        
        tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
        
        words = []
        
        # Handle numbers up to 999,999
        if number >= 1000000:
            millions = number // 1000000
            words.extend(self.number_to_words(millions))
            words.append('million')
            number %= 1000000
        
        if number >= 1000:
            thousands = number // 1000
            words.extend(self.number_to_words(thousands))
            words.append('thousand')
            number %= 1000
        
        if number >= 100:
            hundreds = number // 100
            words.append(ones[hundreds])
            words.append('hundred')
            number %= 100
        
        if number >= 20:
            tens_digit = number // 10
            ones_digit = number % 10
            words.append(tens[tens_digit])
            if ones_digit > 0:
                words.append(ones[ones_digit])
        elif number > 0:
            words.append(ones[number])
        
        return words
    
    def speak_number(self, number_str):
        """Convert number string to speech and play it."""
        if not self.tts_enabled:
            print("TTS not available")
            return
        
        try:
            # Convert to integer and then to words
            number = int(number_str)
            words = self.number_to_words(number)
            
            print(f"Speaking: {number_str} -> {' '.join(words)}")
            
            # Check which words we have
            available_words = []
            missing_words = []
            
            for word in words:
                if word in self.voice_library:
                    available_words.append(word)
                else:
                    missing_words.append(word)
            
            if missing_words:
                print(f"Missing voice files for: {', '.join(missing_words)}")
                # Still try to speak available words
                words = available_words
            
            print(f"Available words to speak: {available_words}")
            print(f"Will attempt to play: {words}")
            
            if not words:
                print("No available voice files to speak this number")
                return
            
            # Combine and play audio
            threading.Thread(target=self._play_words_audio, args=(words,), daemon=True).start()
            
        except ValueError:
            print(f"Invalid number format: {number_str}")
        except Exception as e:
            print(f"Error in TTS: {e}")
    
    def _play_words_audio(self, words):
        """Play audio for a sequence of words."""
        try:
            print(f"Starting to play {len(words)} words: {words}")
            for i, word in enumerate(words):
                print(f"Playing word {i+1}/{len(words)}: '{word}'")
                if word in self.voice_library:
                    data = self.voice_library[word]
                    
                    if data.get('is_fallback'):
                        print(f"  Playing fallback WAV: {data['wav_file']}")
                        # Play fallback WAV file (waits for completion)
                        self._play_fallback_wav_file(data['wav_file'])
                    else:
                        print(f"  Playing user recording")
                        # Play recorded audio (waits for completion)
                        audio_data = np.array(data['audio_data'], dtype=np.int16)
                        self._play_audio_array(audio_data)
                    
                    # Small pause between words
                    time.sleep(0.2)
                else:
                    print(f"  Word '{word}' not found in voice library!")
                    
            print("Finished playing all words")
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    def _play_fallback_wav_file(self, wav_file):
        """Play a fallback WAV file using pygame."""
        try:
            sound = pygame.mixer.Sound(wav_file)
            sound.play()
            # Wait for the sound to finish
            while pygame.mixer.get_busy():
                time.sleep(0.01)
        except Exception as e:
            print(f"Error playing fallback WAV {wav_file}: {e}")
    
    def _play_audio_array(self, audio_data):
        """Play audio data using pygame."""
        try:
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Write audio to temporary file
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(22050)  # sample rate
                wav_file.writeframes(audio_data.tobytes())
            
            # Play using pygame Sound instead of music
            sound = pygame.mixer.Sound(temp_path)
            sound.play()
            
            # Wait for the sound to finish
            while pygame.mixer.get_busy():
                time.sleep(0.01)
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
            
        except Exception as e:
            print(f"Error playing audio array: {e}")
    
    def repeat_last_prediction(self):
        """Repeat the last prediction using TTS."""
        if not self.tts_enabled:
            print("TTS not available")
            return
        
        if not self.last_prediction:
            print("No prediction to repeat")
            return
        
        print(f"Repeating: {self.last_prediction}")
        self.speak_number(self.last_prediction)
    
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
        self.confidence_label.config(text="Draw digits to predict")
        
        # Clear individual digit predictions
        for widget in self.digit_predictions:
            widget.destroy()
        self.digit_predictions.clear()
        
        # Reset all confidence scores
        for i in range(10):
            self.score_labels[i].config(text="0.0%", foreground="black")
        
        self.status_label.config(text="Canvas cleared. Ready to draw numbers!", foreground="green")
    
    def update_brush_size(self, value):
        """Update brush size."""
        self.brush_size = int(value)
    
    def canvas_to_array(self):
        """Convert canvas drawing to 28x28 numpy array for the network."""
        # Create a PIL image from the canvas
        self.canvas.update()
        
        # Create PIL image and draw the canvas content
        img = Image.new('RGB', (self.canvas_width, self.canvas_height), 'black')
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
    
    def segment_digits(self, img_array):
        """
        Segment a multi-digit image into individual digit images.
        Returns a list of individual digit arrays ready for prediction.
        """
        # Find connected components (digits)
        # First, find all non-zero pixels
        rows, cols = np.where(img_array > 0)
        
        if len(rows) == 0:
            return []
        
        # Get bounding box of all content
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        
        # Extract the content region
        content = img_array[min_row:max_row+1, min_col:max_col+1]
        content_height, content_width = content.shape
        
        # Find vertical separations between digits
        # Sum pixels vertically to find gaps
        vertical_sums = np.sum(content, axis=0)
        
        # Find digit boundaries by looking for gaps
        digit_boundaries = []
        in_digit = False
        start_col = 0
        
        # Define a threshold for what constitutes a gap
        gap_threshold = np.max(vertical_sums) * 0.1  # 10% of max column sum
        
        for col in range(content_width):
            if vertical_sums[col] > gap_threshold:  # We're in a digit
                if not in_digit:
                    start_col = col
                    in_digit = True
            else:  # We're in a gap
                if in_digit:
                    digit_boundaries.append((start_col, col))
                    in_digit = False
        
        # Don't forget the last digit if the image ends with content
        if in_digit:
            digit_boundaries.append((start_col, content_width))
        
        # If no clear boundaries found, treat as single digit
        if not digit_boundaries:
            digit_boundaries = [(0, content_width)]
        
        # Extract individual digits
        digit_images = []
        for start_col, end_col in digit_boundaries:
            # Extract the digit region
            digit_region = content[:, start_col:end_col]
            
            # Skip very narrow regions (likely noise)
            if end_col - start_col < content_width * 0.1:
                continue
            
            # Find the actual content within this region
            digit_rows, digit_cols = np.where(digit_region > 0)
            
            if len(digit_rows) == 0:
                continue
            
            # Get tight bounding box for this digit
            digit_min_row = digit_rows.min()
            digit_max_row = digit_rows.max()
            digit_min_col = digit_cols.min()
            digit_max_col = digit_cols.max()
            
            # Extract the tight digit
            tight_digit = digit_region[digit_min_row:digit_max_row+1, 
                                     digit_min_col:digit_max_col+1]
            
            # Process this digit like a single digit
            digit_img = self.process_single_digit(tight_digit)
            digit_images.append(digit_img)
        
        return digit_images
    
    def process_single_digit(self, digit_array):
        """Process a single digit array into 28x28 format for the network."""
        height, width = digit_array.shape
        
        # Calculate aspect ratio and size for centering
        max_dim = max(height, width)
        
        # Add some padding
        padding = max(int(max_dim * 0.2), 2)
        canvas_size = max_dim + 2 * padding
        
        # Create a black canvas
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
        
        # Calculate position to center the digit
        y_offset = (canvas_size - height) // 2
        x_offset = (canvas_size - width) // 2
        
        # Place the digit in the center
        canvas[y_offset:y_offset+height, x_offset:x_offset+width] = digit_array
        
        # Convert to PIL for high-quality resize
        canvas_img = Image.fromarray(canvas.astype(np.uint8))
        
        # Resize to 28x28
        resized_img = canvas_img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert back to numpy and normalize
        final_array = np.array(resized_img, dtype=np.float32) / 255.0
        
        # Apply smoothing if available
        try:
            from scipy import ndimage
            final_array = ndimage.gaussian_filter(final_array, sigma=0.5)
        except ImportError:
            pass
        
        # Ensure proper normalization
        if final_array.max() > 0:
            final_array = final_array / final_array.max()
        
        return final_array.reshape(1, 784)
    
    def predict_digits(self):
        """Predict multiple digits from the drawn number."""
        if not self.model_loaded:
            messagebox.showerror("Error", "No model loaded. Please check the model file.")
            return
        
        # Check if anything is drawn
        if len(self.canvas.find_all()) == 0:
            messagebox.showinfo("No Drawing", "Please draw some digits first!")
            return
        
        try:
            # Get the full drawn image
            img = Image.new('RGB', (self.canvas_width, self.canvas_height), 'black')
            draw = ImageDraw.Draw(img)
            
            # Recreate the drawing
            for item in self.canvas.find_all():
                coords = self.canvas.coords(item)
                if len(coords) == 4:  # oval
                    draw.ellipse(coords, fill='white')
            
            # Convert to grayscale and numpy array
            img = img.convert('L')
            img_array = np.array(img, dtype=np.float32)
            
            # Segment into individual digits
            digit_images = self.segment_digits(img_array)
            
            if not digit_images:
                messagebox.showinfo("No Digits Found", "Could not identify any digits in the drawing.")
                return
            
            # Clear previous digit predictions
            for widget in self.digit_predictions:
                widget.destroy()
            self.digit_predictions.clear()
            
            # Predict each digit
            predicted_number = ""
            all_confidences = []
            
            for i, digit_img in enumerate(digit_images):
                # Get prediction
                probabilities = self.network.forward(digit_img)[0]
                predicted_digit = np.argmax(probabilities)
                confidence = probabilities[predicted_digit]
                
                predicted_number += str(predicted_digit)
                all_confidences.append(confidence)
                
                # Create label for this digit prediction
                digit_frame = ttk.Frame(self.digits_frame)
                digit_frame.grid(row=i//5, column=i%5, padx=5, pady=5)
                
                digit_label = ttk.Label(digit_frame, text=str(predicted_digit), 
                                       font=("Arial", 20, "bold"), foreground="blue")
                digit_label.grid(row=0, column=0)
                
                conf_label = ttk.Label(digit_frame, text=f"{confidence:.1%}", 
                                      font=("Arial", 8))
                conf_label.grid(row=1, column=0)
                
                self.digit_predictions.extend([digit_frame, digit_label, conf_label])
            
            # Update main prediction display
            self.prediction_label.config(text=predicted_number)
            avg_confidence = np.mean(all_confidences)
            self.confidence_label.config(text=f"Detected {len(digit_images)} digits - Avg confidence: {avg_confidence:.1%}")
            
            # Update confidence scores for the last digit processed
            if digit_images:
                last_probabilities = self.network.forward(digit_images[-1])[0]
                last_predicted = np.argmax(last_probabilities)
                
                for i in range(10):
                    score_text = f"{last_probabilities[i]:.1%}"
                    color = "red" if i == last_predicted else "black"
                    
                    self.score_labels[i].config(text=score_text, foreground=color)
                    if i == last_predicted:
                        self.score_labels[i].config(font=("Arial", 10, "bold"))
                    else:
                        self.score_labels[i].config(font=("Arial", 10, "normal"))
            
            self.status_label.config(
                text=f"Predicted number: {predicted_number} ({len(digit_images)} digits detected)", 
                foreground="blue"
            )
            
            # Store and speak the prediction
            self.last_prediction = predicted_number
            if self.tts_enabled:
                self.speak_number(predicted_number)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error predicting digits: {str(e)}")
            print(f"Prediction error: {e}")
    
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
            
            # Clear multi-digit predictions
            for widget in self.digit_predictions:
                widget.destroy()
            self.digit_predictions.clear()
            
            # Update main prediction display
            self.prediction_label.config(text=str(predicted_digit))
            self.confidence_label.config(text=f"Single digit - Confidence: {confidence:.1%}")
            
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
            
            self.status_label.config(text=f"Single digit prediction: {predicted_digit} ({confidence:.1%} confidence)", 
                                   foreground="blue")
            
            # Store and speak the prediction
            self.last_prediction = str(predicted_digit)
            if self.tts_enabled:
                self.speak_number(str(predicted_digit))
            
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
            img = Image.new('RGB', (self.canvas_width, self.canvas_height), 'black')
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
    print("Starting MNIST Multi-Digit Recognition GUI...")
    print("Draw single digits or multi-digit numbers (like 1024) and click 'Predict Digits'!")
    print("Use 'Single Digit Mode' for individual digit recognition.")
    print("Use 'Predict Digits' for multi-digit number recognition.")
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
    except Exception as e:
        print(f"Application error: {e}")


if __name__ == "__main__":
    main()