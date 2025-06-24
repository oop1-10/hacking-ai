"""
Voice Number Recorder & Text-to-Speech System
Records number words (one, two, three, etc.) and synthesizes any number using your voice.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import wave
import threading
import time
import os
import json
from datetime import datetime
import tempfile
import pygame
import inflect

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("PyAudio not available. Install with: pip install pyaudio")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("SoundFile not available. Install with: pip install soundfile")


class VoiceNumberRecorder:
    def __init__(self, root):
        self.root = root
        self.root.title("Personal Voice Number TTS Recorder")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        print("Setting up GUI window...")
        
        # Audio settings
        self.sample_rate = 44100
        self.channels = 1
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
        
        # Recording state
        self.is_recording = False
        self.current_recording = None
        self.audio_data = []
        self.pyaudio_instance = None
        self.stream = None
        
        # Voice data storage
        self.voice_library = {}
        self.voice_library_path = "voice_library.json"
        self.fallback_dir = "fallback"
        self.user_dir = "user"
        
        # Number words to record
        self.number_words = {
            0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
            11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
            16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
            30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
            80: "eighty", 90: "ninety", 100: "hundred", 1000: "thousand",
            1000000: "million", 1000000000: "billion"
        }
        
        # Additional words
        self.additional_words = ["and", "point", "minus", "negative"]
        
        # All words to record
        self.all_words = list(self.number_words.values()) + self.additional_words
        self.current_word_index = 0
        
        # Initialize pygame for audio playback
        try:
            print("Initializing pygame mixer...")
            pygame.mixer.pre_init(frequency=self.sample_rate, size=-16, channels=self.channels, buffer=512)
            pygame.mixer.init()
            print("Pygame mixer initialized")
        except Exception as e:
            print(f"Pygame mixer initialization failed: {e}")
            print("   Audio playback may not work properly")
        
        # Initialize inflect for number conversion
        self.inflect_engine = inflect.engine()
        
        # Setup UI
        print("Creating user interface...")
        self.setup_ui()
        print("UI setup complete")
        
        # Load existing voice library
        print("Loading voice library...")
        self.load_fallback_voices()
        self.load_voice_library()
        
        # Initialize PyAudio if available
        if PYAUDIO_AVAILABLE:
            print("Initializing audio system...")
            self.init_pyaudio()
        else:
            self.status_var.set("PyAudio not available - install with: pip install pyaudio")
            # Disable recording button
            self.disable_recording_features()
        
        print("Application initialization complete")
    
    def setup_ui(self):
        """Setup the user interface."""
        try:
            print("  Creating notebook tabs...")
            # Create notebook for tabs
            self.notebook = ttk.Notebook(self.root)
            self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            print("  Setting up recording tab...")
            # Recording tab
            self.setup_recording_tab()
            
            print("  Setting up synthesis tab...")
            # Synthesis tab
            self.setup_synthesis_tab()
            
            print("  Setting up management tab...")
            # Management tab
            self.setup_management_tab()
            
            print("  Creating status bar...")
            # Status bar
            self.status_var = tk.StringVar()
            self.status_var.set("Ready to record voice samples")
            self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
            self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
            
            print("  UI components created successfully")
            
        except Exception as e:
            print(f"Error setting up UI: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def setup_recording_tab(self):
        """Setup the recording interface."""
        self.recording_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.recording_frame, text="Record Words")
        
        # Title
        title_label = ttk.Label(self.recording_frame, text="Voice Sample Recording", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(10, 20))
        
        # Instructions
        instructions = ttk.Label(self.recording_frame, 
                                text="Record each number word clearly. Click 'Record' and speak the highlighted word.",
                                font=("Arial", 10))
        instructions.pack(pady=(0, 20))
        
        # Progress frame
        progress_frame = ttk.LabelFrame(self.recording_frame, text="Recording Progress", padding="10")
        progress_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=len(self.all_words))
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Progress text
        self.progress_text = ttk.Label(progress_frame, text=f"0 / {len(self.all_words)} words recorded")
        self.progress_text.pack()
        
        # Current word frame
        word_frame = ttk.LabelFrame(self.recording_frame, text="Current Word", padding="20")
        word_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        self.current_word_label = ttk.Label(word_frame, text=self.all_words[0], 
                                           font=("Arial", 24, "bold"), foreground="blue")
        self.current_word_label.pack()
        
        # Pronunciation guide
        self.pronunciation_label = ttk.Label(word_frame, text="Say this word clearly", 
                                           font=("Arial", 10), foreground="gray")
        self.pronunciation_label.pack(pady=(5, 0))
        
        # Recording controls
        controls_frame = ttk.Frame(self.recording_frame)
        controls_frame.pack(pady=20)
        
        self.record_button = ttk.Button(controls_frame, text="Start Recording", 
                                       command=self.toggle_recording, style="Accent.TButton")
        self.record_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.play_button = ttk.Button(controls_frame, text="Play Recording", 
                                     command=self.play_current_recording, state="disabled")
        self.play_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.save_button = ttk.Button(controls_frame, text="Save & Next", 
                                     command=self.save_and_next, state="disabled")
        self.save_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.skip_button = ttk.Button(controls_frame, text="Skip Word", 
                                     command=self.skip_word)
        self.skip_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Navigation controls
        nav_frame = ttk.Frame(self.recording_frame)
        nav_frame.pack(pady=10)
        
        self.prev_button = ttk.Button(nav_frame, text="Previous", 
                                     command=self.previous_word)
        self.prev_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.next_button = ttk.Button(nav_frame, text="Next", 
                                     command=self.next_word)
        self.next_button.pack(side=tk.LEFT)
        
        # Recording level indicator
        level_frame = ttk.LabelFrame(self.recording_frame, text="Recording Level", padding="10")
        level_frame.pack(fill=tk.X, padx=20, pady=(20, 0))
        
        self.level_var = tk.DoubleVar()
        self.level_bar = ttk.Progressbar(level_frame, variable=self.level_var, 
                                       maximum=100, mode='determinate')
        self.level_bar.pack(fill=tk.X)
        
        self.level_label = ttk.Label(level_frame, text="Speak to see audio level")
        self.level_label.pack()
    
    def setup_synthesis_tab(self):
        """Setup the number synthesis interface."""
        self.synthesis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.synthesis_frame, text="Speak Numbers")
        
        # Title
        title_label = ttk.Label(self.synthesis_frame, text="Number Text-to-Speech", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(10, 20))
        
        # Instructions
        instructions = ttk.Label(self.synthesis_frame, 
                                text="Enter any number and hear it spoken in your recorded voice!",
                                font=("Arial", 10))
        instructions.pack(pady=(0, 20))
        
        # Number input frame
        input_frame = ttk.LabelFrame(self.synthesis_frame, text="Number Input", padding="20")
        input_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Number entry
        entry_frame = ttk.Frame(input_frame)
        entry_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(entry_frame, text="Enter number:", font=("Arial", 12)).pack(side=tk.LEFT)
        
        self.number_entry = ttk.Entry(entry_frame, font=("Arial", 14), width=20)
        self.number_entry.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        self.number_entry.bind('<Return>', lambda e: self.synthesize_number())
        
        # Preset buttons
        preset_frame = ttk.Frame(input_frame)
        preset_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(preset_frame, text="Quick examples:").pack(side=tk.LEFT)
        
        presets = ["123", "1024", "2023", "999", "1000000"]
        for preset in presets:
            btn = ttk.Button(preset_frame, text=preset, width=8,
                           command=lambda p=preset: self.set_number(p))
            btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # Synthesis controls
        synth_controls = ttk.Frame(self.synthesis_frame)
        synth_controls.pack(pady=20)
        
        self.speak_button = ttk.Button(synth_controls, text="Speak Number", 
                                      command=self.synthesize_number, style="Accent.TButton")
        self.speak_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(synth_controls, text="Stop", 
                                     command=self.stop_playback)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.save_audio_button = ttk.Button(synth_controls, text="Save Audio", 
                                           command=self.save_synthesized_audio)
        self.save_audio_button.pack(side=tk.LEFT)
        
        # Text breakdown frame
        breakdown_frame = ttk.LabelFrame(self.synthesis_frame, text="Word Breakdown", padding="10")
        breakdown_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(20, 0))
        
        # Text widget for breakdown
        self.breakdown_text = tk.Text(breakdown_frame, height=8, wrap=tk.WORD, font=("Arial", 10))
        scrollbar = ttk.Scrollbar(breakdown_frame, orient=tk.VERTICAL, command=self.breakdown_text.yview)
        self.breakdown_text.configure(yscrollcommand=scrollbar.set)
        
        self.breakdown_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Speed control
        speed_frame = ttk.LabelFrame(self.synthesis_frame, text="Playback Speed", padding="10")
        speed_frame.pack(fill=tk.X, padx=20, pady=(10, 0))
        
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(speed_frame, from_=0.5, to=2.0, variable=self.speed_var, 
                               orient=tk.HORIZONTAL)
        speed_scale.pack(fill=tk.X, pady=(0, 5))
        
        speed_label = ttk.Label(speed_frame, text="Speed: 1.0x")
        speed_label.pack()
        
        # Update speed label
        def update_speed_label(*args):
            speed_label.config(text=f"Speed: {self.speed_var.get():.1f}x")
        self.speed_var.trace('w', update_speed_label)
    
    def setup_management_tab(self):
        """Setup the voice library management interface."""
        self.management_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.management_frame, text="Manage Library")
        
        # Title
        title_label = ttk.Label(self.management_frame, text="Voice Library Management", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(10, 20))
        
        # Library info frame
        info_frame = ttk.LabelFrame(self.management_frame, text="Library Information", padding="10")
        info_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        self.library_info_label = ttk.Label(info_frame, text="No recordings found")
        self.library_info_label.pack()
        
        # Word list frame
        list_frame = ttk.LabelFrame(self.management_frame, text="Recorded Words", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Treeview for word list
        columns = ("Word", "Status", "Quality", "Duration")
        self.word_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            self.word_tree.heading(col, text=col)
            self.word_tree.column(col, width=100)
        
        # Scrollbar for treeview
        tree_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.word_tree.yview)
        self.word_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.word_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Management controls
        mgmt_controls = ttk.Frame(self.management_frame)
        mgmt_controls.pack(pady=20)
        
        self.play_selected_button = ttk.Button(mgmt_controls, text="Play Selected", 
                                              command=self.play_selected_word)
        self.play_selected_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.delete_selected_button = ttk.Button(mgmt_controls, text="Delete Selected", 
                                                command=self.delete_selected_word)
        self.delete_selected_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.export_button = ttk.Button(mgmt_controls, text="Export Library", 
                                       command=self.export_library)
        self.export_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.import_button = ttk.Button(mgmt_controls, text="Import Library", 
                                       command=self.import_library)
        self.import_button.pack(side=tk.LEFT)
        
        # Bind treeview selection
        self.word_tree.bind('<<TreeviewSelect>>', self.on_word_select)
    
    def disable_recording_features(self):
        """Disable recording features when PyAudio is not available."""
        # This will be called after UI setup, so we need to defer it
        def disable_after_ui():
            try:
                self.record_button.config(state="disabled", text="PyAudio Required")
                self.play_button.config(state="disabled")
                self.save_button.config(state="disabled")
            except AttributeError:
                # UI not ready yet, try again later
                self.root.after(100, disable_after_ui)
        
        self.root.after(100, disable_after_ui)
    
    def init_pyaudio(self):
        """Initialize PyAudio."""
        global PYAUDIO_AVAILABLE
        
        if not PYAUDIO_AVAILABLE:
            self.status_var.set("PyAudio not available - recording disabled")
            return
        
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Test if we can find a suitable input device
            self.find_suitable_input_device()
            
            self.status_var.set("Audio system initialized - ready to record")
        except Exception as e:
            self.status_var.set(f"Audio initialization failed: {e}")
            print(f"PyAudio initialization error: {e}")
            PYAUDIO_AVAILABLE = False
    
    def find_suitable_input_device(self):
        """Find a suitable input device and update audio settings."""
        if not self.pyaudio_instance:
            return
        
        # Get default input device info
        try:
            default_device = self.pyaudio_instance.get_default_input_device_info()
            print(f"Default input device: {default_device['name']}")
            
            # Use the default device's preferred sample rate if available
            if 'defaultSampleRate' in default_device:
                suggested_rate = int(default_device['defaultSampleRate'])
                if suggested_rate in [44100, 48000, 22050, 16000]:
                    self.sample_rate = suggested_rate
                    print(f"Using sample rate: {self.sample_rate} Hz")
            
            # Test if the device supports our format
            self.test_audio_format()
            
        except Exception as e:
            print(f"Error finding input device: {e}")
            # Try to list available devices for debugging
            self.list_audio_devices()
    
    def test_audio_format(self):
        """Test if the current audio format is supported."""
        try:
            # Test if we can open a stream with current settings
            test_stream = self.pyaudio_instance.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            test_stream.close()
            print("Audio format test successful")
        except Exception as e:
            print(f"Audio format test failed: {e}")
            # Try alternative settings
            self.try_alternative_audio_settings()
    
    def try_alternative_audio_settings(self):
        """Try alternative audio settings if default ones fail."""
        alternative_rates = [44100, 48000, 22050, 16000, 8000]
        alternative_formats = [pyaudio.paInt16, pyaudio.paInt24, pyaudio.paFloat32]
        
        for rate in alternative_rates:
            for format_type in alternative_formats:
                try:
                    test_stream = self.pyaudio_instance.open(
                        format=format_type,
                        channels=1,  # Force mono
                        rate=rate,
                        input=True,
                        frames_per_buffer=1024
                    )
                    test_stream.close()
                    
                    # If we get here, this combination works
                    self.sample_rate = rate
                    self.audio_format = format_type
                    self.channels = 1
                    self.chunk_size = 1024
                    
                    print(f"Found working audio settings: {rate}Hz, format={format_type}")
                    return
                    
                except Exception:
                    continue
        
        # If nothing works, disable recording
        print("Could not find working audio settings")
        raise Exception("No compatible audio format found")
    
    def list_audio_devices(self):
        """List available audio devices for debugging."""
        try:
            print("\nAvailable audio devices:")
            for i in range(self.pyaudio_instance.get_device_count()):
                device_info = self.pyaudio_instance.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:  # Input device
                    print(f"  {i}: {device_info['name']} (inputs: {device_info['maxInputChannels']})")
        except Exception as e:
            print(f"Error listing devices: {e}")
    
    def toggle_recording(self):
        """Toggle recording state."""
        global PYAUDIO_AVAILABLE
        
        if not PYAUDIO_AVAILABLE:
            messagebox.showerror("Error", "PyAudio not available. Please install PyAudio.")
            return
        
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        """Start recording audio."""
        global PYAUDIO_AVAILABLE
        
        if not PYAUDIO_AVAILABLE:
            messagebox.showerror("Error", "PyAudio not available. Please install PyAudio first.")
            return
        
        if not self.pyaudio_instance:
            messagebox.showerror("Error", "Audio system not initialized.")
            return
            
        try:
            self.is_recording = True
            self.audio_data = []
            self.record_button.config(text="Stop Recording")
            self.status_var.set(f"Recording '{self.all_words[self.current_word_index]}'...")
            
            # Try to start recording stream with error handling
            try:
                self.stream = self.pyaudio_instance.open(
                    format=self.audio_format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self.audio_callback
                )
                
                self.stream.start_stream()
                print(f"Recording started: {self.sample_rate}Hz, {self.channels} channel(s)")
                
            except Exception as stream_error:
                print(f"Stream creation failed: {stream_error}")
                # Try alternative method without callback
                self.start_blocking_recording()
            
        except Exception as e:
            error_msg = str(e)
            if "Unanticipated host error" in error_msg:
                error_msg = ("Audio device error. Try:\n"
                           "1. Check microphone permissions\n"
                           "2. Close other audio applications\n"
                           "3. Try a different microphone\n"
                           "4. Restart the application")
            
            messagebox.showerror("Recording Error", f"Could not start recording:\n{error_msg}")
            self.is_recording = False
            self.record_button.config(text="Start Recording")
            self.status_var.set("Recording failed - check audio settings")
    
    def start_blocking_recording(self):
        """Alternative recording method using blocking mode."""
        try:
            print("Trying blocking recording mode...")
            self.stream = self.pyaudio_instance.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            # Start recording in a separate thread
            self.recording_thread = threading.Thread(target=self.blocking_recording_loop)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            print("Blocking recording started successfully")
            
        except Exception as e:
            print(f"Blocking recording also failed: {e}")
            raise e
    
    def blocking_recording_loop(self):
        """Recording loop for blocking mode."""
        try:
            while self.is_recording and self.stream:
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    self.audio_data.append(audio_chunk)
                    
                    # Update level indicator
                    if len(audio_chunk) > 0:
                        level = np.abs(audio_chunk).mean() / 32767.0 * 100
                        self.level_var.set(min(level, 100))
                        
                        # Update level text on main thread
                        if level > 70:
                            text = "Recording level: HIGH"
                        elif level > 30:
                            text = "Recording level: GOOD"
                        elif level > 5:
                            text = "Recording level: LOW"
                        else:
                            text = "Recording level: SILENT"
                        
                        self.root.after(0, lambda: self.level_label.config(text=text))
                
                except Exception as e:
                    print(f"Error in recording loop: {e}")
                    break
                    
        except Exception as e:
            print(f"Recording loop error: {e}")
            self.root.after(0, lambda: self.status_var.set(f"Recording error: {e}"))
    
    def stop_recording(self):
        """Stop recording audio."""
        self.is_recording = False
        
        # Wait for recording thread to finish if using blocking mode
        if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1.0)
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"Error stopping stream: {e}")
            finally:
                self.stream = None
        
        self.record_button.config(text="Start Recording")
        
        if self.audio_data:
            # Convert to numpy array
            self.current_recording = np.concatenate(self.audio_data)
            self.play_button.config(state="normal")
            self.save_button.config(state="normal")
            
            duration = len(self.current_recording) / self.sample_rate
            self.status_var.set(f"Recording complete ({duration:.1f}s). Review and save.")
        else:
            self.status_var.set("No audio recorded")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for recording."""
        if self.is_recording:
            # Convert to numpy array
            audio_chunk = np.frombuffer(in_data, dtype=np.int16)
            self.audio_data.append(audio_chunk)
            
            # Update level indicator
            if len(audio_chunk) > 0:
                level = np.abs(audio_chunk).mean() / 32767.0 * 100
                self.level_var.set(min(level, 100))
                
                # Update level color based on volume
                if level > 70:
                    self.level_label.config(text="Recording level: HIGH")
                elif level > 30:
                    self.level_label.config(text="Recording level: GOOD")
                elif level > 5:
                    self.level_label.config(text="Recording level: LOW")
                else:
                    self.level_label.config(text="Recording level: SILENT")
        
        return (in_data, pyaudio.paContinue)
    
    def play_current_recording(self):
        """Play the current recording."""
        if self.current_recording is not None:
            self.play_audio_array(self.current_recording)
    
    def play_mp3_file(self, mp3_file):
        """Play an MP3 file using pygame."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=1024)
            
            # Stop any currently playing audio
            pygame.mixer.stop()
            pygame.mixer.music.stop()
            
            # Load and play MP3
            pygame.mixer.music.load(mp3_file)
            pygame.mixer.music.play()
            
            self.status_var.set(f"Playing MP3: {os.path.basename(mp3_file)}")
            
        except Exception as e:
            print(f"Error playing MP3 {mp3_file}: {e}")
            self.status_var.set(f"MP3 playback error: {e}")

    def play_fallback_wav_file(self, wav_file):
        """Play a fallback WAV file using pygame."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=1024)
            
            # Stop any currently playing audio
            pygame.mixer.stop()
            pygame.mixer.music.stop()
            
            # Load and play WAV
            pygame.mixer.music.load(wav_file)
            pygame.mixer.music.play()
            
            self.status_var.set(f"Playing fallback: {os.path.basename(wav_file)}")
            
        except Exception as e:
            print(f"Error playing fallback WAV {wav_file}: {e}")
            self.status_var.set(f"WAV playback error: {e}")

    def play_audio_array(self, audio_data, speed=1.0):
        """Play audio data using pygame."""
        try:
            # Adjust speed if needed
            if speed != 1.0:
                # Simple speed adjustment by resampling
                new_length = int(len(audio_data) / speed)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), new_length),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Ensure audio is in correct format
            audio_data = audio_data.astype(np.int16)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Write audio to temporary file
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(int(self.sample_rate))
                wav_file.writeframes(audio_data.tobytes())
            
            # Play using pygame
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            
            # Clean up temp file after a delay
            def cleanup():
                time.sleep(len(audio_data) / self.sample_rate + 1)
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            threading.Thread(target=cleanup, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Playback Error", f"Could not play audio: {e}")
    
    def save_and_next(self):
        """Save current recording and move to next word."""
        if self.current_recording is not None:
            word = self.all_words[self.current_word_index]
            
            # Save to voice library
            self.voice_library[word] = {
                'audio_data': self.current_recording.tolist(),
                'sample_rate': self.sample_rate,
                'duration': len(self.current_recording) / self.sample_rate,
                'recorded_at': datetime.now().isoformat()
            }
            
            # Save to file
            self.save_voice_library()
            
            # Update UI
            self.update_progress()
            self.update_word_list()
            
            # Move to next word
            self.next_word()
            
            # Reset recording state
            self.current_recording = None
            self.play_button.config(state="disabled")
            self.save_button.config(state="disabled")
            
            self.status_var.set(f"Saved '{word}'. Ready for next word.")
    
    def skip_word(self):
        """Skip current word without saving."""
        self.next_word()
        self.status_var.set("Word skipped")
    
    def next_word(self):
        """Move to next word."""
        if self.current_word_index < len(self.all_words) - 1:
            self.current_word_index += 1
        else:
            self.current_word_index = 0  # Loop back to start
        
        self.update_current_word_display()
    
    def previous_word(self):
        """Move to previous word."""
        if self.current_word_index > 0:
            self.current_word_index -= 1
        else:
            self.current_word_index = len(self.all_words) - 1  # Loop to end
        
        self.update_current_word_display()
    
    def update_current_word_display(self):
        """Update the current word display."""
        word = self.all_words[self.current_word_index]
        self.current_word_label.config(text=word)
        
        # Update pronunciation guide
        if word in ["and", "point", "minus", "negative"]:
            self.pronunciation_label.config(text=f"Say '{word}' clearly")
        else:
            self.pronunciation_label.config(text=f"Say the number '{word}' clearly")
        
        # Reset recording state
        self.current_recording = None
        self.play_button.config(state="disabled")
        self.save_button.config(state="disabled")
        self.level_var.set(0)
    
    def update_progress(self):
        """Update progress indicators."""
        recorded_count = len(self.voice_library)
        total_count = len(self.all_words)
        
        self.progress_var.set(recorded_count)
        self.progress_text.config(text=f"{recorded_count} / {total_count} words recorded")
        
        # Update library info
        self.library_info_label.config(
            text=f"Voice Library: {recorded_count} words recorded\n"
                 f"Completion: {recorded_count/total_count*100:.1f}%"
        )
    
    def set_number(self, number_str):
        """Set the number in the entry field."""
        self.number_entry.delete(0, tk.END)
        self.number_entry.insert(0, number_str)
    
    def synthesize_number(self):
        """Synthesize speech for the entered number."""
        number_str = self.number_entry.get().strip()
        
        if not number_str:
            messagebox.showwarning("Input Required", "Please enter a number")
            return
        
        try:
            # Convert number to words
            number = float(number_str)
            
            if '.' in number_str:
                # Handle decimal numbers
                words = self.number_to_words_decimal(number)
            else:
                # Handle integers
                words = self.number_to_words_custom(int(number))
        
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number")
            return
        
        # Show breakdown
        self.show_word_breakdown(number_str, words)
        
        # Check if we have all required words
        missing_words = [word for word in words if word not in self.voice_library]
        
        if missing_words:
            messagebox.showwarning(
                "Missing Recordings", 
                f"Missing voice recordings for: {', '.join(missing_words)}\n\n"
                "Please record these words first in the 'Record Words' tab."
            )
            return
        
        # Synthesize audio
        try:
            combined_audio = self.combine_word_audio(words)
            
            if combined_audio is not None:
                speed = self.speed_var.get()
                self.play_audio_array(combined_audio, speed)
                self.last_synthesized_audio = combined_audio
                self.status_var.set(f"Speaking number: {number_str}")
            else:
                messagebox.showerror("Synthesis Error", "Could not synthesize audio")
                
        except Exception as e:
            messagebox.showerror("Synthesis Error", f"Error creating speech: {e}")
    
    def number_to_words_custom(self, number):
        """Convert number to words using basic individual words."""
        if number < 0:
            words = ['negative']
            number = abs(number)
        else:
            words = []
        
        if number == 0:
            return ['zero']
        
        # Basic number words
        ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 
                'seventeen', 'eighteen', 'nineteen']
        
        tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
        
        # Handle numbers up to 999,999
        if number >= 1000000:
            millions = number // 1000000
            words.extend(self.number_to_words_custom(millions))
            words.append('million')
            number %= 1000000
        
        if number >= 1000:
            thousands = number // 1000
            words.extend(self.number_to_words_custom(thousands))
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

    def number_to_words_decimal(self, number):
        """Convert decimal number to words."""
        if number < 0:
            words = ['negative']
            number = abs(number)
        else:
            words = []
        
        # Split into integer and decimal parts
        integer_part = int(number)
        decimal_part = number - integer_part
        
        # Convert integer part
        if integer_part == 0:
            words.append('zero')
        else:
            words.extend(self.number_to_words_custom(integer_part))
        
        # Add decimal part
        if decimal_part > 0:
            words.append('point')
            # Convert each decimal digit separately
            decimal_str = f"{decimal_part:.10f}".split('.')[1].rstrip('0')
            digit_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
            for digit in decimal_str:
                words.append(digit_names[int(digit)])
        
        return words
    
    def show_word_breakdown(self, number, words):
        """Show the word breakdown in the text widget."""
        self.breakdown_text.delete(1.0, tk.END)
        
        breakdown = f"Number: {number}\n\n"
        breakdown += f"Word sequence: {' '.join(words)}\n\n"
        breakdown += "Individual words:\n"
        
        for i, word in enumerate(words, 1):
            status = "Recorded" if word in self.voice_library else "Missing"
            breakdown += f"{i:2d}. {word:<15} {status}\n"
        
        self.breakdown_text.insert(1.0, breakdown)
    
    def combine_word_audio(self, words):
        """Combine audio for multiple words."""
        combined_audio = np.array([], dtype=np.int16)
        
        for word in words:
            if word in self.voice_library:
                data = self.voice_library[word]
                
                if data.get('is_fallback'):
                    # Handle fallback WAV file - load and convert to numpy array
                    word_audio = self.load_wav_as_array(data['wav_file'])
                else:
                    # Handle recorded audio
                    word_audio = np.array(data['audio_data'], dtype=np.int16)
                
                if word_audio is not None:
                    # Add the word audio
                    combined_audio = np.concatenate([combined_audio, word_audio])
                    
                    # Add a small pause between words (100ms)
                    pause_length = int(0.1 * self.sample_rate)
                    pause = np.zeros(pause_length, dtype=np.int16)
                    combined_audio = np.concatenate([combined_audio, pause])
        
        return combined_audio if len(combined_audio) > 0 else None
    
    def load_wav_as_array(self, wav_file):
        """Load WAV file and convert to numpy array."""
        try:
            # Load WAV file
            with wave.open(wav_file, 'rb') as wav_reader:
                frames = wav_reader.readframes(-1)
                sample_rate = wav_reader.getframerate()
                audio_data = np.frombuffer(frames, dtype=np.int16)
            
            return audio_data
            
        except Exception as e:
            print(f"Error loading WAV {wav_file}: {e}")
            return None
    
    def stop_playback(self):
        """Stop current audio playback."""
        pygame.mixer.music.stop()
        self.status_var.set("Playback stopped")
    
    def save_synthesized_audio(self):
        """Save the last synthesized audio to file."""
        if not hasattr(self, 'last_synthesized_audio'):
            messagebox.showwarning("No Audio", "No synthesized audio to save")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            title="Save Synthesized Audio"
        )
        
        if filename:
            try:
                with wave.open(filename, 'wb') as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(self.last_synthesized_audio.tobytes())
                
                messagebox.showinfo("Success", f"Audio saved to {filename}")
                
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save audio: {e}")
    
    def update_word_list(self):
        """Update the word list in management tab."""
        # Clear existing items
        for item in self.word_tree.get_children():
            self.word_tree.delete(item)
        
        # Add recorded words
        for word in self.all_words:
            if word in self.voice_library:
                data = self.voice_library[word]
                duration = data['duration']
                
                if data.get('is_fallback'):
                    status = "Fallback MP3"
                    quality = "Good"
                else:
                    status = "Recorded"
                    quality = "Good" if duration > 0.5 else "Short"
            else:
                duration = 0
                status = "Missing"
                quality = "N/A"
            
            self.word_tree.insert('', tk.END, values=(
                word, status, quality, f"{duration:.1f}s" if duration > 0 else "N/A"
            ))
    
    def on_word_select(self, event):
        """Handle word selection in tree view."""
        pass  # Can be expanded for word-specific actions
    
    def play_selected_word(self):
        """Play the selected word from the tree."""
        selection = self.word_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a word to play")
            return
        
        item = self.word_tree.item(selection[0])
        word = item['values'][0]
        
        if word in self.voice_library:
            data = self.voice_library[word]
            if data.get('is_fallback'):
                # Play fallback WAV file
                self.play_fallback_wav_file(data['wav_file'])
            else:
                # Play recorded audio
                audio_data = np.array(data['audio_data'], dtype=np.int16)
                self.play_audio_array(audio_data)
        else:
            messagebox.showwarning("No Recording", f"No recording found for '{word}'")
    
    def delete_selected_word(self):
        """Delete the selected word recording."""
        selection = self.word_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a word to delete")
            return
        
        item = self.word_tree.item(selection[0])
        word = item['values'][0]
        
        if word in self.voice_library:
            if messagebox.askyesno("Confirm Delete", f"Delete recording for '{word}'?"):
                del self.voice_library[word]
                self.save_voice_library()
                self.update_word_list()
                self.update_progress()
                self.status_var.set(f"Deleted recording for '{word}'")
    
    def save_voice_library(self):
        """Save voice library to file."""
        try:
            # Create user directory
            os.makedirs(self.user_dir, exist_ok=True)
            
            # Save metadata (only for user recordings, not fallback MP3s)
            metadata = {}
            for word, data in self.voice_library.items():
                if data.get('is_fallback'):
                    continue  # Skip fallback MP3s in metadata
                    
                metadata[word] = {
                    'sample_rate': data['sample_rate'],
                    'duration': data['duration'],
                    'recorded_at': data['recorded_at']
                }
                
                # Save audio data to user directory
                audio_file = f"{self.user_dir}/{word}.wav"
                audio_data = np.array(data['audio_data'], dtype=np.int16)
                
                with wave.open(audio_file, 'wb') as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(audio_data.tobytes())
            
            # Save metadata
            with open(self.voice_library_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save voice library: {e}")
    
    def load_fallback_voices(self):
        """Load WAV files from fallback directory if available."""
        try:
            if not os.path.exists(self.fallback_dir):
                print("No fallback directory found")
                return
                
            print(f"Loading fallback voices from {self.fallback_dir}/...")
            loaded_count = 0
            
            for word in self.all_words:
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
                            'sample_rate': sample_rate,
                            'duration': duration,
                            'recorded_at': 'fallback'
                        }
                        loaded_count += 1
                        print(f"  Loaded fallback: {word}")
                        
                    except Exception as e:
                        print(f"  Error loading {wav_file}: {e}")
                        
            print(f"Loaded {loaded_count} fallback WAV files")
            
        except Exception as e:
            print(f"Error loading fallback voices: {e}")

    def load_voice_library(self):
        """Load voice library from file."""
        try:
            if os.path.exists(self.voice_library_path):
                with open(self.voice_library_path, 'r') as f:
                    metadata = json.load(f)
                
                # Load user recorded audio files
                for word, meta in metadata.items():
                    # Skip if we already have a fallback MP3 for this word
                    if word in self.voice_library and self.voice_library[word].get('is_fallback'):
                        continue
                        
                    audio_file = f"{self.user_dir}/{word}.wav"
                        
                    if os.path.exists(audio_file):
                        try:
                            with wave.open(audio_file, 'rb') as wav_file:
                                frames = wav_file.readframes(-1)
                                audio_data = np.frombuffer(frames, dtype=np.int16)
                            
                            self.voice_library[word] = {
                                'audio_data': audio_data.tolist(),
                                'sample_rate': meta['sample_rate'],
                                'duration': meta['duration'],
                                'recorded_at': meta['recorded_at'],
                                'is_fallback': False
                            }
                        except Exception as e:
                            print(f"Could not load {audio_file}: {e}")
                
                self.update_progress()
                self.update_word_list()
                
        except Exception as e:
            print(f"Could not load voice library: {e}")
    
    def export_library(self):
        """Export voice library to a zip file."""
        try:
            import zipfile
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".zip",
                filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")],
                title="Export Voice Library"
            )
            
            if filename:
                with zipfile.ZipFile(filename, 'w') as zipf:
                    # Add metadata
                    zipf.write(self.voice_library_path, "voice_library.json")
                    
                    # Add audio files from user directory only
                    for word in self.voice_library:
                        if not self.voice_library[word].get('is_fallback'):
                            audio_file = f"{self.user_dir}/{word}.wav"
                            if os.path.exists(audio_file):
                                zipf.write(audio_file, f"user/{word}.wav")
                
                messagebox.showinfo("Success", f"Voice library exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not export library: {e}")
    
    def import_library(self):
        """Import voice library from a zip file."""
        try:
            import zipfile
            
            filename = filedialog.askopenfilename(
                filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")],
                title="Import Voice Library"
            )
            
            if filename:
                with zipfile.ZipFile(filename, 'r') as zipf:
                    zipf.extractall(".")
                
                self.load_voice_library()
                messagebox.showinfo("Success", "Voice library imported successfully")
                
        except Exception as e:
            messagebox.showerror("Import Error", f"Could not import library: {e}")
    
    def on_closing(self):
        """Handle application closing."""
        if self.is_recording:
            self.stop_recording()
        
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
        
        pygame.mixer.quit()
        self.root.destroy()


def main():
    """Main function to run the application."""
    print("Starting Personal Voice Number TTS Recorder...")
    
    if not PYAUDIO_AVAILABLE:
        print("Warning: PyAudio not available. Recording will be disabled.")
        print("To enable recording, install PyAudio with: pip install pyaudio")
        print("You can still use synthesis features if you have existing recordings.")
    else:
        print("Audio recording enabled - ready to record voice samples!")
    
    root = tk.Tk()
    
    try:
        app = VoiceNumberRecorder(root)
    except Exception as e:
        print(f"Error initializing application: {e}")
        return
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    print("Application started successfully!")
    print("Record number words and create your own text-to-speech system!")
    
    # Ensure window is visible
    root.deiconify()  # Make sure window is not minimized
    root.lift()       # Bring to front
    root.focus_force() # Give it focus
    
    print("GUI window should now be visible")
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
