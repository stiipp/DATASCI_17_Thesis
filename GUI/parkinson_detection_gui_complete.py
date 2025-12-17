"""
Parkinson's Disease Detection System - Complete GUI Application
Author: Thesis Project
Description: Complete pipeline for PD detection with preprocessing, training, and evaluation
"""

import os
import sys
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
import scipy.interpolate as interp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             precision_recall_curve, average_precision_score, roc_curve, auc)
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Bidirectional, LSTM, Dense, Dropout, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


class Attention(tf.keras.layers.Layer):
    """Custom Attention layer for BiLSTM model"""
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(1,), initializer="zeros")
        super(Attention, self).build(input_shape)
    
    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)


class AttentionWithWeights(tf.keras.layers.Layer):
    """Attention layer that returns both output and attention weights for visualization"""
    def __init__(self, **kwargs):
        super(AttentionWithWeights, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(1,), initializer="zeros")
        super(AttentionWithWeights, self).build(input_shape)
    
    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1), a


class ParkinsonDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Parkinson's Disease Detection System")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)
        
        # Make the window responsive - set minimum size
        self.root.minsize(1000, 700)
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Data storage
        self.healthy_path = None
        self.patient_path = None
        self.df_train = None
        self.df_test = None
        self.model = None
        self.scaler = None
        self.attention_model = None
        self.processing_thread = None
        self.log_queue = queue.Queue()
        
        # Results storage
        self.results = {}
        self.prediction_results = []
        self.attention_weights = None
        
        # Channel names (excluding Microphone)
        self.channel_names = [
            "Fingergrip", "Axial_Pressure",
            "Tilt_X", "Tilt_Y", "Tilt_Z"
        ]
        
        # Reproducibility settings
        self.seed = 42
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        
        import random
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        
        # Build GUI
        self.create_widgets()
        
        # Start log update checker
        self.root.after(100, self.process_log_queue)
        
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tab 1: Data Upload & Preprocessing
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text='1. Data Preprocessing')
        self.create_preprocessing_tab()
        
        # Tab 2: Model Training
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text='2. Model Training')
        self.create_training_tab()
        
        # Tab 3: Prediction
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text='3. Prediction')
        self.create_prediction_tab()
        
        # Tab 4: Results & Evaluation
        self.tab4 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab4, text='4. Results & Evaluation')
        self.create_results_tab()
        
        # Tab 5: Attention Analysis
        self.tab5 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab5, text='5. Attention Analysis & Important Time Steps')
        self.create_attention_tab()
        
    def create_preprocessing_tab(self):
        """Create preprocessing tab widgets"""
        main_frame = ttk.Frame(self.tab1, padding="10")
        main_frame.pack(fill='both', expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Step 1: Data Preprocessing",
            font=('Helvetica', 14, 'bold')
        )
        title_label.grid(row=0, column=0, pady=10)
        
        # Upload frame
        upload_frame = ttk.LabelFrame(main_frame, text="Upload Raw Signal Data", padding="10")
        upload_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        upload_frame.columnconfigure(1, weight=1)
        
        # Healthy signals
        ttk.Label(upload_frame, text="Healthy Signals Folder:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.healthy_path_var = tk.StringVar(value="No folder selected")
        ttk.Entry(upload_frame, textvariable=self.healthy_path_var, state='readonly').grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5
        )
        ttk.Button(upload_frame, text="Browse", command=self.browse_healthy_folder).grid(row=0, column=2, pady=5)
        
        # Patient signals
        ttk.Label(upload_frame, text="Patient Signals Folder:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.patient_path_var = tk.StringVar(value="No folder selected")
        ttk.Entry(upload_frame, textvariable=self.patient_path_var, state='readonly').grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5
        )
        ttk.Button(upload_frame, text="Browse", command=self.browse_patient_folder).grid(row=1, column=2, pady=5)
        
        # Preprocessing controls
        control_frame = ttk.LabelFrame(main_frame, text="Preprocessing Controls", padding="10")
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.preprocess_btn = ttk.Button(
            control_frame,
            text="Start Preprocessing",
            command=self.start_preprocessing,
            state='disabled'
        )
        self.preprocess_btn.pack(pady=5)
        
        # Progress
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            control_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate',
            length=600
        )
        self.progress_bar.pack(pady=5)
        
        self.status_var = tk.StringVar(value="Ready to upload data")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var, foreground='blue')
        self.status_label.pack(pady=5)
        
        # Log
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="10")
        log_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=100, height=20, font=('Courier', 9))
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Button(log_frame, text="Clear Log", command=self.clear_log).grid(row=1, column=0, pady=5)
        
    def create_training_tab(self):
        """Create training tab widgets"""
        main_frame = ttk.Frame(self.tab2, padding="10")
        main_frame.pack(fill='both', expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Step 2: Model Training",
            font=('Helvetica', 14, 'bold')
        )
        title_label.grid(row=0, column=0, pady=10)
        
        # Training controls
        control_frame = ttk.LabelFrame(main_frame, text="Training Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.train_btn = ttk.Button(
            control_frame,
            text="Start Training",
            command=self.start_training,
            state='disabled'
        )
        self.train_btn.pack(pady=5)
        
        # Training progress
        self.train_progress_var = tk.DoubleVar()
        self.train_progress_bar = ttk.Progressbar(
            control_frame,
            variable=self.train_progress_var,
            maximum=100,
            mode='determinate',
            length=600
        )
        self.train_progress_bar.pack(pady=5)
        
        self.train_status_var = tk.StringVar(value="Waiting for preprocessing to complete")
        self.train_status_label = ttk.Label(control_frame, textvariable=self.train_status_var, foreground='blue')
        self.train_status_label.pack(pady=5)
        
        # Training log
        log_frame = ttk.LabelFrame(main_frame, text="Training Log", padding="10")
        log_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.train_log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=100, height=25, font=('Courier', 9))
        self.train_log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def create_prediction_tab(self):
        """Create prediction tab for uploading and predicting new files"""
        main_frame = ttk.Frame(self.tab3, padding="10")
        main_frame.pack(fill='both', expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Step 3: Predict on New Files",
            font=('Helvetica', 14, 'bold')
        )
        title_label.grid(row=0, column=0, pady=10)
        
        # Model status
        status_frame = ttk.LabelFrame(main_frame, text="Model Status", padding="10")
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.model_status_var = tk.StringVar(value="⚠ No model loaded. Please train a model first.")
        self.model_status_label = ttk.Label(
            status_frame,
            textvariable=self.model_status_var,
            font=('Helvetica', 11)
        )
        self.model_status_label.pack()
        
        # Upload folders
        upload_frame = ttk.LabelFrame(main_frame, text="Upload Folders for Person-Level Prediction", padding="10")
        upload_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(upload_frame, text="Select one or more folders containing signal files (.txt):").pack(anchor='w', pady=5)
        
        # Initialize folder list
        self.prediction_folders = []
        
        # Listbox to show selected folders
        folder_display_frame = ttk.Frame(upload_frame)
        folder_display_frame.pack(fill='x', pady=5)
        
        self.folder_listbox = tk.Listbox(folder_display_frame, height=4)
        self.folder_listbox.pack(side='left', fill='both', expand=True)
        
        folder_scrollbar = ttk.Scrollbar(folder_display_frame, orient='vertical', command=self.folder_listbox.yview)
        self.folder_listbox.configure(yscrollcommand=folder_scrollbar.set)
        folder_scrollbar.pack(side='right', fill='y')
        
        btn_frame = ttk.Frame(upload_frame)
        btn_frame.pack(pady=5)
        
        self.browse_folder_btn = ttk.Button(
            btn_frame,
            text="Add Folder",
            command=self.browse_prediction_folder,
            state='disabled'
        )
        self.browse_folder_btn.pack(side='left', padx=5)
        
        self.clear_folders_btn = ttk.Button(
            btn_frame,
            text="Clear All",
            command=self.clear_prediction_folders,
            state='disabled'
        )
        self.clear_folders_btn.pack(side='left', padx=5)
        
        self.predict_btn = ttk.Button(
            btn_frame,
            text="Predict (Person-Level)",
            command=self.start_prediction,
            state='disabled'
        )
        self.predict_btn.pack(side='left', padx=5)
        
        # Progress
        self.predict_progress_var = tk.DoubleVar()
        self.predict_progress_bar = ttk.Progressbar(
            upload_frame,
            variable=self.predict_progress_var,
            maximum=100,
            mode='determinate',
            length=600
        )
        self.predict_progress_bar.pack(pady=5)
        
        self.predict_status_var = tk.StringVar(value="Ready to predict")
        ttk.Label(upload_frame, textvariable=self.predict_status_var, foreground='blue').pack()
        
        # Results display
        results_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        results_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Create Treeview for results
        columns = ('Person_ID', 'Num_Files', 'Prediction', 'Probability', 'Confidence')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=10)
        
        column_widths = {'Person_ID': 150, 'Num_Files': 100, 'Prediction': 150, 'Probability': 120, 'Confidence': 120}
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=column_widths[col])
        
        scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Bind selection event
        self.results_tree.bind('<<TreeviewSelect>>', self.on_prediction_select)
        
        # Export button
        btn_frame2 = ttk.Frame(results_frame)
        btn_frame2.grid(row=1, column=0, pady=5)
        
        self.export_btn = ttk.Button(
            btn_frame2,
            text="Export Results to CSV",
            command=self.export_predictions,
            state='disabled'
        )
        self.export_btn.pack(side='left', padx=5)
        
        self.export_pdf_btn = ttk.Button(
            btn_frame2,
            text="Export Results to PDF",
            command=self.export_predictions_pdf,
            state='disabled'
        )
        self.export_pdf_btn.pack(side='left', padx=5)
        
        self.view_attention_btn = ttk.Button(
            btn_frame2,
            text="View Attention Analysis",
            command=self.show_selected_attention,
            state='disabled'
        )
        self.view_attention_btn.pack(side='left', padx=5)
        
    def create_results_tab(self):
        """Create results tab widgets"""
        main_frame = ttk.Frame(self.tab4, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Step 4: Results & Evaluation",
            font=('Helvetica', 14, 'bold')
        )
        title_label.pack(pady=10)
        
        # Results display area
        self.results_frame = ttk.Frame(main_frame)
        self.results_frame.pack(fill='both', expand=True)
        
        # Placeholder
        self.results_placeholder = ttk.Label(
            self.results_frame,
            text="Results will appear here after training completes",
            font=('Helvetica', 12),
            foreground='gray'
        )
        self.results_placeholder.pack(pady=50)
    
    def create_attention_tab(self):
        """Create attention visualization tab with smooth scrolling"""
        # Create scrollable canvas
        canvas = tk.Canvas(self.tab5, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab5, orient="vertical", command=canvas.yview)
        self.attention_scrollable_frame = ttk.Frame(canvas)
        
        self.attention_scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.attention_scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Bind mouse wheel for smooth scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        # Store canvas for later reference
        self.attention_canvas = canvas
        
        # Main content frame
        main_frame = ttk.Frame(self.attention_scrollable_frame, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        title_label = ttk.Label(
            main_frame, 
            text="Attention Weights Analysis & Important Time Steps Identification",
            font=('Helvetica', 14, 'bold')
        )
        title_label.pack(pady=10)
        
        # Info frame
        info_frame = ttk.LabelFrame(main_frame, text="About Attention Mechanism & Important Time Steps", padding="10")
        info_frame.pack(fill='x', padx=10, pady=5)
        
        info_text = (
            "🧠 ATTENTION MECHANISM ANALYSIS:\n"
            "The attention layer learns which time steps in the movement sequence are most important for diagnosis.\n\n"
            "📊 KEY VISUALIZATIONS:\n"
            "• Plot 1: Side-by-side comparison showing average attention for Healthy vs Patient groups\n"
            "  - Darker bars highlight the TOP 5 MOST DISCRIMINATIVE TIME STEPS\n"
            "  - Yellow annotations mark the top 3 most important steps\n"
            "  - Dashed line shows uniform baseline (what random attention would look like)\n\n"
            "• Plot 2 & 3: Heatmaps showing attention consistency within each group\n"
            "  - Gold vertical lines mark the most discriminative time steps\n"
            "  - Brighter colors = higher attention weight at that time step\n\n"
            "• Plot 4: Focused view of ONLY the top 5 most discriminative time steps\n"
            "  - Shows exactly where Healthy and Patient groups differ most\n"
            "  - These time steps are critical for the model's classification decision\n\n"
            "🔍 INTERPRETATION:\n"
            "• Time steps with high difference between groups = discriminative features\n"
            "• Patients often attend more to later time steps (symptom manifestation during task)\n"
            "• Healthy subjects show more uniform or early-focused attention patterns"
        )
        ttk.Label(info_frame, text=info_text, justify='left', wraplength=1300, font=('Courier', 9)).pack()
        
        # Test Set Analysis Section
        test_section = ttk.LabelFrame(main_frame, text="📊 Test Set Analysis (From Training)", padding="10")
        test_section.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.test_attention_frame = ttk.Frame(test_section)
        self.test_attention_frame.pack(fill='both', expand=True)
        
        self.test_attention_placeholder = ttk.Label(
            self.test_attention_frame,
            text="Test set attention analysis will appear here after training completes",
            font=('Helvetica', 11),
            foreground='gray'
        )
        self.test_attention_placeholder.pack(pady=30)
        
        # Prediction Analysis Section
        prediction_section = ttk.LabelFrame(main_frame, text="🔍 Individual Prediction Analysis", padding="10")
        prediction_section.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.prediction_attention_frame = ttk.Frame(prediction_section)
        self.prediction_attention_frame.pack(fill='both', expand=True)
        
        self.prediction_attention_placeholder = ttk.Label(
            self.prediction_attention_frame,
            text="Select a prediction from Tab 3 and click 'View Attention Analysis' to see individual analysis",
            font=('Helvetica', 11),
            foreground='gray'
        )
        self.prediction_attention_placeholder.pack(pady=30)
        
    def browse_healthy_folder(self):
        """Browse for healthy signals folder"""
        folder = filedialog.askdirectory(title="Select Healthy Signals Folder")
        if folder:
            self.healthy_path = folder
            self.healthy_path_var.set(folder)
            self.log_message(f"✓ Healthy signals folder selected: {folder}")
            self.check_ready_to_preprocess()
            
    def browse_patient_folder(self):
        """Browse for patient signals folder"""
        folder = filedialog.askdirectory(title="Select Patient Signals Folder")
        if folder:
            self.patient_path = folder
            self.patient_path_var.set(folder)
            self.log_message(f"✓ Patient signals folder selected: {folder}")
            self.check_ready_to_preprocess()
            
    def check_ready_to_preprocess(self):
        """Enable preprocessing button if both folders are selected"""
        if self.healthy_path and self.patient_path:
            self.preprocess_btn['state'] = 'normal'
            self.status_var.set("Ready to preprocess")
        else:
            self.preprocess_btn['state'] = 'disabled'
            
    def log_message(self, message):
        """Add message to log queue (thread-safe)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}\n")
        
    def train_log_message(self, message):
        """Add message to training log (thread-safe)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        def update():
            self.train_log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.train_log_text.see(tk.END)
        self.root.after(0, update)
        
    def process_log_queue(self):
        """Process messages from log queue and update GUI"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message)
                self.log_text.see(tk.END)
                self.log_text.update_idletasks()
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_log_queue)
            
    def clear_log(self):
        """Clear the log text area"""
        self.log_text.delete(1.0, tk.END)
        
    def start_preprocessing(self):
        """Start the preprocessing in a separate thread"""
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Processing", "Preprocessing is already running!")
            return
            
        self.preprocess_btn['state'] = 'disabled'
        self.status_var.set("Processing... Please wait")
        self.progress_var.set(0)
        
        self.processing_thread = threading.Thread(target=self.run_preprocessing, daemon=True)
        self.processing_thread.start()
        
    def run_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        try:
            self.log_message("="*60)
            self.log_message("STARTING PREPROCESSING PIPELINE")
            self.log_message("="*60)
            
            # Step 1: Load and parse files
            self.progress_var.set(10)
            self.log_message("\n[1/7] Loading and parsing signal files...")
            all_records, sigMea_records, sigSp_records = self.load_and_parse_files()
            self.log_message(f"✓ Loaded {len(all_records)} total files")
            
            # Step 2: Create DataFrames
            self.progress_var.set(20)
            self.log_message("\n[2/7] Creating DataFrames...")
            df_sigMea = pd.DataFrame(sigMea_records)
            df_sigSp = pd.DataFrame(sigSp_records)
            
            # Step 3: Data cleaning
            self.progress_var.set(30)
            self.log_message("\n[3/7] Cleaning data...")
            df_sigMea_clean, df_sigSp_clean = self.clean_data(df_sigMea, df_sigSp)
            
            # Step 4: Split data
            self.progress_var.set(40)
            self.log_message("\n[4/7] Splitting into train/test sets...")
            df_train, df_test = self.split_data(df_sigMea_clean, df_sigSp_clean)
            
            # Step 5: Data augmentation
            self.progress_var.set(50)
            self.log_message("\n[5/7] Applying data augmentation...")
            df_train_aug = self.augment_data(df_train)
            
            # Step 6: Feature extraction
            self.progress_var.set(70)
            self.log_message("\n[6/7] Extracting features...")
            df_features_train = self.extract_features(df_train_aug, "training")
            df_features_test = self.extract_features(df_test, "test")
            
            # Step 7: Add delta features and save
            self.progress_var.set(90)
            self.log_message("\n[7/7] Adding delta features and saving...")
            df_delta_train = self.add_delta_features(df_features_train)
            df_delta_test = self.add_delta_features(df_features_test)
            
            # Save to pickle files
            df_delta_train.to_pickle('df_train.pkl')
            df_delta_test.to_pickle('df_test.pkl')
            
            self.df_train = df_delta_train
            self.df_test = df_delta_test
            
            self.log_message(f"✓ Data saved successfully!")
            self.log_message(f"  - df_train.pkl: {df_delta_train.shape}")
            self.log_message(f"  - df_test.pkl: {df_delta_test.shape}")
            
            self.progress_var.set(100)
            self.log_message("\n" + "="*60)
            self.log_message("PREPROCESSING COMPLETED SUCCESSFULLY!")
            self.log_message("="*60)
            
            self.status_var.set("✓ Preprocessing completed!")
            
            # Enable training button
            self.root.after(0, lambda: self.train_btn.config(state='normal'))
            self.root.after(0, lambda: self.train_status_var.set("Ready to train model"))
            
            messagebox.showinfo("Success", "Preprocessing completed!\nYou can now proceed to Model Training.")
            
        except Exception as e:
            self.log_message(f"\n❌ ERROR: {str(e)}")
            import traceback
            self.log_message(f"Traceback:\n{traceback.format_exc()}")
            self.status_var.set("❌ Error during preprocessing")
            messagebox.showerror("Error", f"Preprocessing failed:\n{str(e)}")
            
        finally:
            self.preprocess_btn['state'] = 'normal'
            
    # [Include all preprocessing methods from previous version]
    def parse_signal_file(self, file_path, label):
        """Parse a single signal file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        meta_info = {}
        signal_start_idx = 0
        in_meta_section = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line == "#<meta>":
                in_meta_section = True
                continue
            elif line == "#</meta>":
                in_meta_section = False
                signal_start_idx = i + 1
                break
            if in_meta_section and line.startswith("#<") and ">" in line:
                try:
                    tag_start = line.find("<") + 1
                    tag_end = line.find(">")
                    if tag_start > 0 and tag_end > tag_start:
                        key = line[tag_start:tag_end]
                        value_start = tag_end + 1
                        closing_tag = f"</{key}>"
                        value_end = line.find(closing_tag)
                        if value_end > value_start:
                            value = line[value_start:value_end]
                        else:
                            value = ""
                        if value.isdigit():
                            value = int(value)
                        elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
                            value = float(value)
                        elif value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif value == '':
                            value = None
                        meta_info[key] = value
                except Exception:
                    continue
        
        signal_lines = lines[signal_start_idx:]
        signal_array = np.loadtxt(signal_lines, delimiter="\t")
        if signal_array.ndim == 1:
            signal_array = signal_array[1:6].reshape(1, -1)
        else:
            signal_array = signal_array[:, 1:6]
        
        return signal_array, label, meta_info
        
    def load_and_parse_files(self):
        """Load and parse all signal files"""
        all_records = []
        sigMea_records = []
        sigSp_records = []
        
        base_paths = {"healthy": self.healthy_path, "patient": self.patient_path}
        category_labels = {"healthy": 0, "patient": 1}
        metadata_fields = ['Person_ID_Number', 'Age', 'Gender', 'Writing_Hand']
        
        for category, folder_path in base_paths.items():
            label = category_labels[category]
            files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
            self.log_message(f"  Processing {len(files)} {category} files...")
            
            for file_name in files:
                file_path = os.path.join(folder_path, file_name)
                try:
                    signal_data, label_val, meta_info = self.parse_signal_file(file_path, label)
                    filtered_meta = {k: meta_info[k] for k in metadata_fields if k in meta_info}
                    record = {"file_name": file_name, "label": label_val, "signal": signal_data, **filtered_meta}
                    all_records.append(record)
                    if file_name.startswith("sigMea"):
                        sigMea_records.append(record)
                    elif file_name.startswith("sigSp"):
                        sigSp_records.append(record)
                except Exception as e:
                    self.log_message(f"  ⚠ Error parsing {file_name}: {e}")
                    
        return all_records, sigMea_records, sigSp_records
        
    def clean_data(self, df_sigMea, df_sigSp):
        """Clean the data"""
        df_sigMea_clean = df_sigMea.dropna(subset=['signal', 'label', 'file_name'])
        df_sigSp_clean = df_sigSp.dropna(subset=['signal', 'label', 'file_name'])
        df_sigMea_clean = df_sigMea_clean[df_sigMea_clean['signal'].apply(lambda x: np.all(x >= 0))]
        df_sigSp_clean = df_sigSp_clean[df_sigSp_clean['signal'].apply(lambda x: np.all(x >= 0))]
        return df_sigMea_clean, df_sigSp_clean
        
    def split_data(self, df_sigMea_clean, df_sigSp_clean):
        """Split data into train and test sets"""
        df_combined = pd.concat([df_sigMea_clean, df_sigSp_clean], ignore_index=True)
        person_labels = df_combined.drop_duplicates(subset='Person_ID_Number')[['Person_ID_Number', 'label']]
        train_ids, test_ids = train_test_split(
            person_labels['Person_ID_Number'],
            test_size=0.15,
            random_state=self.seed,
            stratify=person_labels['label']
        )
        df_train = df_combined[df_combined['Person_ID_Number'].isin(train_ids)]
        df_test = df_combined[df_combined['Person_ID_Number'].isin(test_ids)]
        self.log_message(f"  ✓ Train: {len(train_ids)} persons, Test: {len(test_ids)} persons")
        return df_train, df_test
        
    def augment_data(self, df_train):
        """Apply data augmentation"""
        def apply_jitter(signal, rng, noise_level=0.01):
            noise = rng.normal(0, noise_level, signal.shape)
            return signal + noise
        
        def apply_time_warp(signal, rng, target_length=None):
            orig_len = len(signal)
            if target_length is not None:
                warp_factor = target_length / orig_len * rng.uniform(0.95, 1.05)
            else:
                warp_factor = rng.uniform(0.8, 1.2)
            new_len = int(orig_len * warp_factor)
            old_time = np.arange(orig_len)
            new_time = np.linspace(0, orig_len - 1, new_len)
            warped_signal = np.zeros((new_len, signal.shape[1]))
            for i in range(signal.shape[1]):
                interpolator = interp.CubicSpline(old_time, signal[:, i])
                warped_signal[:, i] = interpolator(new_time)
            if target_length is None:
                final_time = np.linspace(0, new_len - 1, orig_len)
                final_signal = np.zeros((orig_len, signal.shape[1]))
                for i in range(signal.shape[1]):
                    interpolator = interp.CubicSpline(new_time, warped_signal[:, i])
                    final_signal[:, i] = interpolator(final_time)
                return final_signal
            return warped_signal
        
        signal_durations = df_train['signal'].apply(len)
        median_duration = signal_durations.median()
        std_duration = signal_durations.std()
        short_threshold = median_duration - 0.5 * std_duration
        long_threshold = median_duration + 0.5 * std_duration
        
        rng = np.random.RandomState(self.seed)
        augmented_records = []
        original_records = []
        
        for record in df_train.to_dict('records'):
            record['augmented'] = False
            original_records.append(record)
        
        total = len(df_train)
        for idx, row in df_train.iterrows():
            if idx % 50 == 0:
                self.log_message(f"  Augmenting: {idx}/{total} files...")
            original_signal = row['signal']
            signal_length = len(original_signal)
            
            if signal_length < short_threshold:
                n_augmentations = 2
                target_lengths = [int(signal_length * rng.uniform(1.1, 1.3)) for _ in range(n_augmentations)]
            elif signal_length > long_threshold:
                n_augmentations = min(3, int(signal_length / median_duration))
                target_lengths = [int(median_duration * rng.uniform(0.9, 1.1)) for _ in range(n_augmentations)]
            else:
                n_augmentations = 2
                target_lengths = [int(signal_length * rng.uniform(0.9, 1.1)) for _ in range(n_augmentations)]
            
            for target_length in target_lengths:
                new_record = row.copy()
                augmented_signal = original_signal.copy()
                noise_level = rng.uniform(0.005, 0.02)
                augmented_signal = apply_jitter(augmented_signal, rng, noise_level)
                augmented_signal = apply_time_warp(augmented_signal, rng, target_length=target_length)
                new_record['signal'] = augmented_signal
                new_record['augmented'] = True
                augmented_records.append(new_record)
        
        return pd.DataFrame(original_records + augmented_records)
        
    def extract_features(self, df, dataset_name):
        """Extract features from signals"""
        def compute_basic_stats(signal):
            return {
                "mean": np.mean(signal), "std": np.std(signal), "min": np.min(signal),
                "max": np.max(signal), "range": np.max(signal) - np.min(signal), "median": np.median(signal),
            }
        
        def compute_fft_features(signal, sampling_rate=1000):
            N = len(signal)
            if N == 0:
                return {}
            yf = np.fft.fft(signal)
            xf = np.fft.fftfreq(N, 1 / sampling_rate)
            yf_mag = np.abs(yf[0:N//2])
            xf_pos = xf[0:N//2]
            if len(yf_mag) == 0:
                return {}
            dominant_freq = xf_pos[np.argmax(yf_mag)] if len(xf_pos) > 0 else 0
            power_spectrum = yf_mag**2
            norm_power = power_spectrum / np.sum(power_spectrum) if np.sum(power_spectrum) > 0 else power_spectrum
            spectral_entropy = entropy(norm_power)
            energy_low = np.sum(power_spectrum[(xf_pos >= 0.1) & (xf_pos < 5)])
            energy_mid = np.sum(power_spectrum[(xf_pos >= 5) & (xf_pos < 20)])
            energy_high = np.sum(power_spectrum[(xf_pos >= 20) & (xf_pos < 50)])
            return {
                "fft_mean": np.mean(yf_mag), "fft_std": np.std(yf_mag),
                "fft_dominant_freq": dominant_freq, "fft_spectral_entropy": spectral_entropy,
                "fft_energy_low": energy_low, "fft_energy_mid": energy_mid, "fft_energy_high": energy_high,
            }
        
        def extract_window_features(window_data):
            features = {}
            for i, channel in enumerate(self.channel_names):
                signal = window_data[:, i]
                stats = compute_basic_stats(signal)
                for stat_name, stat_val in stats.items():
                    features[f"{channel}_{stat_name}"] = stat_val
                
                jerk = np.gradient(signal, 1/1000)
                snap = np.gradient(jerk, 1/1000)
                features[f"{channel}_jerk_mass"] = np.sum(np.abs(jerk))
                features[f"{channel}_snap_mass"] = np.sum(np.abs(snap))
                
                fft_features = compute_fft_features(signal)
                for feat_name, feat_val in fft_features.items():
                    features[f"{channel}_{feat_name}"] = feat_val
            return features
        
        window_size = 1000
        step_size = 500
        features = []
        total = len(df)
        
        for idx, row in df.iterrows():
            if idx % 50 == 0:
                self.log_message(f"  Extracting from {dataset_name}: {idx}/{total}...")
            signal = row["signal"]
            if signal.shape[0] < window_size:
                continue
            metadata = {k: v for k, v in row.items() if k not in ['signal', 'label', 'file_name']}
            for start in range(0, signal.shape[0] - window_size + 1, step_size):
                window_features = extract_window_features(signal[start:start+window_size, :])
                window_features.update({"label": row["label"], "file_name": row["file_name"], 
                                       "start_index": start, "end_index": start+window_size, **metadata})
                features.append(window_features)
        
        return pd.DataFrame(features)
        
    def add_delta_features(self, df):
        """Add delta features"""
        delta_features = []
        base_non_feature_cols = ['label', 'file_name', 'start_index', 'end_index', 'Person_ID_Number',
                                'Age', 'Gender', 'Writing_Hand', 'augmented', 'original_duration', 'target_duration']
        
        for file_name, group in df.groupby("file_name"):
            group = group.sort_values("start_index").reset_index(drop=True)
            group_delta = group.copy()
            feature_cols = [col for col in group.columns if col not in base_non_feature_cols]
            for col in feature_cols:
                group_delta[f"delta_{col}"] = group[col].diff()
            delta_cols = [f"delta_{col}" for col in feature_cols]
            group_delta.loc[0, delta_cols] = 0
            delta_features.append(group_delta)
        
        return pd.concat(delta_features, ignore_index=True)
        
    def start_training(self):
        """Start model training"""
        if self.df_train is None or self.df_test is None:
            messagebox.showerror("Error", "Please complete preprocessing first!")
            return
            
        self.train_btn['state'] = 'disabled'
        self.train_status_var.set("Training in progress...")
        self.train_progress_var.set(0)
        
        training_thread = threading.Thread(target=self.run_training, daemon=True)
        training_thread.start()
        
    def run_training(self):
        """Run the complete training pipeline"""
        try:
            self.train_log_message("="*60)
            self.train_log_message("STARTING MODEL TRAINING")
            self.train_log_message("="*60)
            
            # Load data
            self.train_progress_var.set(5)
            self.train_log_message("\n[1/6] Loading preprocessed data...")
            df_train = pd.read_pickle('df_train.pkl')
            df_test = pd.read_pickle('df_test.pkl')
            self.train_log_message(f"✓ Train: {len(df_train)}, Test: {len(df_test)}")
            
            # Create sequences
            self.train_progress_var.set(15)
            self.train_log_message("\n[2/6] Creating sequences...")
            X_train_final, y_train_seq, train_files, train_persons, X_test_final, y_test_seq, test_files, test_persons = self.create_sequences(df_train, df_test)
            self.train_log_message(f"✓ Train sequences: {X_train_final.shape}, Test sequences: {X_test_final.shape}")
            
            # Build model
            self.train_progress_var.set(25)
            self.train_log_message("\n[3/6] Building Attention-Based Bi-LSTM model...")
            model = self.create_bilstm_model((X_train_final.shape[1], X_train_final.shape[2]))
            self.train_log_message("✓ Model architecture created")
            
            # Train model
            self.train_progress_var.set(35)
            self.train_log_message("\n[4/6] Training model (13 epochs)...")
            history = model.fit(X_train_final, y_train_seq, epochs=13, batch_size=64, verbose=0)
            self.train_log_message("✓ Training completed")
            
            # Evaluate
            self.train_progress_var.set(70)
            self.train_log_message("\n[5/6] Evaluating on test set...")
            results = self.evaluate_model(model, X_test_final, y_test_seq, test_files, test_persons)
            self.train_log_message("✓ Evaluation completed")
            
            # Extract attention weights on test set
            self.train_log_message("\n[5.5/6] Extracting attention weights...")
            self.model = model  # Set model first
            self.attention_model = self.build_attention_extraction_model()
            if self.attention_model is not None:
                results['attention_weights'] = self.extract_attention_weights(X_test_final)
                results['attention_y_test'] = y_test_seq
                self.train_log_message("✓ Attention weights extracted for test set")
            
            # Save model
            self.train_progress_var.set(95)
            self.train_log_message("\n[6/6] Saving model and results...")
            model.save('final_model.h5')
            self.model = model
            self.results = results
            self.train_log_message("✓ Model saved to final_model.h5")
            
            self.train_progress_var.set(100)
            self.train_log_message("\n" + "="*60)
            self.train_log_message("TRAINING COMPLETED SUCCESSFULLY!")
            self.train_log_message("="*60)
            
            # Enable prediction tab
            self.root.after(0, lambda: self.model_status_var.set("✅ Model trained and ready for prediction"))
            self.root.after(0, lambda: self.browse_folder_btn.config(state='normal'))
            
            # Display results
            self.root.after(0, self.display_results)
            self.root.after(0, self.display_test_attention)
            
            self.train_status_var.set("✓ Training completed!")
            messagebox.showinfo("Success", "Model training completed!\nCheck the Results and Attention Analysis tabs.")
            
        except Exception as e:
            self.train_log_message(f"\n❌ ERROR: {str(e)}")
            import traceback
            self.train_log_message(f"Traceback:\n{traceback.format_exc()}")
            self.train_status_var.set("❌ Error during training")
            messagebox.showerror("Error", f"Training failed:\n{str(e)}")
            
        finally:
            self.train_btn['state'] = 'normal'
            
    def create_sequences(self, df_train, df_test):
        base_non_feature_cols = ['label', 'file_name', 'start_index', 'end_index', 'Person_ID_Number',
                                'Age', 'Gender', 'Writing_Hand', 'augmented', 'original_duration', 'target_duration']
        feature_cols = [col for col in df_train.columns if col not in base_non_feature_cols]
        feature_cols = [col for col in feature_cols if np.issubdtype(df_train[col].dtype, np.number)]
        
        SEQUENCE_LENGTH = 20
        STEP_SIZE = 10
        
        def create_seq(df, dataset_name):
            sequences, labels, files, persons = [], [], [], []
            skipped_files = []  # Track skipped files
            
            grouped = df.groupby('file_name')
            self.train_log_message(f"Creating sequences of length {SEQUENCE_LENGTH} with step {STEP_SIZE} for {dataset_name}...")
            
            for file_name, group in grouped:
                features = group[feature_cols].values
                n_windows = len(features)
                
                # Check if enough windows for sequence creation
                if n_windows < SEQUENCE_LENGTH:
                    skipped_files.append({
                        'file': file_name,
                        'windows': n_windows,
                        'needed': SEQUENCE_LENGTH,
                        'person_id': group['Person_ID_Number'].iloc[0],
                        'label': 'Healthy' if group['label'].iloc[0] == 0 else 'Patient'
                    })
                    continue
                
                label = group['label'].iloc[0]
                person_id = group['Person_ID_Number'].iloc[0]
                
                for i in range(0, n_windows - SEQUENCE_LENGTH + 1, STEP_SIZE):
                    sequences.append(features[i:i+SEQUENCE_LENGTH])
                    labels.append(label)
                    files.append(file_name)
                    persons.append(person_id)
            
            # Report skipped files (matching training notebook style)
            if skipped_files:
                self.train_log_message(f"\n{'='*70}")
                self.train_log_message(f"⚠️ SKIPPED {len(skipped_files)} FILES in {dataset_name} (Insufficient Windows)")
                self.train_log_message(f"{'='*70}")
                
                skipped_df = pd.DataFrame(skipped_files)
                self.train_log_message("\nBreakdown by class:")
                class_counts = skipped_df.groupby('label').size()
                for label_name, count in class_counts.items():
                    self.train_log_message(f"  {label_name}: {count} files")
                
                # Show sample or all skipped files
                if len(skipped_files) <= 10:
                    self.train_log_message("\nSkipped files:")
                    for sf in skipped_files:
                        self.train_log_message(f"  • {sf['file']}: {sf['windows']} windows (need {sf['needed']})")
                else:
                    self.train_log_message("\nSample of skipped files (first 10):")
                    for sf in skipped_files[:10]:
                        self.train_log_message(f"  • {sf['file']}: {sf['windows']} windows (need {sf['needed']})")
                
                # Calculate minimum samples needed
                min_samples = (SEQUENCE_LENGTH - 1) * 500 + 1000
                self.train_log_message(f"\n⚠️ Note: Files need at least {SEQUENCE_LENGTH} windows "
                                    f"(~{min_samples} samples) to create sequences.")
            else:
                self.train_log_message(f"✅ All files in {dataset_name} had sufficient windows for sequence creation")
            
            return np.array(sequences), np.array(labels), np.array(files), np.array(persons)
        
        # Create sequences for both datasets
        X_train, y_train, train_files, train_persons = create_seq(df_train, "training set")
        X_test, y_test, test_files, test_persons = create_seq(df_test, "test set")
        
        self.train_log_message(f"\n✅ Final sequence shapes:")
        self.train_log_message(f"  Training: {X_train.shape}")
        self.train_log_message(f"  Test: {X_test.shape}")
        
        # Normalize
        n_features = X_train.shape[2]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
        
        self.scaler = scaler
        with open('final_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        return X_train_scaled, y_train, train_files, train_persons, X_test_scaled, y_test, test_files, test_persons

    def create_bilstm_model(self, input_shape):
        """Create Attention-Based BiLSTM model"""
        inputs = Input(shape=input_shape)
        x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.0001)))(inputs)
        x = Dropout(0.4)(x)
        x = Attention()(x)
        x = Dense(16, activation='relu', kernel_regularizer=l2(0.0001))(x)
        x = Dropout(0.4)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        return model
        
    def evaluate_model(self, model, X_test, y_test, test_files, test_persons):
        """Evaluate model at multiple levels"""
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        results = {}
        
        # Sequence level
        results['sequence'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob.flatten(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # File level
        df_pred = pd.DataFrame({
            "file_name": test_files,
            "true_label": y_test,
            "pred_prob": y_pred_prob.flatten()
        })
        file_prob = df_pred.groupby("file_name")["pred_prob"].mean()
        file_pred = (file_prob >= 0.5).astype(int)
        file_true = df_pred.groupby("file_name")["true_label"].first()
        
        results['file'] = {
            'accuracy': accuracy_score(file_true, file_pred),
            'y_true': file_true.values,
            'y_pred': file_pred.values,
            'y_pred_prob': file_prob.values,
            'classification_report': classification_report(file_true, file_pred, output_dict=True)
        }
        
        # Person level
        df_pred_person = pd.DataFrame({
            "person_id": test_persons,
            "true_label": y_test,
            "pred_prob": y_pred_prob.flatten()
        })
        person_prob = df_pred_person.groupby("person_id")["pred_prob"].mean()
        person_pred = (person_prob >= 0.5).astype(int)
        person_true = df_pred_person.groupby("person_id")["true_label"].first()
        
        results['person'] = {
            'accuracy': accuracy_score(person_true, person_pred),
            'y_true': person_true.values,
            'y_pred': person_pred.values,
            'y_pred_prob': person_prob.values,
            'classification_report': classification_report(person_true, person_pred, output_dict=True)
        }
        
        # Log metrics
        self.train_log_message(f"\n📊 EVALUATION RESULTS:")
        self.train_log_message(f"  Sequence-level Accuracy: {results['sequence']['accuracy']:.4f}")
        self.train_log_message(f"  File-level Accuracy: {results['file']['accuracy']:.4f}")
        self.train_log_message(f"  Person-level Accuracy: {results['person']['accuracy']:.4f}")
        
        return results
    
    def build_attention_extraction_model(self):
        """Build model to extract attention weights"""
        if self.model is None:
            return None
            
        try:
            # Find trained attention layer
            trained_attention = None
            for layer in self.model.layers:
                if isinstance(layer, Attention):
                    trained_attention = layer
                    break
            
            if trained_attention is None:
                self.train_log_message("⚠ No attention layer found in model")
                return None
            
            # Build extraction model
            inputs = Input(shape=self.model.input_shape[1:])
            
            # Get BiLSTM layer
            bilstm_layer = self.model.layers[1]
            x = bilstm_layer(inputs)
            
            # Get Dropout layer
            dropout_layer = self.model.layers[2]
            x = dropout_layer(x, training=False)
            
            # Create attention layer that returns weights
            attention_with_weights = AttentionWithWeights()
            attention_output, attention_weights = attention_with_weights(x)
            
            # Build model
            attention_model = Model(inputs=inputs, outputs=attention_weights)
            
            # Copy trained weights
            attention_with_weights.set_weights(trained_attention.get_weights())
            
            self.train_log_message("✅ Attention extraction model built")
            return attention_model
            
        except Exception as e:
            self.train_log_message(f"⚠ Error building attention model: {e}")
            return None
    
    def extract_attention_weights(self, sequences):
        """Extract attention weights for given sequences"""
        if self.attention_model is None:
            self.attention_model = self.build_attention_extraction_model()
        
        if self.attention_model is None:
            return None
        
        try:
            weights = self.attention_model.predict(sequences, verbose=0)
            return weights.squeeze()
        except Exception as e:
            print(f"Error extracting attention: {e}")
            return None
    
    def get_responsive_figsize(self, default_width=14, default_height=10):
        """Calculate responsive figure size based on window width"""
        try:
            # Get current window width in pixels
            window_width = self.root.winfo_width()
            
            # Calculate available width (accounting for padding, scrollbar, etc.)
            available_width = window_width - 100  # Leave margin for scrollbar and padding
            
            # Convert pixels to inches (assuming 80 DPI)
            width_inches = available_width / 80
            
            # Cap at maximum and minimum sizes
            width_inches = max(10, min(width_inches, 18))
            
            # Maintain aspect ratio
            aspect_ratio = default_height / default_width
            height_inches = width_inches * aspect_ratio
            
            return (width_inches, height_inches)
        except:
            # Fallback to default size if error
            return (default_width, default_height)
    
    def visualize_attention_weights(self, attention_weights, y_true=None, title="Attention Weights", figsize=None):
        """Create attention visualization plots matching final training code with responsive sizing"""
        # Use provided figsize or calculate responsive size
        if figsize is None:
            figsize = self.get_responsive_figsize(14, 10 if y_true is not None else 6)
        
        fig = Figure(figsize=figsize, dpi=80)
        
        if y_true is not None:
            # ===== EXACTLY MATCHING FINAL TRAINING CODE =====
            # Group by class
            healthy_weights = attention_weights[y_true == 0]
            patient_weights = attention_weights[y_true == 1]
            
            avg_healthy = healthy_weights.mean(axis=0)
            avg_patient = patient_weights.mean(axis=0)
            std_healthy = healthy_weights.std(axis=0)
            std_patient = patient_weights.std(axis=0)
            
            # Calculate difference and identify important time steps
            attention_diff = avg_patient - avg_healthy
            uniform_baseline = 1 / len(avg_healthy)
            
            # Get top 5 most discriminative time steps
            top_k = 5
            diff_importance = np.abs(attention_diff)
            top_diff_indices = np.argsort(diff_importance)[-top_k:][::-1]
            
            # Calculate patient importance
            patient_importance = np.abs(avg_patient - uniform_baseline)
            top_patient_indices = np.argsort(patient_importance)[-top_k:][::-1]
            
            # Print important time steps to console
            print("\n" + "="*70)
            print(f"📊 TOP {top_k} TIME STEPS WITH LARGEST ATTENTION DIFFERENCE:")
            print("="*70)
            for rank, idx in enumerate(top_diff_indices, 1):
                time_step = idx + 1
                print(f"{rank}. Time Step {time_step:2d}: "
                      f"Healthy={avg_healthy[idx]:.4f}, "
                      f"Patient={avg_patient[idx]:.4f}, "
                      f"Diff={attention_diff[idx]:+.4f}")
            
            print(f"\n📊 TOP {top_k} TIME STEPS MOST ATTENDED BY PATIENTS:")
            print("="*70)
            for rank, idx in enumerate(top_patient_indices, 1):
                time_step = idx + 1
                print(f"{rank}. Time Step {time_step:2d}: "
                      f"Patient={avg_patient[idx]:.4f} "
                      f"(vs Uniform={uniform_baseline:.4f}, "
                      f"+{(avg_patient[idx]-uniform_baseline)*100:.1f}%)")
            
            # VISUALIZATION 1: SIDE-BY-SIDE COMPARISON WITH HIGHLIGHTED IMPORTANT STEPS
            ax1 = fig.add_subplot(2, 2, 1)
            x = np.arange(1, len(avg_healthy) + 1)
            width = 0.35
            
            # Plot all bars first
            bars_healthy = ax1.bar(x - width/2, avg_healthy, width, 
                                   label='Healthy', color='#2ecc71', alpha=0.7, edgecolor='black')
            bars_patient = ax1.bar(x + width/2, avg_patient, width, 
                                   label='Patient', color='#e74c3c', alpha=0.7, edgecolor='black')
            
            # Highlight top discriminative time steps (darker bars)
            for idx in top_diff_indices:
                bars_healthy[idx].set_color('#27ae60')
                bars_healthy[idx].set_alpha(1.0)
                bars_healthy[idx].set_edgecolor('darkgreen')
                bars_healthy[idx].set_linewidth(2.5)
                
                bars_patient[idx].set_color('#c0392b')
                bars_patient[idx].set_alpha(1.0)
                bars_patient[idx].set_edgecolor('darkred')
                bars_patient[idx].set_linewidth(2.5)
            
            # Add uniform baseline
            ax1.axhline(uniform_baseline, color='black', linestyle='--', 
                       linewidth=2, label=f'Uniform ({uniform_baseline:.4f})', zorder=0)
            
            # Mark top 3 with annotations
            for rank, idx in enumerate(top_diff_indices[:3], 1):
                time_step = idx + 1
                max_height = max(avg_healthy[idx], avg_patient[idx])
                ax1.annotate(f'#{rank}', 
                           xy=(time_step, max_height + 0.003),
                           ha='center', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            ax1.set_xlabel('Time Step', fontsize=11)
            ax1.set_ylabel('Attention Weight', fontsize=11)
            ax1.set_title('Attention Weights: Healthy vs Patient (Test Set)\n(Darker bars = Top 5 Most Discriminative)', 
                         fontweight='bold', fontsize=12)
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(alpha=0.3, axis='y')
            
            # VISUALIZATION 2: HEATMAP - HEALTHY
            ax2 = fig.add_subplot(2, 2, 2)
            sample_size = min(50, len(healthy_weights))
            im2 = ax2.imshow(healthy_weights[:sample_size], cmap='Greens', aspect='auto')
            ax2.set_xlabel('Time Step', fontsize=11)
            ax2.set_ylabel('Sample', fontsize=11)
            ax2.set_title(f'Healthy Class - Attention Heatmap (n={sample_size})', fontweight='bold', fontsize=12)
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('Attention Weight', fontsize=10)
            
            # Highlight important time steps
            for idx in top_diff_indices:
                ax2.axvline(idx, color='gold', linestyle='--', alpha=0.7, linewidth=1.5)
            
            # VISUALIZATION 3: HEATMAP - PATIENT  
            ax3 = fig.add_subplot(2, 2, 3)
            sample_size = min(50, len(patient_weights))
            im3 = ax3.imshow(patient_weights[:sample_size], cmap='Reds', aspect='auto')
            ax3.set_xlabel('Time Step', fontsize=11)
            ax3.set_ylabel('Sample', fontsize=11)
            ax3.set_title(f'Patient Class - Attention Heatmap (n={sample_size})', fontweight='bold', fontsize=12)
            cbar3 = plt.colorbar(im3, ax=ax3)
            cbar3.set_label('Attention Weight', fontsize=10)
            
            # Highlight important time steps
            for idx in top_diff_indices:
                ax3.axvline(idx, color='gold', linestyle='--', alpha=0.7, linewidth=1.5)
            
            # VISUALIZATION 4: FOCUSED VIEW - TOP DISCRIMINATIVE TIME STEPS
            ax4 = fig.add_subplot(2, 2, 4)
            
            # Show only top time steps
            top_indices_sorted = sorted(top_diff_indices)
            x_pos = np.arange(len(top_indices_sorted))
            width = 0.35
            
            healthy_vals = [avg_healthy[i] for i in top_indices_sorted]
            patient_vals = [avg_patient[i] for i in top_indices_sorted]
            labels = [f'Step {i+1}' for i in top_indices_sorted]
            
            bars1 = ax4.bar(x_pos - width/2, healthy_vals, width, 
                           label='Healthy', color='#27ae60', alpha=0.8, edgecolor='darkgreen', linewidth=2)
            bars2 = ax4.bar(x_pos + width/2, patient_vals, width, 
                           label='Patient', color='#c0392b', alpha=0.8, edgecolor='darkred', linewidth=2)
            
            ax4.axhline(uniform_baseline, color='black', linestyle='--', 
                       linewidth=2, label='Uniform', zorder=0)
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(labels, fontsize=10)
            ax4.set_xlabel('Time Step', fontsize=11)
            ax4.set_ylabel('Attention Weight', fontsize=11)
            ax4.set_title(f'Top {top_k} Most Discriminative Time Steps: Healthy vs Patient', 
                         fontweight='bold', fontsize=12)
            ax4.legend(fontsize=10)
            ax4.grid(alpha=0.3, axis='y')
            
        else:
            # Single sample visualization (individual prediction) - ONLY SHOW ATTENTION FLOW
            # Calculate average attention
            avg_attention = attention_weights.mean(axis=0) if attention_weights.ndim > 1 else attention_weights
            x = np.arange(1, len(avg_attention) + 1)
            uniform_baseline = 1/len(avg_attention)
            
            # Identify top 5 most attended time steps for printing
            top_k = 5
            top_indices = np.argsort(avg_attention)[-top_k:][::-1]
            
            # Print top time steps
            print("\n" + "="*70)
            print(f"📊 TOP {top_k} MOST ATTENDED TIME STEPS (INDIVIDUAL):")
            print("="*70)
            for rank, idx in enumerate(top_indices, 1):
                time_step = idx + 1
                print(f"{rank}. Time Step {time_step:2d}: Weight={avg_attention[idx]:.4f} "
                      f"(vs Uniform={uniform_baseline:.4f}, "
                      f"+{(avg_attention[idx]-uniform_baseline)*100:.1f}%)")
            
            # Create single plot - Attention Flow Over Time
            ax = fig.add_subplot(1, 1, 1)
            
            ax.plot(x, avg_attention, marker='o', color='#e74c3c', 
                    linewidth=2.5, markersize=7, label='Attention', zorder=3)
            ax.axhline(uniform_baseline, color='black', linestyle='--', 
                       label=f'Uniform ({uniform_baseline:.4f})', linewidth=2, zorder=2)
            ax.fill_between(x, avg_attention, uniform_baseline, 
                            where=(avg_attention > uniform_baseline),
                            alpha=0.3, color='#e74c3c', label='Above Uniform', zorder=1)
            
            # Annotate top 3 time steps
            for rank, idx in enumerate(top_indices[:3], 1):
                ax.annotate(f'#{rank}', 
                           xy=(idx+1, avg_attention[idx]),
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8, edgecolor='orange'),
                           arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))
            
            ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel('Attention Weight', fontsize=12)
            ax.set_title('Attention Flow Over Time\n(Shows which time steps the model focused on)', 
                        fontweight='bold', fontsize=14)
            ax.legend(fontsize=11, loc='best')
            ax.grid(alpha=0.3, linestyle=':', linewidth=0.8)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        fig.tight_layout()
        
        return fig
    
    def browse_prediction_folder(self):
        """Browse and add folder containing signal files to predict"""
        folder = filedialog.askdirectory(title="Select Folder with Signal Files")
        if folder:
            # Check if folder already added
            if folder in self.prediction_folders:
                messagebox.showinfo("Already Added", "This folder has already been added!")
                return
            
            # Count .txt files in folder
            txt_files = [f for f in os.listdir(folder) if f.endswith('.txt')]
            if len(txt_files) == 0:
                messagebox.showwarning("No Files", "No .txt files found in selected folder!")
                return
            
            # Add to list
            self.prediction_folders.append(folder)
            self.folder_listbox.insert(tk.END, f"{os.path.basename(folder)} ({len(txt_files)} files)")
            
            # Enable buttons
            self.predict_btn['state'] = 'normal'
            self.clear_folders_btn['state'] = 'normal'
    
    def clear_prediction_folders(self):
        """Clear all selected folders"""
        self.prediction_folders = []
        self.folder_listbox.delete(0, tk.END)
        self.predict_btn['state'] = 'disabled'
        self.clear_folders_btn['state'] = 'disabled'
    
    def start_prediction(self):
        """Start person-level prediction with attention analysis"""
        if not hasattr(self, 'prediction_folders') or len(self.prediction_folders) == 0:
            messagebox.showerror("Error", "Please add at least one folder first!")
            return
        
        # Check if model and scaler are available
        if self.model is None:
            messagebox.showerror("Error", "No model loaded! Please train a model first.")
            return
        
        if self.scaler is None:
            messagebox.showerror("Error", "No scaler found! Please train a model first.")
            return
        
        self.predict_btn['state'] = 'disabled'
        self.browse_folder_btn['state'] = 'disabled'
        self.clear_folders_btn['state'] = 'disabled'
        self.predict_status_var.set("Processing files for person-level prediction...")
        self.predict_progress_var.set(0)
        
        thread = threading.Thread(target=self.run_prediction, daemon=True)
        thread.start()
    
    def run_prediction(self):
        """Run person-level prediction on uploaded folders"""
        try:
            # Collect all .txt files from all folders
            all_files_with_paths = []
            for folder in self.prediction_folders:
                folder_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.txt')]
                all_files_with_paths.extend(folder_files)
            
            total_files = len(all_files_with_paths)
            
            # Store file-level data
            file_data = []
            
            # Process each file
            for idx, file_path in enumerate(all_files_with_paths):
                file_name = os.path.basename(file_path)
                
                try:
                    # Parse file and extract metadata
                    signal, _, meta_info = self.parse_signal_file(file_path, label=None)
                    
                    # Extract Person_ID from metadata
                    person_id = meta_info.get('Person_ID_Number', 'Unknown')
                    
                    # Check minimum length
                    if signal.shape[0] < 1000:
                        file_data.append({
                            'file_name': file_name,
                            'person_id': person_id,
                            'status': 'too_short',
                            'sequences': None,
                            'predictions': None,
                            'attention_weights': None,
                            'error': f'Signal too short: {signal.shape[0]} samples'
                        })
                        self.root.after(0, lambda v=(idx + 1) / total_files * 100: self.predict_progress_var.set(v))
                        continue
                    
                    # Extract features
                    features = self.extract_features_from_signal(signal)
                    if features is None:
                        file_data.append({
                            'file_name': file_name,
                            'person_id': person_id,
                            'status': 'error',
                            'sequences': None,
                            'predictions': None,
                            'attention_weights': None,
                            'error': 'Feature extraction failed'
                        })
                        self.root.after(0, lambda v=(idx + 1) / total_files * 100: self.predict_progress_var.set(v))
                        continue
                    
                    # Create sequences
                    sequences = self.create_sequences_from_features(features)
                    if sequences is None or len(sequences) == 0:
                        file_data.append({
                            'file_name': file_name,
                            'person_id': person_id,
                            'status': 'too_short',
                            'sequences': None,
                            'predictions': None,
                            'attention_weights': None,
                            'error': 'Not enough windows for sequence'
                        })
                        self.root.after(0, lambda v=(idx + 1) / total_files * 100: self.predict_progress_var.set(v))
                        continue
                    
                    # Predict and extract attention
                    predictions = self.model.predict(sequences, verbose=0)
                    attention_weights = self.extract_attention_weights(sequences)
                    
                    file_data.append({
                        'file_name': file_name,
                        'person_id': person_id,
                        'status': 'success',
                        'sequences': sequences,
                        'predictions': predictions,
                        'attention_weights': attention_weights,
                        'error': None
                    })
                    
                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    file_data.append({
                        'file_name': file_name,
                        'person_id': 'Unknown',
                        'status': 'error',
                        'sequences': None,
                        'predictions': None,
                        'attention_weights': None,
                        'error': str(e)[:100]
                    })
                
                self.root.after(0, lambda v=(idx + 1) / total_files * 100: self.predict_progress_var.set(v))
            
            # Aggregate at person level
            self.prediction_results = self.aggregate_person_level(file_data)
            
            # Update UI
            self.root.after(0, self.display_predictions)
            num_persons = len(self.prediction_results)
            self.root.after(0, lambda: self.predict_status_var.set(
                f"✅ Predicted {num_persons} person(s) from {total_files} file(s)"))
            
        except Exception as e:
            error_msg = f"Error during prediction:\n{str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Prediction Error", error_msg))
        finally:
            self.root.after(0, lambda: self.predict_btn.config(state='normal'))
            self.root.after(0, lambda: self.browse_folder_btn.config(state='normal'))
            self.root.after(0, lambda: self.clear_folders_btn.config(state='normal'))
    
    def aggregate_person_level(self, file_data):
        """Aggregate file-level predictions to person-level"""
        from collections import defaultdict
        
        # Group by person_id
        person_groups = defaultdict(list)
        for file_info in file_data:
            person_groups[file_info['person_id']].append(file_info)
        
        person_results = []
        
        for person_id, files in person_groups.items():
            # Collect predictions and attention weights from successful files
            all_predictions = []
            all_attention_weights = []
            successful_files = 0
            error_files = []
            
            for file_info in files:
                if file_info['status'] == 'success':
                    all_predictions.append(file_info['predictions'])
                    if file_info['attention_weights'] is not None:
                        all_attention_weights.append(file_info['attention_weights'])
                    successful_files += 1
                else:
                    error_files.append(file_info['file_name'])
            
            # Calculate person-level prediction
            if len(all_predictions) > 0:
                # Ensure all predictions have consistent shape before concatenation
                cleaned_predictions = []
                for pred in all_predictions:
                    if pred.ndim == 1:
                        pred = pred.reshape(-1, 1)
                    cleaned_predictions.append(pred)
                
                # Concatenate all predictions from all files
                all_preds_concat = np.concatenate(cleaned_predictions, axis=0)
                person_prob = all_preds_concat.mean()
                person_prediction = "Patient (PD)" if person_prob > 0.5 else "Healthy (Control)"
                confidence = abs(person_prob - 0.5) * 2 * 100
                
                # Concatenate attention weights from all files
                if len(all_attention_weights) > 0:
                    # Ensure consistent shape for attention weights
                    cleaned_attention = []
                    for attn in all_attention_weights:
                        if attn is not None:
                            # Ensure 2D shape (num_sequences, num_timesteps)
                            if attn.ndim == 1:
                                attn = attn.reshape(1, -1)
                            cleaned_attention.append(attn)
                    
                    if len(cleaned_attention) > 0:
                        person_attention = np.concatenate(cleaned_attention, axis=0)
                    else:
                        person_attention = None
                else:
                    person_attention = None
                
                person_results.append({
                    'person_id': person_id,
                    'num_files': len(files),
                    'successful_files': successful_files,
                    'prediction': person_prediction,
                    'probability': float(person_prob),
                    'confidence': f"{confidence:.1f}%",
                    'attention_weights': person_attention,
                    'error_files': error_files
                })
            else:
                # All files failed
                error_summary = "; ".join(error_files[:3])
                if len(error_files) > 3:
                    error_summary += f" (+{len(error_files)-3} more)"
                
                person_results.append({
                    'person_id': person_id,
                    'num_files': len(files),
                    'successful_files': 0,
                    'prediction': 'Error',
                    'probability': 0.0,
                    'confidence': f'All files failed',
                    'attention_weights': None,
                    'error_files': error_files
                })
        
        return person_results
    
    def extract_features_from_signal(self, signal):
        """Extract features from a single signal for prediction"""
        if signal.shape[0] < 1000:
            return None
        
        def compute_basic_stats(sig):
            return {"mean": np.mean(sig), "std": np.std(sig), "min": np.min(sig),
                   "max": np.max(sig), "range": np.max(sig) - np.min(sig), "median": np.median(sig)}
        
        def compute_fft_features(sig):
            N = len(sig)
            if N == 0: return {}
            yf, xf = np.fft.fft(sig), np.fft.fftfreq(N, 1/1000)
            yf_mag, xf_pos = np.abs(yf[0:N//2]), xf[0:N//2]
            if len(yf_mag) == 0: return {}
            power_spectrum = yf_mag**2
            norm_power = power_spectrum / np.sum(power_spectrum) if np.sum(power_spectrum) > 0 else power_spectrum
            return {"fft_mean": np.mean(yf_mag), "fft_std": np.std(yf_mag),
                   "fft_dominant_freq": xf_pos[np.argmax(yf_mag)] if len(xf_pos) > 0 else 0,
                   "fft_spectral_entropy": entropy(norm_power),
                   "fft_energy_low": np.sum(power_spectrum[(xf_pos >= 0.1) & (xf_pos < 5)]),
                   "fft_energy_mid": np.sum(power_spectrum[(xf_pos >= 5) & (xf_pos < 20)]),
                   "fft_energy_high": np.sum(power_spectrum[(xf_pos >= 20) & (xf_pos < 50)])}
        
        features = []
        for start in range(0, signal.shape[0] - 1000 + 1, 500):
            window_data = signal[start:start+1000, :]
            window_features = []
            for i, channel in enumerate(self.channel_names):
                sig = window_data[:, i]
                stats = compute_basic_stats(sig)
                jerk, snap = np.gradient(sig, 1/1000), np.gradient(np.gradient(sig, 1/1000), 1/1000)
                fft = compute_fft_features(sig)
                
                for k, v in stats.items():
                    window_features.append(v)
                window_features.extend([np.sum(np.abs(jerk)), np.sum(np.abs(snap))])
                for k, v in fft.items():
                    window_features.append(v)
            features.append(window_features)
        
        # Add delta features
        features_array = np.array(features)
        delta_features = np.diff(features_array, axis=0, prepend=features_array[0:1])
        combined_features = np.concatenate([features_array, delta_features], axis=1)
        
        return combined_features
    
    def create_sequences_from_features(self, features):
        """Create sequences from extracted features"""
        sequences = []
        for i in range(0, len(features) - 20 + 1, 10):
            sequences.append(features[i:i+20])
        
        if len(sequences) == 0:
            # Return None to indicate failure rather than empty array
            return None
        
        sequences = np.array(sequences)
        # Normalize using saved scaler
        sequences_scaled = self.scaler.transform(sequences.reshape(-1, sequences.shape[2])).reshape(sequences.shape)
        return sequences_scaled
    
    def display_predictions(self):
        """Display person-level prediction results in treeview"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        for result in self.prediction_results:
            self.results_tree.insert('', 'end', values=(
                result['person_id'],
                result['num_files'],
                result['prediction'],
                f"{result['probability']:.4f}",
                result['confidence']
            ))
        
        # Enable export buttons when predictions exist
        has_preds = bool(self.prediction_results)
        self.export_btn['state'] = 'normal' if has_preds else 'disabled'
        # PDF export should match CSV logic
        try:
            self.export_pdf_btn['state'] = 'normal' if has_preds else 'disabled'
        except Exception:
            pass
        self.view_attention_btn['state'] = 'normal' if has_preds else 'disabled'
    
    def export_predictions(self):
        """Export predictions to CSV"""
        if not self.prediction_results:
            messagebox.showwarning("No Data", "No predictions to export!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Predictions As"
        )
        
        if file_path:
            # Export person-level results without attention weights and error_files
            export_data = [{k: v for k, v in r.items() if k not in ['attention_weights', 'error_files']} 
                          for r in self.prediction_results]
            df_results = pd.DataFrame(export_data)
            df_results.to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"Person-level predictions exported to:\n{file_path}")

    def export_predictions_pdf(self):
        """Export predictions and attention charts to a PDF report."""
        if not self.prediction_results:
            messagebox.showwarning("No Data", "No predictions to export!")
            return

        # Lazy import and dependency check for fpdf
        try:
            from fpdf import FPDF
        except Exception:
            messagebox.showerror("Missing Dependency",
                                 "The package 'fpdf2' is required to export PDF reports.\nInstall with: pip install fpdf2")
            return

        import tempfile
        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            title="Save PDF Report As"
        )
        if not file_path:
            return

        # Prepare table data (exclude heavy fields)
        export_data = [{k: v for k, v in r.items() if k not in ['attention_weights', 'error_files']} for r in self.prediction_results]
        df_results = pd.DataFrame(export_data)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Parkinson's Disease Detection Report", ln=True, align='C')
        pdf.ln(5)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(4)

        # Add predictions table header
        pdf.set_font("Arial", 'B', 11)
        columns = ['person_id', 'num_files', 'prediction', 'probability', 'confidence']
        # Increase first column width to accommodate long person IDs
        col_widths = [60, 22, 55, 28, 28]
        for i, col in enumerate(columns):
            pdf.cell(col_widths[i], 8, str(col), border=1)
        pdf.ln()

        pdf.set_font("Arial", '', 10)
        for r in self.prediction_results:
            row_vals = [str(r.get('person_id', '')), str(r.get('num_files', '')), str(r.get('prediction', '')),
                        f"{r.get('probability', 0):.4f}", str(r.get('confidence', ''))]
            for i, val in enumerate(row_vals):
                # Truncate long values to avoid overflow
                pdf.cell(col_widths[i], 7, val[:40], border=1)
            pdf.ln()

        pdf.ln(6)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Attention Weights (per person)", ln=True)
        pdf.ln(3)

        # Add attention plots per person (average attention across sequences)
        for r in self.prediction_results:
            pid = r.get('person_id', 'Unknown')
            att = r.get('attention_weights', None)
            if att is None:
                continue

            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 6, f"Person: {pid}    Prediction: {r.get('prediction', '')}", ln=True)

            try:
                arr = np.array(att)
                if arr.ndim > 1:
                    avg_att = arr.mean(axis=0)
                else:
                    avg_att = arr

                fig, ax = plt.subplots(figsize=(6, 2.5))
                ax.plot(avg_att, color='tab:blue', linewidth=1.5)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Attention')
                ax.set_title('Average Attention Weights')
                ax.grid(alpha=0.3)
                fig.tight_layout()

                tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                plt.savefig(tmp.name, dpi=150)
                plt.close(fig)
                tmp.close()

                # Insert image into PDF, leaving margins
                try:
                    pdf.image(tmp.name, w=pdf.w - 30)
                except Exception:
                    # If insertion fails, skip
                    pass
                finally:
                    try:
                        os.unlink(tmp.name)
                    except Exception:
                        pass

                pdf.ln(4)
            except Exception as e:
                # continue on plotting errors
                print(f"Warning: unable to plot attention for {pid}: {e}")
                continue

        try:
            pdf.output(file_path)
            messagebox.showinfo("Success", f"PDF report exported to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("PDF Export Error", f"Failed to export PDF:\n{e}")
    
    def on_prediction_select(self, event):
        """Handle selection of person prediction result"""
        selection = self.results_tree.selection()
        if selection:
            # Check if selected person has valid attention data
            item = self.results_tree.item(selection[0])
            person_id = item['values'][0]
            
            # Find result for this person
            has_attention = False
            for result in self.prediction_results:
                if result['person_id'] == person_id and result['attention_weights'] is not None:
                    has_attention = True
                    break
            
            self.view_attention_btn['state'] = 'normal' if has_attention else 'disabled'
        else:
            self.view_attention_btn['state'] = 'disabled'
    
    def show_selected_attention(self):
        """Show attention for selected person prediction"""
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a person from the table first!")
            return
        
        item = self.results_tree.item(selection[0])
        person_id = item['values'][0]
        
        # Find attention weights for this person
        found = False
        for result in self.prediction_results:
            if result['person_id'] == person_id:
                found = True
                if result['attention_weights'] is not None:
                    # Show in attention tab
                    self.notebook.select(self.tab5)
                    
                    # Clear previous prediction visualization
                    for widget in self.prediction_attention_frame.winfo_children():
                        widget.destroy()
                    
                    # Create title with person info
                    prediction = result['prediction']
                    num_files = result['num_files']
                    title = f"Person-Level Attention Analysis: Person {person_id}\n{prediction} ({num_files} files)"
                    
                    # Create new visualization with responsive sizing
                    fig = self.visualize_attention_weights(
                        result['attention_weights'],
                        y_true=None,
                        title=title
                    )
                    
                    canvas = FigureCanvasTkAgg(fig, master=self.prediction_attention_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill='both', expand=True)
                    
                    # Update scroll region after adding the figure
                    self.root.after(100, lambda: self.attention_canvas.configure(scrollregion=self.attention_canvas.bbox("all")))
                    
                    return
                else:
                    messagebox.showwarning("No Attention Data", 
                                         f"No attention weights available for Person {person_id}.\\n"
                                         "This may be because:\\n"
                                         "- All files were too short\\n"
                                         "- An error occurred during prediction\\n"
                                         "- Attention extraction failed")
                    return
        
        if not found:
            messagebox.showerror("Error", f"Could not find prediction data for Person {person_id}")
    
    def display_test_attention(self):
        """Display attention analysis from test set evaluation"""
        if not hasattr(self, 'results') or 'attention_weights' not in self.results:
            return
        
        # Clear previous widgets and placeholder
        if hasattr(self, 'test_attention_placeholder'):
            self.test_attention_placeholder.pack_forget()
        
        for widget in self.test_attention_frame.winfo_children():
            widget.destroy()
        
        # Create visualization with responsive sizing
        fig = self.visualize_attention_weights(
            self.results['attention_weights'],
            self.results['attention_y_test'],
            title="Test Set Attention Analysis: Healthy vs Patient Patterns"
        )
        
        canvas = FigureCanvasTkAgg(fig, master=self.test_attention_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Update scroll region after adding the figure
        self.root.after(100, lambda: self.attention_canvas.configure(scrollregion=self.attention_canvas.bbox("all")))
        
    def display_results(self):
        """Display results in the results tab with smooth scrolling"""
        # Clear placeholder
        self.results_placeholder.pack_forget()
        
        # Create scrollable canvas
        canvas = tk.Canvas(self.results_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind mouse wheel for smooth scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Results summary
        summary_frame = ttk.LabelFrame(scrollable_frame, text="Performance Summary", padding="10")
        summary_frame.pack(fill='x', padx=10, pady=10)
        
        # Create summary table
        summary_data = []
        for level in ['sequence', 'file', 'person']:
            level_results = self.results[level]
            report = level_results['classification_report']
            summary_data.append({
                'Level': level.capitalize(),
                'Accuracy': f"{level_results['accuracy']:.4f}",
                'Precision (0)': f"{report['0']['precision']:.4f}",
                'Recall (0)': f"{report['0']['recall']:.4f}",
                'F1 (0)': f"{report['0']['f1-score']:.4f}",
                'Precision (1)': f"{report['1']['precision']:.4f}",
                'Recall (1)': f"{report['1']['recall']:.4f}",
                'F1 (1)': f"{report['1']['f1-score']:.4f}",
            })
        
        summary_text = tk.Text(summary_frame, height=8, width=100, font=('Courier', 10))
        summary_text.pack()
        
        # Format table
        header = "Level       Accuracy  Prec(0)  Rec(0)  F1(0)   Prec(1)  Rec(1)  F1(1)\n"
        header += "="*80 + "\n"
        summary_text.insert('1.0', header)
        
        for row in summary_data:
            line = f"{row['Level']:<12}{row['Accuracy']:<10}{row['Precision (0)']:<9}{row['Recall (0)']:<8}{row['F1 (0)']:<8}"
            line += f"{row['Precision (1)']:<9}{row['Recall (1)']:<8}{row['F1 (1)']}\n"
            summary_text.insert(tk.END, line)
        
        summary_text.config(state='disabled')
        
        # Confusion matrices
        cm_frame = ttk.LabelFrame(scrollable_frame, text="Confusion Matrices", padding="10")
        cm_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        fig = Figure(figsize=(12, 4))
        
        for i, level in enumerate(['sequence', 'file', 'person'], 1):
            ax = fig.add_subplot(1, 3, i)
            cm = confusion_matrix(self.results[level]['y_true'], self.results[level]['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{level.capitalize()} Level")
        
        fig.tight_layout()
        
        canvas_widget = FigureCanvasTkAgg(fig, master=cm_frame)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack()
        
        # ROC Curves
        roc_frame = ttk.LabelFrame(scrollable_frame, text="ROC Curves", padding="10")
        roc_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        fig2 = Figure(figsize=(12, 4))
        
        for i, level in enumerate(['sequence', 'file', 'person'], 1):
            ax = fig2.add_subplot(1, 3, i)
            fpr, tpr, _ = roc_curve(self.results[level]['y_true'], self.results[level]['y_pred_prob'])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.4f}')
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{level.capitalize()} Level ROC')
            ax.legend(loc='lower right')
            ax.grid(alpha=0.3)
        
        fig2.tight_layout()
        
        canvas_widget2 = FigureCanvasTkAgg(fig2, master=roc_frame)
        canvas_widget2.draw()
        canvas_widget2.get_tk_widget().pack()
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Switch to results tab
        self.notebook.select(self.tab4)


def main():
    root = tk.Tk()
    app = ParkinsonDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
