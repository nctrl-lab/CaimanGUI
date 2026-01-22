# -*- coding: utf-8 -*-
"""
CaImAn Processing Runner - GUI for running CaImAn processing pipeline

Dohoung Kim (kimd42@postech.ac.kr)
"""
import os
import sys
import json
import time
from pathlib import Path
from natsort import natsorted

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from pyqtgraph.parametertree import Parameter, ParameterTree

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.utils.visualization import nb_inspect_correlation_pnr

pg.setConfigOptions(imageAxisOrder='row-major')


class CaimanRunner(QtWidgets.QDialog):
    """Dialog for running CaImAn processing pipeline."""
    
    def __init__(self, parent=None):
        super(CaimanRunner, self).__init__(parent)
        self.setWindowTitle('CaImAn Processing Runner')
        self.resize(1200, 700)
        
        # Set window icon
        icon_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'Caiman_logo_2.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))
        
        self.data_path = None
        self.cluster = None
        self.n_processes = 8
        self.cnmfe_model = None
        self.video_frames = None
        self.video_frames_dff = None  # Cached dF/F frames
        self.F0_baseline = None  # Cached baseline for dF/F
        self.current_frame_idx = 0
        self.is_playing = False
        self.timer = None
        self.display_cache = {}  # Cache for normalized display frames
        self.gain = 1.0  # Gain multiplier for brightness adjustment
        self.video_min = None  # Overall video min for fixed normalization
        self.video_max = None  # Overall video max for fixed normalization
        self.dff_min = None  # Overall dF/F min for fixed normalization
        self.dff_max = None  # Overall dF/F max for fixed normalization
        self.correlation_image = None  # Correlation image from preview
        self.peak_to_noise_ratio = None  # PNR image from preview
        self._syncing_min_corr = False  # Flag to prevent infinite sync loops
        self._syncing_min_pnr = False  # Flag to prevent infinite sync loops
        self.selected_files = []  # List of selected file paths
        
        # Main horizontal layout (left panel + right panel)
        main_layout = QtWidgets.QHBoxLayout()
        self.setLayout(main_layout)
        
        # Left panel
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(500)
        
        # Parameter tree for configuration
        self.param_tree = ParameterTree(showHeader=False)
        self.setup_parameters()
        left_layout.addWidget(self.param_tree)
        
        # File selection list
        file_selection_label = QtWidgets.QLabel('Select Files to Process:')
        left_layout.addWidget(file_selection_label)
        
        # List widget with checkboxes for file selection
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setMaximumHeight(150)
        self.file_list.itemChanged.connect(self.on_file_selection_changed)
        left_layout.addWidget(self.file_list)
        
        # Buttons for file selection
        file_btn_layout = QtWidgets.QHBoxLayout()
        self.select_all_files_btn = QtWidgets.QPushButton('Select All')
        self.select_all_files_btn.clicked.connect(self.select_all_files)
        self.deselect_all_files_btn = QtWidgets.QPushButton('Deselect All')
        self.deselect_all_files_btn.clicked.connect(self.deselect_all_files)
        file_btn_layout.addWidget(self.select_all_files_btn)
        file_btn_layout.addWidget(self.deselect_all_files_btn)
        left_layout.addLayout(file_btn_layout)
        
        # Progress text area
        self.progress_text = QtWidgets.QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMaximumHeight(150)
        left_layout.addWidget(QtWidgets.QLabel('Progress:'))
        left_layout.addWidget(self.progress_text)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.select_dir_btn = QtWidgets.QPushButton('Select Data Directory')
        self.select_dir_btn.clicked.connect(self.select_directory)
        select_dir_shortcut = QtGui.QShortcut(QtGui.QKeySequence('Ctrl+O'), self)
        select_dir_shortcut.activated.connect(self.select_directory)

        self.preview_btn = QtWidgets.QPushButton('Preview Parameters')
        self.preview_btn.clicked.connect(self.preview_parameters)
        self.preview_btn.setEnabled(False)
        self.run_btn = QtWidgets.QPushButton('Run CaImAn')
        self.run_btn.clicked.connect(self.run_pipeline)
        self.run_btn.setEnabled(False)
        self.close_btn = QtWidgets.QPushButton('Close')
        self.close_btn.clicked.connect(self.close)
        close_shortcut = QtGui.QShortcut(QtGui.QKeySequence('Ctrl+W'), self)
        close_shortcut.activated.connect(self.close)
        
        button_layout.addWidget(self.select_dir_btn)
        button_layout.addWidget(self.preview_btn)
        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.close_btn)
        left_layout.addLayout(button_layout)
        
        main_layout.addWidget(left_panel)
        
        # Right panel - Video viewer
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # Video display
        self.video_label = QtWidgets.QLabel('No video loaded')
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(600, 400)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        right_layout.addWidget(self.video_label)
        
        # Video controls
        controls_layout = QtWidgets.QHBoxLayout()
        
        self.play_btn = QtWidgets.QPushButton('Play')
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        
        self.stop_btn = QtWidgets.QPushButton('Stop')
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.setEnabled(False)
        
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        self.frame_slider.setEnabled(False)
        
        self.frame_label = QtWidgets.QLabel('Frame: 0/0')
        
        self.show_dff_check = QtWidgets.QCheckBox('Show dF/F')
        self.show_dff_check.setEnabled(False)
        self.show_dff_check.stateChanged.connect(self.update_frame_display)
        
        # Gain slider
        gain_label = QtWidgets.QLabel('Gain:')
        self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gain_slider.setMinimum(10)  # 0.1x
        self.gain_slider.setMaximum(500)  # 5.0x
        self.gain_slider.setValue(100)  # 1.0x
        self.gain_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.gain_slider.setTickInterval(50)
        self.gain_slider.setMaximumWidth(150)
        self.gain_slider.valueChanged.connect(self.gain_changed)
        self.gain_slider.setEnabled(False)
        
        self.gain_value_label = QtWidgets.QLabel('1.0x')
        self.gain_value_label.setMinimumWidth(40)
        
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.frame_slider)
        controls_layout.addWidget(self.frame_label)
        controls_layout.addWidget(self.show_dff_check)
        controls_layout.addWidget(gain_label)
        controls_layout.addWidget(self.gain_slider)
        controls_layout.addWidget(self.gain_value_label)
        
        right_layout.addLayout(controls_layout)
        
        # Create plot widget with two subplots
        self.image_plot = pg.GraphicsLayoutWidget()
        self.image_plot.setMinimumHeight(300)
        self.image_plot.setMaximumHeight(400)
        
        # Correlation image plot
        self.corr_plot = self.image_plot.addPlot(row=0, col=0, title='Correlation Image')
        self.corr_img_item = pg.ImageItem()
        self.corr_plot.addItem(self.corr_img_item)
        self.corr_plot.setAspectLocked()
        self.corr_plot.invertY()
        self.corr_plot.hideAxis('left')
        self.corr_plot.hideAxis('bottom')
        
        # PNR image plot
        self.pnr_plot = self.image_plot.addPlot(row=0, col=1, title='Peak-to-Noise Ratio')
        self.pnr_img_item = pg.ImageItem()
        self.pnr_plot.addItem(self.pnr_img_item)
        self.pnr_plot.setAspectLocked()
        self.pnr_plot.invertY()
        self.pnr_plot.hideAxis('left')
        self.pnr_plot.hideAxis('bottom')
        
        right_layout.addWidget(self.image_plot)
        
        # Threshold sliders for correlation and PNR
        threshold_layout = QtWidgets.QHBoxLayout()
        
        # Correlation threshold slider (coupled with parameter tree)
        corr_thresh_label = QtWidgets.QLabel('min_corr:')
        self.corr_thresh_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.corr_thresh_slider.setMinimum(0)
        self.corr_thresh_slider.setMaximum(100)  # 0.00 to 1.00 in steps of 0.01
        self.corr_thresh_slider.setValue(85)  # Default 0.85
        self.corr_thresh_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.corr_thresh_slider.setTickInterval(10)
        self.corr_thresh_slider.valueChanged.connect(self.corr_thresh_changed)
        self.corr_thresh_slider.setEnabled(True)
        
        self.corr_thresh_value_label = QtWidgets.QLabel('0.85')
        self.corr_thresh_value_label.setMinimumWidth(40)
        
        # PNR threshold slider (coupled with parameter tree)
        pnr_thresh_label = QtWidgets.QLabel('min_pnr:')
        self.pnr_thresh_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pnr_thresh_slider.setMinimum(0)
        self.pnr_thresh_slider.setMaximum(200)  # 0 to 20 in steps of 0.1
        self.pnr_thresh_slider.setValue(100)  # Default 10.0
        self.pnr_thresh_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.pnr_thresh_slider.setTickInterval(20)
        self.pnr_thresh_slider.valueChanged.connect(self.pnr_thresh_changed)
        self.pnr_thresh_slider.setEnabled(True)
        
        self.pnr_thresh_value_label = QtWidgets.QLabel('10.0')
        self.pnr_thresh_value_label.setMinimumWidth(40)
        
        threshold_layout.addWidget(corr_thresh_label)
        threshold_layout.addWidget(self.corr_thresh_slider)
        threshold_layout.addWidget(self.corr_thresh_value_label)
        threshold_layout.addWidget(pnr_thresh_label)
        threshold_layout.addWidget(self.pnr_thresh_slider)
        threshold_layout.addWidget(self.pnr_thresh_value_label)
        
        right_layout.addLayout(threshold_layout)
        
        main_layout.addWidget(right_panel)
        
        # Initialize sliders with parameter values (after sliders are created)
        self._initialize_sliders()
        
        # Setup timer for video playback
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)
        
        self.log('CaImAn Runner initialized. Please select a data directory.')
    
    def setup_parameters(self):
        """Setup parameter tree with default values."""
        params = [
            {'name': 'Data Parameters', 'type': 'group', 'children': [
                {'name': 'Data Directory', 'type': 'str', 'value': '', 'readonly': True},
                {'name': 'Frame Rate (Hz)', 'type': 'float', 'value': 20.0},
                {'name': 'Decay Time', 'type': 'float', 'value': 0.4},
            ]},
            {'name': 'Cluster Parameters', 'type': 'group', 'children': [
                {'name': 'Number of Processes', 'type': 'int', 'value': 8, 'limits': (1, 32)},
            ]},
            {'name': 'Motion Correction', 'type': 'group', 'children': [
                {'name': 'gSig_filt', 'type': 'int', 'value': 7, 'limits': (1, 33)},
                {'name': 'max_shifts', 'type': 'int', 'value': 20},
            ]},
            {'name': 'CNMF-E Parameters', 'type': 'group', 'children': [
                {'name': 'gSig', 'type': 'int', 'value': 6, 'limits': (1, 12)},
                {'name': 'gSiz', 'type': 'int', 'value': 15},
                {'name': 'min_corr', 'type': 'float', 'value': 0.85, 'limits': (0, 1), 'step': 0.01},
                {'name': 'min_pnr', 'type': 'float', 'value': 10.0},
                {'name': 'merge_thr', 'type': 'float', 'value': 0.65, 'limits': (0, 1), 'step': 0.01},
            ]},
            {'name': 'Quality Control', 'type': 'group', 'children': [
                {'name': 'min_SNR', 'type': 'float', 'value': 1.5},
                {'name': 'rval_thr', 'type': 'float', 'value': 0.85, 'limits': (0, 1), 'step': 0.01},
            ]},
        ]
        self.params = Parameter.create(name='params', type='group', children=params)
        self.param_tree.setParameters(self.params, showTop=False)
        
        # Connect parameter changes to sync with sliders (bidirectional sync for min_corr and min_pnr)
        self.params.child('CNMF-E Parameters', 'min_corr').sigValueChanged.connect(
            lambda param, value: self._sync_min_corr_from_param(value)
        )
        self.params.child('CNMF-E Parameters', 'min_pnr').sigValueChanged.connect(
            lambda param, value: self._sync_min_pnr_from_param(value)
        )
    
    def _initialize_sliders(self):
        """Initialize slider values from parameter tree (called after UI is created)."""
        min_corr_val = self.params.child('CNMF-E Parameters', 'min_corr').value()
        min_pnr_val = self.params.child('CNMF-E Parameters', 'min_pnr').value()
        self.corr_thresh_slider.setValue(int(min_corr_val * 100))
        self.pnr_thresh_slider.setValue(int(min_pnr_val * 10))
        self.corr_thresh_value_label.setText(f'{min_corr_val:.2f}')
        self.pnr_thresh_value_label.setText(f'{min_pnr_val:.1f}')
    
    def populate_file_list(self, avi_files):
        """Populate the file list widget with AVI files."""
        self.file_list.clear()
        self.selected_files = []
        
        for file_path in avi_files:
            item = QtWidgets.QListWidgetItem(file_path.name)
            item.setCheckState(QtCore.Qt.Checked)  # All files selected by default
            item.setData(QtCore.Qt.UserRole, file_path)  # Store full path
            self.file_list.addItem(item)
            self.selected_files.append(file_path)
        
        self.update_buttons_state()
    
    def on_file_selection_changed(self, item):
        """Handle file selection checkbox change."""
        file_path = item.data(QtCore.Qt.UserRole)
        if item.checkState() == QtCore.Qt.Checked:
            if file_path not in self.selected_files:
                self.selected_files.append(file_path)
        else:
            if file_path in self.selected_files:
                self.selected_files.remove(file_path)
        self.update_buttons_state()
    
    def select_all_files(self):
        """Select all files in the list."""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            item.setCheckState(QtCore.Qt.Checked)
    
    def deselect_all_files(self):
        """Deselect all files in the list."""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            item.setCheckState(QtCore.Qt.Unchecked)
    
    def update_buttons_state(self):
        """Update button states based on selected files."""
        has_selected = len(self.selected_files) > 0
        self.run_btn.setEnabled(has_selected)
        self.preview_btn.setEnabled(has_selected)
        if has_selected:
            self.log(f'{len(self.selected_files)} file(s) selected for processing')
    
    def _format_duration(self, duration):
        """Format duration in seconds to human-readable string."""
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    
    def log(self, message):
        """Add message to progress log."""
        self.progress_text.append(message)
        QtWidgets.QApplication.processEvents()
    
    def select_directory(self):
        """Open dialog to select data directory."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select Data Directory', '')
        if directory:
            self.data_path = Path(directory)
            self.params.child('Data Parameters', 'Data Directory').setValue(str(self.data_path))
            self.log(f'Selected directory: {self.data_path}')
            
            # Try to load metadata
            meta_path = self.data_path / 'metaData.json'
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    framerate = float(str(meta['frameRate']).upper().replace('FPS', '').strip())
                    self.params.child('Data Parameters', 'Frame Rate (Hz)').setValue(framerate)
                    self.log(f'Loaded frame rate from metadata: {framerate:.2f} Hz')
                except Exception as e:
                    self.log(f'Could not load metadata: {e}')
            
            # Check for AVI files and populate file list
            avi_files = natsorted(self.data_path.glob('*.avi'))
            if avi_files:
                self.log(f'Found {len(avi_files)} AVI files')
                self.populate_file_list(avi_files)
                # Load first video for preview
                if avi_files:
                    self.load_video_preview(avi_files[0])
            else:
                self.log('Warning: No AVI files found in directory')
                self.file_list.clear()
                self.run_btn.setEnabled(False)
                self.preview_btn.setEnabled(False)
    
    def load_video_preview(self, video_path, max_frames=1000):
        """Load video frames for preview (limited to max_frames for performance)."""
        try:
            import cv2
            self.log(f'Loading video preview: {Path(video_path).name}')
            
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            frame_count = 0
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert to grayscale if needed
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame.astype(np.float32))
                frame_count += 1
            
            cap.release()
            
            if len(frames) == 0:
                self.video_label.setText('Could not load video frames')
                return
            
            self.video_frames = np.array(frames)
            self.current_frame_idx = 0
            self.frame_slider.setMaximum(len(self.video_frames) - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(True)
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.show_dff_check.setEnabled(True)
            self.gain_slider.setEnabled(True)
            
            # Pre-calculate F0 baseline for dF/F (20th percentile) - PER PIXEL
            # This calculates the 20th percentile along the time axis (axis=0),
            # resulting in a 2D array (H, W) where each pixel has its own F0 baseline
            self.log('Calculating per-pixel baseline for dF/F...')
            self.F0_baseline = np.percentile(self.video_frames, 20, axis=0)  # Shape: (H, W) - per pixel
            self.F0_baseline = np.maximum(self.F0_baseline, 1e-6)
            self.video_frames_dff = None  # Will be calculated on demand
            self.display_cache = {}  # Clear cache
            
            # Calculate overall video statistics for fixed normalization
            self.log('Calculating video statistics...')
            self.video_min = np.percentile(self.video_frames, 1)  # 1st percentile to ignore outliers
            self.video_max = np.percentile(self.video_frames, 99)  # 99th percentile to ignore outliers
            self.dff_min = None  # Will be calculated when dF/F is computed
            self.dff_max = None
            
            self.update_frame_display()
            self.log(f'Loaded {len(self.video_frames)} frames for preview')
            
        except Exception as e:
            self.log(f'Error loading video: {str(e)}')
            self.video_label.setText(f'Error loading video: {str(e)}')
    
    def gain_changed(self, value):
        """Handle gain slider change."""
        self.gain = value / 100.0  # Convert slider value (10-500) to gain (0.1-5.0)
        self.gain_value_label.setText(f'{self.gain:.2f}x')
        # Clear cache when gain changes to force recalculation
        self.display_cache = {}
        self.update_frame_display()
    
    def _sync_min_corr_from_param(self, value):
        """Sync min_corr slider from parameter change (prevents infinite loops)."""
        if self._syncing_min_corr:
            return
        self._syncing_min_corr = True
        try:
            self.corr_thresh_slider.setValue(int(value * 100))
            self.corr_thresh_value_label.setText(f'{value:.2f}')
            self.update_image_plots()
        finally:
            self._syncing_min_corr = False
    
    def _sync_min_pnr_from_param(self, value):
        """Sync min_pnr slider from parameter change (prevents infinite loops)."""
        if self._syncing_min_pnr:
            return
        self._syncing_min_pnr = True
        try:
            self.pnr_thresh_slider.setValue(int(value * 10))
            self.pnr_thresh_value_label.setText(f'{value:.1f}')
            self.update_image_plots()
        finally:
            self._syncing_min_pnr = False
    
    def update_frame_display(self):
        """Update the displayed frame (original or dF/F) - optimized version."""
        if self.video_frames is None or len(self.video_frames) == 0:
            return
        
        use_dff = self.show_dff_check.isChecked()
        cache_key = (self.current_frame_idx, use_dff, self.gain)
        
        # Check cache first
        if cache_key in self.display_cache:
            pixmap = self.display_cache[cache_key]
        else:
            # Get frame (original or dF/F)
            if use_dff:
                # Pre-calculate all dF/F frames if not already done
                # dF/F is calculated per-pixel: (F - F0) / F0 for each pixel independently
                if self.video_frames_dff is None:
                    self.log('Pre-calculating per-pixel dF/F for all frames...')
                    # Broadcasting: (T, H, W) - (H, W) = (T, H, W) - each pixel normalized independently
                    self.video_frames_dff = (self.video_frames - self.F0_baseline) / self.F0_baseline
                    # Calculate dF/F statistics (using percentiles to ignore outliers)
                    self.dff_min = np.percentile(self.video_frames_dff, 1)
                    self.dff_max = np.percentile(self.video_frames_dff, 99)
                    self.log('dF/F calculation complete')
                
                frame = self.video_frames_dff[self.current_frame_idx]
                # Use fixed normalization range for dF/F
                frame_min = self.dff_min
                frame_max = self.dff_max
            else:
                frame = self.video_frames[self.current_frame_idx]
                # Use fixed normalization range for original video
                frame_min = self.video_min
                frame_max = self.video_max
            
            # Apply gain
            frame = frame * self.gain
            
            # Clip and normalize to 0-255 using fixed range (not per-frame min/max)
            # This prevents outliers from affecting the whole frame
            frame_clipped = np.clip(frame, frame_min, frame_max)
            if frame_max > frame_min:
                frame_norm = ((frame_clipped - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
            else:
                frame_norm = np.clip(frame_clipped, 0, 255).astype(np.uint8)
            
            # Convert to QImage
            height, width = frame_norm.shape
            q_image = QtGui.QImage(frame_norm.data, width, height, width, QtGui.QImage.Format_Grayscale8)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            
            # Scale to fit label (cache the scaled version)
            label_size = self.video_label.size()
            scaled_pixmap = pixmap.scaled(
                label_size, 
                QtCore.Qt.KeepAspectRatio, 
                QtCore.Qt.SmoothTransformation
            )
            
            # Cache the scaled pixmap (limit cache size to avoid memory issues)
            if len(self.display_cache) > 100:
                # Clear oldest entries (simple FIFO)
                keys_to_remove = list(self.display_cache.keys())[:50]
                for key in keys_to_remove:
                    del self.display_cache[key]
            
            self.display_cache[cache_key] = scaled_pixmap
            pixmap = scaled_pixmap
        
        # Display the cached pixmap
        self.video_label.setPixmap(pixmap)
        
        # Update frame label
        self.frame_label.setText(f'Frame: {self.current_frame_idx + 1}/{len(self.video_frames)}')
    
    def toggle_play(self):
        """Toggle video playback."""
        if self.video_frames is None:
            return
        
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText('Play')
        else:
            framerate = self.params.child('Data Parameters', 'Frame Rate (Hz)').value()
            interval = int(1000 / framerate)  # milliseconds per frame
            self.timer.start(interval)
            self.is_playing = True
            self.play_btn.setText('Pause')
    
    def stop_video(self):
        """Stop video playback and reset to first frame."""
        self.timer.stop()
        self.is_playing = False
        self.play_btn.setText('Play')
        self.current_frame_idx = 0
        self.frame_slider.setValue(0)
        self.update_frame_display()
    
    def clear_video_cache(self):
        """Clear video-related caches."""
        self.video_frames = None
        self.video_frames_dff = None
        self.F0_baseline = None
        self.display_cache = {}
    
    def next_frame(self):
        """Advance to next frame during playback."""
        if self.video_frames is None:
            return
        
        self.current_frame_idx += 1
        if self.current_frame_idx >= len(self.video_frames):
            self.current_frame_idx = 0  # Loop back to start
        
        self.frame_slider.setValue(self.current_frame_idx)
        self.update_frame_display()
    
    def slider_changed(self, value):
        """Handle slider value change."""
        if self.video_frames is None:
            return
        
        self.current_frame_idx = value
        self.update_frame_display()
    
    def corr_thresh_changed(self, value):
        """Handle correlation threshold slider change - sync with parameter tree."""
        if self._syncing_min_corr:
            return
        min_corr = value / 100.0  # Convert slider value (0-100) to threshold (0.0-1.0)
        self.corr_thresh_value_label.setText(f'{min_corr:.2f}')
        # Update parameter value
        self._syncing_min_corr = True
        try:
            self.params.child('CNMF-E Parameters', 'min_corr').setValue(min_corr)
        finally:
            self._syncing_min_corr = False
        self.update_image_plots()
    
    def pnr_thresh_changed(self, value):
        """Handle PNR threshold slider change - sync with parameter tree."""
        if self._syncing_min_pnr:
            return
        min_pnr = value / 10.0  # Convert slider value (0-200) to threshold (0.0-20.0)
        self.pnr_thresh_value_label.setText(f'{min_pnr:.1f}')
        # Update parameter value
        self._syncing_min_pnr = True
        try:
            self.params.child('CNMF-E Parameters', 'min_pnr').setValue(min_pnr)
        finally:
            self._syncing_min_pnr = False
        self.update_image_plots()
    
    def update_image_plots(self):
        """Update correlation and PNR image plots with thresholding."""
        if self.correlation_image is not None:
            # Get threshold from parameter tree
            min_corr = self.params.child('CNMF-E Parameters', 'min_corr').value()
            # Create thresholded image: pixels below threshold are shown in red/black
            corr_mask = self.correlation_image < min_corr
            
            # Normalize correlation image for display (0-1 range)
            corr_min = np.min(self.correlation_image)
            corr_max = np.max(self.correlation_image)
            corr_range = corr_max - corr_min + 1e-10
            corr_norm = (self.correlation_image - corr_min) / corr_range
            
            # Create RGB image: thresholded pixels in red, others in grayscale
            # Convert to uint8 (0-255) for pyqtgraph compatibility
            corr_display = np.zeros((*corr_norm.shape, 3), dtype=np.uint8)
            corr_uint8 = (corr_norm * 255).astype(np.uint8)
            corr_display[:, :, 0] = corr_uint8  # Red channel
            corr_display[:, :, 1] = corr_uint8  # Green channel
            corr_display[:, :, 2] = corr_uint8  # Blue channel
            
            # Set thresholded pixels to red (high red, low green/blue)
            corr_display[corr_mask, 0] = 255  # Red
            corr_display[corr_mask, 1] = 0    # No green
            corr_display[corr_mask, 2] = 0   # No blue
            
            self.corr_img_item.setImage(corr_display, autoLevels=False)
            gSig = self.params.child("CNMF-E Parameters", "gSig").value()
            self.corr_plot.setTitle(f'Correlation Image (gSig={gSig}, min_corr={min_corr:.2f})')
        
        if self.peak_to_noise_ratio is not None:
            # Get threshold from parameter tree
            min_pnr = self.params.child('CNMF-E Parameters', 'min_pnr').value()
            # Create thresholded image: pixels below threshold are shown in red/black
            pnr_mask = self.peak_to_noise_ratio < min_pnr
            
            # Normalize PNR image for display
            pnr_min = np.min(self.peak_to_noise_ratio)
            pnr_max = np.max(self.peak_to_noise_ratio)
            pnr_range = pnr_max - pnr_min + 1e-10
            pnr_norm = (self.peak_to_noise_ratio - pnr_min) / pnr_range
            
            # Create RGB image: thresholded pixels in red, others in grayscale
            # Convert to uint8 (0-255) for pyqtgraph compatibility
            pnr_display = np.zeros((*pnr_norm.shape, 3), dtype=np.uint8)
            pnr_uint8 = (pnr_norm * 255).astype(np.uint8)
            pnr_display[:, :, 0] = pnr_uint8  # Red channel
            pnr_display[:, :, 1] = pnr_uint8  # Green channel
            pnr_display[:, :, 2] = pnr_uint8  # Blue channel
            
            # Set thresholded pixels to red (high red, low green/blue)
            pnr_display[pnr_mask, 0] = 255  # Red
            pnr_display[pnr_mask, 1] = 0    # No green
            pnr_display[pnr_mask, 2] = 0    # No blue
            
            self.pnr_img_item.setImage(pnr_display, autoLevels=False)
            gSig = self.params.child("CNMF-E Parameters", "gSig").value()
            self.pnr_plot.setTitle(f'Peak-to-Noise Ratio (gSig={gSig}, min_pnr={min_pnr:.1f})')
    
    def preview_parameters(self):
        """Preview parameters using first 500 frames to validate gSig."""
        if not self.data_path or not self.data_path.exists():
            QtWidgets.QMessageBox.warning(self, 'Error', 'Please select a valid data directory')
            return
        
        # Get selected files
        if not self.selected_files:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Please select at least one file to preview')
            return
        
        files = self.selected_files
        
        self.preview_btn.setEnabled(False)
        self.log('\n' + '='*50)
        self.log('Previewing Parameters (first 500 frames)')
        self.log('='*50)
        
        try:
            # Get CNMF-E gSig parameter only
            gSig = self.params.child('CNMF-E Parameters', 'gSig').value()
            
            self.log(f'Testing CNMF-E gSig parameter: {gSig}')
            
            # Load first 500 frames from first file
            self.log(f'Loading first 500 frames from: {files[0].name}')
            import cv2
            
            cap = cv2.VideoCapture(str(files[0]))
            frames = []
            max_frames = 500
            
            for i in range(max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert to grayscale if needed
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame.astype(np.float32))
            
            cap.release()
            
            if len(frames) == 0:
                raise ValueError('Could not load any frames from video file')
            
            images = np.array(frames)  # Shape: (T, H, W)
            self.log(f'Loaded {len(frames)} frames, shape: {images.shape}')

            # Test CNMF-E gSig parameter
            self.log('\nComputing correlation and PNR images...')
            correlation_image, peak_to_noise_ratio = cm.summary_images.correlation_pnr(
                images, gSig=gSig, swap_dim=False
            )
            
            # Store images for display on the right side
            self.correlation_image = correlation_image
            self.peak_to_noise_ratio = peak_to_noise_ratio
            
            # Initialize threshold sliders based on image ranges
            corr_max = float(np.max(correlation_image))
            pnr_max = float(np.max(peak_to_noise_ratio))
            
            # Update slider ranges based on image ranges (but keep parameter values)
            # Correlation: typically 0.0 to 1.0, slider 0-100 (0.00-1.00)
            self.corr_thresh_slider.setMaximum(int(corr_max * 100))
            
            # PNR: typically 0 to 20+, slider 0-200 (0.0-20.0)
            self.pnr_thresh_slider.setMaximum(int(pnr_max * 10))
            
            # Sync sliders with current parameter values
            min_corr_val = self.params.child('CNMF-E Parameters', 'min_corr').value()
            min_pnr_val = self.params.child('CNMF-E Parameters', 'min_pnr').value()
            self.corr_thresh_slider.setValue(int(min_corr_val * 100))
            self.pnr_thresh_slider.setValue(int(min_pnr_val * 10))
            self.corr_thresh_value_label.setText(f'{min_corr_val:.2f}')
            self.pnr_thresh_value_label.setText(f'{min_pnr_val:.1f}')
            
            # Display images in plot on the right side
            self.update_image_plots()
            
            # Calculate statistics
            corr_mean = np.mean(correlation_image)
            corr_max = np.max(correlation_image)
            corr_std = np.std(correlation_image)
            pnr_mean = np.mean(peak_to_noise_ratio)
            pnr_max = np.max(peak_to_noise_ratio)
            
            self.log(f'  Correlation image - Mean: {corr_mean:.3f}, Max: {corr_max:.3f}, Std: {corr_std:.3f}')
            self.log(f'  PNR image - Mean: {pnr_mean:.2f}, Max: {pnr_max:.2f}')
            self.log('\nPreview complete. Images displayed on the right side.')
            
        except Exception as e:
            self.log(f'\nERROR during preview: {str(e)}')
            import traceback
            self.log(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, 'Preview Error', f'Preview failed:\n{str(e)}')
        finally:
            self.preview_btn.setEnabled(True)
    
    def run_pipeline(self):
        """Run the complete CaImAn processing pipeline."""
        if not self.data_path or not self.data_path.exists():
            QtWidgets.QMessageBox.warning(self, 'Error', 'Please select a valid data directory')
            return
        
        self.run_btn.setEnabled(False)
        start_time = time.time()
        self.log('\n' + '='*50)
        self.log('Starting CaImAn Processing Pipeline')
        self.log('='*50)
        
        try:
            # Step 1: Setup cluster
            step_start = time.time()
            self.log('\n[1/5] Setting up cluster...')
            self.setup_cluster()
            step_duration = time.time() - step_start
            self.log(f'  ✓ Cluster setup completed in {self._format_duration(step_duration)}')
            
            # Step 2: Motion correction
            step_start = time.time()
            self.log('\n[2/5] Running motion correction...')
            files_mc = self.run_motion_correction()
            step_duration = time.time() - step_start
            self.log(f'  ✓ Motion correction completed in {self._format_duration(step_duration)}')
            
            # Step 3: CNMF-E
            step_start = time.time()
            self.log('\n[3/5] Running CNMF-E...')
            self.run_cnmfe(files_mc)
            step_duration = time.time() - step_start
            self.log(f'  ✓ CNMF-E completed in {self._format_duration(step_duration)}')
            
            # Step 4: Quality control
            step_start = time.time()
            self.log('\n[4/5] Running quality control...')
            self.run_quality_control(files_mc)
            step_duration = time.time() - step_start
            self.log(f'  ✓ Quality control completed in {self._format_duration(step_duration)}')
            
            # Step 5: Save results
            step_start = time.time()
            self.log('\n[5/5] Saving results...')
            self.save_results(files_mc)
            step_duration = time.time() - step_start
            self.log(f'  ✓ Results saved in {self._format_duration(step_duration)}')
            
            # Calculate and display total duration
            end_time = time.time()
            total_duration = end_time - start_time
            duration_str = self._format_duration(total_duration)
            
            self.log('\n' + '='*50)
            self.log('Processing completed successfully!')
            self.log(f'Total duration: {duration_str} ({total_duration:.2f} seconds)')
            self.log('='*50)
            
            caiman_dir = self.data_path / 'caiman'
            save_path = caiman_dir / f'{self.data_path.name}_data.hdf5'
            QtWidgets.QMessageBox.information(
                self, 'Success', 
                f'Processing completed!\nDuration: {duration_str}\n\nResults saved to:\n{save_path}')
            
        except Exception as e:
            # Calculate and display duration even on error
            end_time = time.time()
            total_duration = end_time - start_time
            duration_str = self._format_duration(total_duration)
            
            self.log(f'\nERROR: {str(e)}')
            self.log(f'Processing failed after {duration_str}')
            import traceback
            self.log(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, 'Error', f'Processing failed after {duration_str}:\n{str(e)}')
        finally:
            # Cleanup
            if self.cluster is not None:
                self.log('\nCleaning up cluster...')
                cm.stop_server(dview=self.cluster)
                self.cluster = None
            self.run_btn.setEnabled(True)
    
    def setup_cluster(self):
        """Setup multiprocessing cluster."""
        n_processes = self.params.child('Cluster Parameters', 'Number of Processes').value()
        self.n_processes = n_processes
        
        if self.cluster is not None:
            cm.stop_server(dview=self.cluster)
        
        _, self.cluster, n_processes = cm.cluster.setup_cluster(
            backend='multiprocessing',
            n_processes=n_processes,
            ignore_preexisting=False
        )
        self.log(f'Cluster with {n_processes} processes set up.')
    
    def run_motion_correction(self):
        """Run motion correction."""
        # Get selected files
        if not self.selected_files:
            raise ValueError('No files selected for processing')
        
        files = self.selected_files
        
        self.log(f'Found {len(files)} AVI files')
        
        # Get parameters
        framerate = self.params.child('Data Parameters', 'Frame Rate (Hz)').value()
        decay_time = self.params.child('Data Parameters', 'Decay Time').value()
        gSig_filt = self.params.child('Motion Correction', 'gSig_filt').value()
        max_shifts = self.params.child('Motion Correction', 'max_shifts').value()
        
        # Setup parameters
        data_params = {
            'data': {
                'fnames': files,
                'fr': framerate,
                'decay_time': decay_time,
            },
            'motion': {
                'gSig_filt': (gSig_filt, gSig_filt),
                'max_shifts': (max_shifts, max_shifts),
                'border_nan': 'copy'
            }
        }
        parameters = cnmf.params.CNMFParams(params_dict=data_params)
        
        # Run motion correction
        mot_correct = MotionCorrect(
            files[0], 
            dview=self.cluster, 
            **parameters.get_group('motion')
        )
        mot_correct.motion_correct(save_movie=True)
        
        # Save memory-mapped file
        files_mc = cm.save_memmap(
            mot_correct.fname_tot_rig,
            base_name='memmap_',
            order='C',
            border_to_0=0
        )
        
        self.log(f'Motion correction completed. Saved to: {files_mc}')
        return files_mc
    
    def run_cnmfe(self, files_mc):
        """Run CNMF-E processing."""
        # Load motion corrected movie
        Yr, dims, T = cm.load_memmap(files_mc)
        images = Yr.T.reshape((T,) + dims, order='F')
        
        # Get parameters
        gSig = self.params.child('CNMF-E Parameters', 'gSig').value()
        gSiz = self.params.child('CNMF-E Parameters', 'gSiz').value()
        min_corr = self.params.child('CNMF-E Parameters', 'min_corr').value()
        min_pnr = self.params.child('CNMF-E Parameters', 'min_pnr').value()
        merge_thr = self.params.child('CNMF-E Parameters', 'merge_thr').value()
        
        # Calculate correlation and PNR images
        self.log('Calculating correlation and PNR images...')
        correlation_image, peak_to_noise_ratio = cm.summary_images.correlation_pnr(
            images, gSig=gSig, swap_dim=False
        )
        
        # Store correlation image for later use
        self.correlation_image = correlation_image
        self.peak_to_noise_ratio = peak_to_noise_ratio
        self.image_max = np.max(images, axis=0)
        self.image_mean = np.mean(images, axis=0)
        self.image_std = np.std(images, axis=0)
        
        # Get framerate for data params
        framerate = self.params.child('Data Parameters', 'Frame Rate (Hz)').value()
        decay_time = self.params.child('Data Parameters', 'Decay Time').value()
        
        # Setup CNMF-E parameters (include data params with motion-corrected file)
        cnmfe_params = {
            'data': {
                'fnames': self.selected_files,  # Use motion-corrected file
                'fr': framerate,
                'decay_time': decay_time,
            },
            'patch': {
                'del_duplicates': True,
                'low_rank_background': None,
                'nb_patch': 0,
                'rf': 48,
                'stride': 8,
            },
            'preprocess': {'p': 0},
            'init': {
                'K': None,
                'center_psf': True,
                'method_init': 'corr_pnr',
                'nb': 0,
                'normalize_init': False,
                'gSig': (gSig, gSig),
                'gSiz': (gSiz, gSiz),
                'min_corr': min_corr,
                'min_pnr': min_pnr,
                'ring_size_factor': 1.4,
                'tsub': 2,
                'ssub': 1,
                'ssub_B': 2,
            },
            'temporal': {'p': 0},
            'merging': {'merge_thr': merge_thr}
        }
        
        # Create and fit CNMF-E model
        parameters = cnmf.params.CNMFParams(params_dict=cnmfe_params)
        self.cnmfe_model = cnmf.CNMF(
            n_processes=self.n_processes,
            dview=self.cluster,
            params=parameters
        )
        
        self.log('Fitting CNMF-E model (this may take a while)...')
        self.cnmfe_model.fit(images)
        self.log('CNMF-E fitting completed.')
    
    def run_quality_control(self, files_mc):
        """Run quality control."""
        min_SNR = self.params.child('Quality Control', 'min_SNR').value()
        rval_thr = self.params.child('Quality Control', 'rval_thr').value()
        
        quality_params = {
            'quality': {
                'min_SNR': min_SNR,
                'rval_thr': rval_thr,
                'use_cnn': False
            }
        }
        self.cnmfe_model.params.change_params(params_dict=quality_params)
        
        Yr, dims, T = cm.load_memmap(files_mc)
        images = Yr.T.reshape((T,) + dims, order='F')
        
        self.log('Evaluating components...')
        self.cnmfe_model.estimates.evaluate_components(images, self.cnmfe_model.params, dview=self.cluster)
        
        if self.cnmfe_model.estimates.F_dff is None:
            self.log('Calculating F_dff...')
            self.cnmfe_model.estimates.detrend_df_f(
                quantileMin=8,
                frames_window=250,
                flag_auto=False,
                use_residuals=False
            )
        
        n_components = len(self.cnmfe_model.estimates.idx_components)
        self.log(f'Quality control completed. {n_components} components passed.')
    
    def save_results(self, files_mc):
        """Save processing results."""
        # Create caiman output directory if it doesn't exist
        caiman_dir = self.data_path / 'caiman'
        caiman_dir.mkdir(exist_ok=True)
        
        save_path = caiman_dir / f'{self.data_path.name}_data.hdf5'
        
        # Add correlation image if not available
        if not hasattr(self.cnmfe_model, 'cn_filter') or self.cnmfe_model.cn_filter is None:
            if hasattr(self, 'correlation_image') and hasattr(self, 'peak_to_noise_ratio') and hasattr(self, 'image_max') and hasattr(self, 'image_mean') and hasattr(self, 'image_std'):
                self.cnmfe_model.cn_filter = self.correlation_image
                self.cnmfe_model.pnr = self.peak_to_noise_ratio
                self.cnmfe_model.image_max = self.image_max
                self.cnmfe_model.image_mean = self.image_mean
                self.cnmfe_model.image_std = self.image_std
            else:
                Yr, dims, T = cm.load_memmap(files_mc)
                images = Yr.T.reshape((T,) + dims, order='F')

                correlation_image, peak_to_noise_ratio = cm.summary_images.correlation_pnr(
                    images, 
                    gSig=self.params.child('CNMF-E Parameters', 'gSig').value(), 
                    swap_dim=False
                )

                self.cnmfe_model.cn_filter = correlation_image
                self.cnmfe_model.pnr = peak_to_noise_ratio

                self.cnmfe_model.image_max = np.max(images, axis=0)
                self.cnmfe_model.image_mean = np.mean(images, axis=0)
                self.cnmfe_model.image_std = np.std(images, axis=0)
        
        self.cnmfe_model.save(str(save_path))
        self.log(f'Results saved to: {save_path}')


def caiman_runner_window(parent=None):
    """Open CaImAn runner dialog."""
    win = CaimanRunner(parent)
    win.exec()
