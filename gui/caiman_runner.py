# -*- coding: utf-8 -*-
"""
CaImAn Processing Runner - GUI for running CaImAn processing pipeline

Dohoung Kim (kimd42@postech.ac.kr)
"""

import os
import json
import time
import xml.etree.ElementTree as ET
import logging
import datetime
from pathlib import Path
from natsort import natsorted
import psutil

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from pyqtgraph.parametertree import Parameter, ParameterTree

import caiman as cm
from caiman.motion_correction import MotionCorrect, high_pass_filter_space
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.cnmf import params as cnmf_params
from caiman.summary_images import local_correlations_movie_offline
from caiman.utils.visualization import nb_inspect_correlation_pnr

pg.setConfigOptions(imageAxisOrder="row-major")

logger = logging.getLogger("caiman")
logger.setLevel(logging.WARNING)


class CaimanRunner(QtWidgets.QDialog):
    """Dialog for running CaImAn processing pipeline."""

    def __init__(self, parent=None):
        super(CaimanRunner, self).__init__(parent)
        self.setWindowTitle("CaImAn Processing Runner")
        self.resize(1200, 700)

        self.data_path = None
        self.caiman_path = None
        self.base_name = ""
        self.cluster = None
        self.n_processes = 8
        self.cnmfe_model = None
        self.video_frames = None
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
        self.save_logger = None  # Logger instance for save folder
        self._preview_video_path = None  # Path for preview (load full video on Play)
        self._frame_count = 0  # Total frames (from file)
        self._first_frame = None  # Single frame for display when video not loaded

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
        file_selection_label = QtWidgets.QLabel("Select Files to Process:")
        left_layout.addWidget(file_selection_label)

        # List widget with checkboxes for file selection
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setMaximumHeight(150)
        self.file_list.itemChanged.connect(self.on_file_selection_changed)
        self.file_list.currentItemChanged.connect(self.on_file_clicked)
        left_layout.addWidget(self.file_list)

        # Buttons for file selection
        file_btn_layout = QtWidgets.QHBoxLayout()
        self.select_all_files_btn = QtWidgets.QPushButton("Select All")
        self.select_all_files_btn.clicked.connect(self.select_all_files)
        self.select_all_files_btn.setEnabled(False)  # Initially disabled
        self.deselect_all_files_btn = QtWidgets.QPushButton("Deselect All")
        self.deselect_all_files_btn.clicked.connect(self.deselect_all_files)
        self.deselect_all_files_btn.setEnabled(False)  # Initially disabled
        file_btn_layout.addWidget(self.select_all_files_btn)
        file_btn_layout.addWidget(self.deselect_all_files_btn)
        left_layout.addLayout(file_btn_layout)

        # Progress text area
        self.progress_text = QtWidgets.QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMaximumHeight(150)
        left_layout.addWidget(QtWidgets.QLabel("Progress:"))
        left_layout.addWidget(self.progress_text)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.select_dir_btn = QtWidgets.QPushButton("Select Data Directory")
        self.select_dir_btn.clicked.connect(self.select_directory)
        select_dir_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+O"), self)
        select_dir_shortcut.activated.connect(self.select_directory)

        self.preview_btn = QtWidgets.QPushButton("Preview Parameters")
        self.preview_btn.clicked.connect(self.preview_parameters)
        self.preview_btn.setEnabled(False)
        self.run_btn = QtWidgets.QPushButton("Run CaImAn")
        self.run_btn.clicked.connect(self.run_pipeline)
        self.run_btn.setEnabled(False)
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        close_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+W"), self)
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
        self.video_label = QtWidgets.QLabel("No video loaded")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(600, 600)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        right_layout.addWidget(self.video_label)

        # Video controls
        controls_layout = QtWidgets.QHBoxLayout()

        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        self.play_btn.setMaximumWidth(60)

        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMaximumWidth(60)

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        self.frame_slider.setEnabled(False)

        self.frame_label = QtWidgets.QLabel("Frame: 0/0")

        self.show_dff_check = QtWidgets.QCheckBox("Show dF/F")
        self.show_dff_check.setEnabled(False)
        self.show_dff_check.stateChanged.connect(self.update_frame_display)

        # Checkbox to apply gSig_filt (high-pass filter) to correlation/PNR images
        self.apply_gsig_filt_check = QtWidgets.QCheckBox("Apply gSig_filt")
        self.apply_gsig_filt_check.setToolTip(
            "High-pass filter correlation and PNR images using Motion Correction gSig_filt"
        )
        self.apply_gsig_filt_check.stateChanged.connect(self.update_frame_display)

        # Gain slider
        gain_label = QtWidgets.QLabel("Gain:")
        self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gain_slider.setMinimum(10)  # 0.1x
        self.gain_slider.setMaximum(500)  # 5.0x
        self.gain_slider.setValue(100)  # 1.0x
        self.gain_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.gain_slider.setTickInterval(50)
        self.gain_slider.setMaximumWidth(150)
        self.gain_slider.valueChanged.connect(self.gain_changed)
        self.gain_slider.setEnabled(False)

        self.gain_value_label = QtWidgets.QLabel("1.0x")
        self.gain_value_label.setMinimumWidth(40)

        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.frame_slider)
        controls_layout.addWidget(self.frame_label)
        controls_layout.addWidget(self.show_dff_check)
        controls_layout.addWidget(self.apply_gsig_filt_check)
        controls_layout.addWidget(gain_label)
        controls_layout.addWidget(self.gain_slider)
        controls_layout.addWidget(self.gain_value_label)

        right_layout.addLayout(controls_layout)

        # Create plot widget with two subplots
        self.image_plot = pg.GraphicsLayoutWidget()
        self.image_plot.setMinimumHeight(300)
        self.image_plot.setMaximumHeight(400)

        # Correlation image plot
        self.corr_plot = self.image_plot.addPlot(
            row=0, col=0, title="Correlation Image"
        )
        self.corr_img_item = pg.ImageItem()
        self.corr_plot.addItem(self.corr_img_item)
        self.corr_plot.setAspectLocked()
        self.corr_plot.invertY()
        self.corr_plot.hideAxis("left")
        self.corr_plot.hideAxis("bottom")

        # PNR image plot
        self.pnr_plot = self.image_plot.addPlot(
            row=0, col=1, title="Peak-to-Noise Ratio"
        )
        self.pnr_img_item = pg.ImageItem()
        self.pnr_plot.addItem(self.pnr_img_item)
        self.pnr_plot.setAspectLocked()
        self.pnr_plot.invertY()
        self.pnr_plot.hideAxis("left")
        self.pnr_plot.hideAxis("bottom")

        right_layout.addWidget(self.image_plot)

        # Threshold sliders for correlation and PNR
        threshold_layout = QtWidgets.QHBoxLayout()

        # Correlation threshold slider (coupled with parameter tree)
        self.corr_thresh_label = QtWidgets.QLabel("min_corr:")
        self.corr_thresh_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.corr_thresh_slider.setMinimum(0)
        self.corr_thresh_slider.setMaximum(100)  # 0.00 to 1.00 in steps of 0.01
        self.corr_thresh_slider.setValue(85)  # Default 0.85
        self.corr_thresh_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.corr_thresh_slider.setTickInterval(10)
        self.corr_thresh_slider.valueChanged.connect(self.corr_thresh_changed)
        self.corr_thresh_slider.setEnabled(True)

        self.corr_thresh_value_label = QtWidgets.QLabel("0.85")
        self.corr_thresh_value_label.setMinimumWidth(40)

        # PNR threshold slider (coupled with parameter tree)
        self.pnr_thresh_label = QtWidgets.QLabel("min_pnr:")
        self.pnr_thresh_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pnr_thresh_slider.setMinimum(0)
        self.pnr_thresh_slider.setMaximum(200)  # 0 to 20 in steps of 0.1
        self.pnr_thresh_slider.setValue(100)  # Default 10.0
        self.pnr_thresh_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.pnr_thresh_slider.setTickInterval(20)
        self.pnr_thresh_slider.valueChanged.connect(self.pnr_thresh_changed)
        self.pnr_thresh_slider.setEnabled(True)

        self.pnr_thresh_value_label = QtWidgets.QLabel("10.0")
        self.pnr_thresh_value_label.setMinimumWidth(40)

        threshold_layout.addWidget(self.corr_thresh_label)
        threshold_layout.addWidget(self.corr_thresh_slider)
        threshold_layout.addWidget(self.corr_thresh_value_label)
        threshold_layout.addWidget(self.pnr_thresh_label)
        threshold_layout.addWidget(self.pnr_thresh_slider)
        threshold_layout.addWidget(self.pnr_thresh_value_label)

        right_layout.addLayout(threshold_layout)

        main_layout.addWidget(right_panel)

        # Initialize sliders with parameter values (after sliders are created)
        self._initialize_sliders()
        # Initialize parameter visibility based on pipeline type (must be after UI widgets exist)
        self._on_pipeline_type_changed(
            self.params.child("Data Parameters", "Pipeline Type").value()
        )

        # Setup timer for video playback
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.log("CaImAn Runner initialized. Please select a data directory.")

    def setup_parameters(self):
        """Setup parameter tree with default values."""
        params = [
            {
                "name": "Data Parameters",
                "type": "group",
                "children": [
                    {
                        "name": "Pipeline Type",
                        "type": "list",
                        "values": ["1p (CNMF-E)", "2p (CNMF)"],
                        "value": "1p (CNMF-E)",
                        "tip": "Select imaging modality: 1-photon (microendoscopic, uses CNMF-E) or 2-photon (uses standard CNMF).",
                    },
                    {
                        "name": "Data Directory",
                        "type": "str",
                        "value": "",
                        "readonly": True,
                        "tip": "Directory containing video files (.avi, .tif, .tiff) to process",
                    },
                    {
                        "name": "Base Name",
                        "type": "str",
                        "value": "",
                        "tip": "Base name for output files (memmap, HDF5). If empty, uses the data directory name.",
                    },
                    {
                        "name": "Frame Rate (Hz)",
                        "type": "float",
                        "value": 20.0,
                        "tip": "Acquisition frame rate in Hz (frames per second). Used for temporal analysis and deconvolution.",
                    },
                    {
                        "name": "Decay Time",
                        "type": "float",
                        "value": 0.4,
                        "tip": "Expected decay time constant of calcium indicator (in seconds). Used for deconvolution. Typical values: GCaMP6s ~0.4s, GCaMP6f ~0.2s",
                    },
                    {
                        "name": "dxy",
                        "type": "float",
                        "value": 2.2,
                        "step": 0.1,
                        "tip": "(2p) Spatial resolution in um per pixel.",
                    },
                ],
            },
            {
                "name": "Cluster Parameters",
                "type": "group",
                "children": [
                    {
                        "name": "Number of Processes",
                        "type": "int",
                        "value": 8,
                        "limits": (1, 32),
                        "tip": "Number of parallel processes for computation. Higher values speed up processing but use more CPU/memory.",
                    },
                ],
            },
            {
                "name": "Motion Correction",
                "type": "group",
                "children": [
                    {
                        "name": "gSig_filt",
                        "type": "int",
                        "value": 5,
                        "limits": (1, 33),
                        "tip": "(1p only) Size of Gaussian kernel for spatial filtering before motion correction (in pixels).",
                    },
                    {
                        "name": "max_shifts",
                        "type": "int",
                        "value": 20,
                        "tip": "Maximum allowed shift in pixels for motion correction. Increase if there is significant motion in the video.",
                    },
                    {
                        "name": "strides_mc",
                        "type": "int",
                        "value": 48,
                        "limits": (16, 256),
                        "tip": "Stride size for piecewise-rigid motion correction patches (in pixels).",
                    },
                    {
                        "name": "overlaps_mc",
                        "type": "int",
                        "value": 24,
                        "limits": (8, 128),
                        "tip": "Overlap between piecewise-rigid motion correction patches (in pixels).",
                    },
                ],
            },
            {
                "name": "CNMF-E Parameters",
                "type": "group",
                "children": [
                    {
                        "name": "p",
                        "type": "int",
                        "value": 1,
                        "limits": (0, 2),
                        "tip": "Order of autoregressive model for deconvolution. p=1 for exponential decay, p=2 for double exponential. Usually 1 for calcium imaging.",
                    },
                    {
                        "name": "gSig",
                        "type": "int",
                        "value": 6,
                        "limits": (1, 12),
                        "tip": "Expected half-size of neurons (in pixels). Critical parameter: should match the typical radius of neurons in your data. Use Preview to validate.",
                    },
                    {
                        "name": "gSiz",
                        "type": "int",
                        "value": 15,
                        "tip": "Size of spatial component (in pixels). Typically 2-3 times gSig. Should encompass the full neuron footprint.",
                    },
                    {
                        "name": "min_corr",
                        "type": "float",
                        "value": 0.8,
                        "limits": (0, 1),
                        "step": 0.01,
                        "tip": "Minimum local correlation threshold for neuron detection. Higher values detect only high-quality neurons. Use Preview to adjust based on correlation image.",
                    },
                    {
                        "name": "min_pnr",
                        "type": "float",
                        "value": 10.0,
                        "tip": "Minimum peak-to-noise ratio threshold for neuron detection. Higher values detect only bright, high-SNR neurons. Use Preview to adjust based on PNR image.",
                    },
                    {
                        "name": "patch_size",
                        "type": "int",
                        "value": 48,
                        "limits": (1, 128),
                        "tip": "Size of patches for parallel processing (in pixels). Smaller values use less memory but may be slower. Recommended: 32-64 pixels.",
                    },
                    {
                        "name": "stride",
                        "type": "int",
                        "value": 16,
                        "limits": (1, 128),
                        "tip": "Overlap between patches (in pixels). Should be at least gSiz to avoid boundary effects. Typically patch_size/4.",
                    },
                    {
                        "name": "merge_thr",
                        "type": "float",
                        "value": 0.7,
                        "limits": (0, 1),
                        "step": 0.01,
                        "tip": "Threshold for merging similar components. Higher values merge fewer components (more conservative). Lower values merge more aggressively.",
                    },
                ],
            },
            {
                "name": "CNMF Parameters",
                "type": "group",
                "children": [
                    {
                        "name": "p",
                        "type": "int",
                        "value": 1,
                        "limits": (0, 2),
                        "tip": "Order of autoregressive model for deconvolution. p=1 for exponential decay, p=2 for double exponential.",
                    },
                    {
                        "name": "K",
                        "type": "int",
                        "value": 4,
                        "limits": (1, 50),
                        "tip": "Number of components per patch. Estimate the expected number of neurons visible in a patch-sized area.",
                    },
                    {
                        "name": "gSig",
                        "type": "int",
                        "value": 4,
                        "limits": (1, 20),
                        "tip": "Expected half-size of neurons (in pixels). For 2p data, typically 3-7 pixels depending on zoom and resolution.",
                    },
                    {
                        "name": "gSiz",
                        "type": "int",
                        "value": 9,
                        "tip": "Size of spatial component (in pixels). Typically 2*gSig+1.",
                    },
                    {
                        "name": "nb",
                        "type": "int",
                        "value": 2,
                        "limits": (0, 5),
                        "tip": "Number of global background components. Typically 2 for 2p imaging.",
                    },
                    {
                        "name": "patch_size",
                        "type": "int",
                        "value": 40,
                        "limits": (1, 128),
                        "tip": "Size of patches (half-size of patch in pixels). Should be large enough to contain several neurons.",
                    },
                    {
                        "name": "stride",
                        "type": "int",
                        "value": 20,
                        "limits": (1, 128),
                        "tip": "Overlap between patches (in pixels). Typically patch_size/2.",
                    },
                    {
                        "name": "merge_thr",
                        "type": "float",
                        "value": 0.85,
                        "limits": (0, 1),
                        "step": 0.01,
                        "tip": "Threshold for merging similar components.",
                    },
                    {
                        "name": "method_init",
                        "type": "list",
                        "values": ["greedy_roi", "sparse_nmf"],
                        "value": "greedy_roi",
                        "tip": "Initialization method for CNMF. greedy_roi is standard for 2p data.",
                    },
                ],
            },
            {
                "name": "Quality Control",
                "type": "group",
                "children": [
                    {
                        "name": "min_SNR",
                        "type": "float",
                        "value": 2.0,
                        "tip": "Minimum signal-to-noise ratio for accepting components. Components with SNR below this are rejected. Typical range: 1.0-3.0.",
                    },
                    {
                        "name": "rval_thr",
                        "type": "float",
                        "value": 0.85,
                        "limits": (0, 1),
                        "step": 0.01,
                        "tip": "Minimum spatial correlation (r-value) for accepting components. Measures how well the spatial footprint matches the data. Range: 0-1, higher is better.",
                    },
                    {
                        "name": "use_cnn",
                        "type": "bool",
                        "value": True,
                        "tip": "Use CNN-based classifier for component evaluation. Recommended for 1p data. For 2p, can be disabled.",
                    },
                ],
            },
            {
                "name": "Detrend Parameters",
                "type": "group",
                "children": [
                    {
                        "name": "quantileMin",
                        "type": "float",
                        "value": 8.0,
                        "limits": (0, 100),
                        "tip": "Quantile used to estimate the baseline (values in [0,100]). Used for DF/F normalization.",
                    },
                    {
                        "name": "frames_window",
                        "type": "int",
                        "value": 500,
                        "limits": (1, 10000),
                        "tip": "Number of frames for computing running quantile. Larger windows provide smoother baselines but are slower.",
                    },
                    {
                        "name": "use_residuals",
                        "type": "bool",
                        "value": False,
                        "tip": "Flag for using non-deconvolved traces (C + YrA) in DF/F calculation.",
                    },
                ],
            },
        ]
        self.params = Parameter.create(name="params", type="group", children=params)
        self.param_tree.setParameters(self.params, showTop=False)

        # Connect pipeline type change to update visible parameters
        self.params.child("Data Parameters", "Pipeline Type").sigValueChanged.connect(
            lambda param, value: self._on_pipeline_type_changed(value)
        )
        # NOTE: _on_pipeline_type_changed is called later in __init__ after UI widgets are created

        # Connect parameter changes to sync with sliders (bidirectional sync for min_corr and min_pnr)
        self.params.child("CNMF-E Parameters", "min_corr").sigValueChanged.connect(
            lambda param, value: self._sync_min_corr_from_param(value)
        )
        self.params.child("CNMF-E Parameters", "min_pnr").sigValueChanged.connect(
            lambda param, value: self._sync_min_pnr_from_param(value)
        )

    def _initialize_sliders(self):
        """Initialize slider values from parameter tree (called after UI is created)."""
        min_corr_val = self.params.child("CNMF-E Parameters", "min_corr").value()
        min_pnr_val = self.params.child("CNMF-E Parameters", "min_pnr").value()
        self.corr_thresh_slider.setValue(int(min_corr_val * 100))
        self.pnr_thresh_slider.setValue(int(min_pnr_val * 10))
        self.corr_thresh_value_label.setText(f"{min_corr_val:.2f}")
        self.pnr_thresh_value_label.setText(f"{min_pnr_val:.1f}")

    def is_2p(self):
        """Check if current pipeline type is 2-photon."""
        return (
            self.params.child("Data Parameters", "Pipeline Type").value() == "2p (CNMF)"
        )

    def _on_pipeline_type_changed(self, value):
        """Show/hide parameter groups based on pipeline type."""
        is_2p = value == "2p (CNMF)"
        # Show/hide CNMF-E vs CNMF parameter groups
        self.params.child("CNMF-E Parameters").setOpts(visible=not is_2p)
        self.params.child("CNMF Parameters").setOpts(visible=is_2p)
        # Update 2p-specific data params
        self.params.child("Data Parameters", "dxy").setOpts(visible=is_2p)
        self.params.child("Data Parameters", "Frame Rate (Hz)").setValue(
            30.137 if is_2p else 20.0
        )
        # Show/hide 1p-specific motion correction params
        self.params.child("Motion Correction", "gSig_filt").setOpts(visible=not is_2p)
        # strides_mc and overlaps_mc are always visible (used for both 1p and 2p)
        # Show/hide correlation/PNR threshold sliders (only for 1p)
        self.corr_thresh_label.setVisible(not is_2p)
        self.corr_thresh_slider.setVisible(not is_2p)
        self.corr_thresh_value_label.setVisible(not is_2p)
        self.pnr_thresh_label.setVisible(not is_2p)
        self.pnr_thresh_slider.setVisible(not is_2p)
        self.pnr_thresh_value_label.setVisible(not is_2p)
        # Update PNR plot title for 2p
        if is_2p:
            self.pnr_plot.setTitle("(Not used for 2p)")
        else:
            self.pnr_plot.setTitle("Peak-to-Noise Ratio")

    def populate_file_list(self, avi_files):
        """Populate the file list widget with AVI files."""
        self.file_list.clear()
        self.selected_files = []

        for file_path in avi_files:
            item = QtWidgets.QListWidgetItem(Path(file_path).name)
            item.setCheckState(QtCore.Qt.Checked)  # All files selected by default
            item.setData(QtCore.Qt.UserRole, file_path)  # Store full path
            self.file_list.addItem(item)
            self.selected_files.append(file_path)

        # Enable select/deselect buttons when files are loaded
        has_files = len(avi_files) > 0
        self.select_all_files_btn.setEnabled(has_files)
        self.deselect_all_files_btn.setEnabled(has_files)

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

    def on_file_clicked(self, current, _previous):
        """Switch video preview when a different file is clicked."""
        if current is None:
            return
        file_path = current.data(QtCore.Qt.UserRole)
        if file_path and file_path != self._preview_video_path:
            self.load_video_preview(file_path)

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

    def _format_duration(self, duration):
        """Format duration in seconds to human-readable string."""
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def log(self, message):
        """Add message to progress log."""
        print(message)
        if self.save_logger is not None:
            self.save_logger.info(message)
        self.progress_text.append(message)
        QtWidgets.QApplication.processEvents()

    def setup_logger_directory(self, directory):
        """Setup logger to write to the specified directory. Only called when directory is selected."""
        if directory is None or not directory:
            return

        dir_path = Path(directory)
        if not dir_path.exists():
            return

        # Remove existing file handlers from the global logger (if any)
        global logger
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)

        # Create new log file in the selected directory
        current_datetime = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
        log_filename = "caiman_runner" + current_datetime + ".log"
        log_path = dir_path / log_filename

        # Add new file handler - this is when logging actually starts
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        self.log(f"Logger initialized: {log_path}")

    def select_directory(self):
        """Open dialog to select data directory."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Data Directory", ""
        )
        if directory:
            self.data_path = Path(directory)
            self.caiman_path = self.data_path / "caiman"
            self.caiman_path.mkdir(parents=True, exist_ok=True)
            os.environ["CAIMAN_TEMP"] = str(self.caiman_path)

            base_val = self.params.child("Data Parameters", "Base Name").value().strip()
            self.base_name = base_val if base_val else self.data_path.name
            self.params.child("Data Parameters", "Base Name").setValue(self.base_name)

            self.params.child("Data Parameters", "Data Directory").setValue(
                str(self.data_path)
            )

            # Update logger to write to selected directory
            self.setup_logger_directory(self.caiman_path)
            self.log(f"Selected directory: {self.data_path}")

            if self.is_2p():
                # 2p: look for Prairie View XML to extract metadata and TIFF files
                pv_tree = None
                pv_xml_path = None
                for xf in self.data_path.glob("*.xml"):
                    try:
                        tree = ET.parse(xf)
                        if tree.getroot().tag == "PVScan":
                            pv_tree = tree
                            pv_xml_path = xf
                            break
                    except ET.ParseError:
                        continue

                if pv_tree is not None:
                    try:
                        pv_meta = self._parse_prairie_view_xml(
                            pv_xml_path, tree=pv_tree
                        )
                        self.log(f"Loaded Prairie View XML: {pv_xml_path.name}")
                        if pv_meta["framerate"] is not None:
                            self.params.child(
                                "Data Parameters", "Frame Rate (Hz)"
                            ).setValue(pv_meta["framerate"])
                            self.log(f"  Frame rate: {pv_meta['framerate']:.4f} Hz")
                        if pv_meta["dxy"] is not None:
                            self.params.child("Data Parameters", "dxy").setValue(
                                pv_meta["dxy"]
                            )
                            self.log(
                                f"  Spatial resolution: {pv_meta['dxy']:.4f} um/pixel"
                            )
                        video_files = pv_meta["tiff_files"]
                    except Exception as e:
                        self.log(f"Error parsing Prairie View XML: {e}")
                        video_files = []
                else:
                    self.log(
                        "Warning: No Prairie View XML found, looking for TIFF files directly"
                    )
                    video_files = []
                    for ext in ["*.tif", "*.tiff"]:
                        video_files.extend(
                            [str(f.resolve()) for f in self.data_path.glob(ext)]
                        )
                    video_files = natsorted(video_files)
            else:
                # 1p: load metadata from metaData.json and find video files
                meta_path = self.data_path / "metaData.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                        framerate = float(
                            str(meta["frameRate"]).upper().replace("FPS", "").strip()
                        )
                        self.params.child(
                            "Data Parameters", "Frame Rate (Hz)"
                        ).setValue(framerate)
                        self.log(f"Loaded frame rate from metadata: {framerate:.2f} Hz")
                    except Exception as e:
                        self.log(f"Could not load metadata: {e}")

                video_files = []
                for ext in ["*.avi", "*.tif", "*.tiff"]:
                    video_files.extend(
                        [str(f.resolve()) for f in self.data_path.glob(ext)]
                    )
                video_files = natsorted(video_files)

            if video_files:
                self.log(f"Found {len(video_files)} video files")
                self.populate_file_list(video_files)
                if video_files:
                    self.load_video_preview(video_files[0])
            else:
                self.log("Warning: No video files found in directory")
                self.file_list.clear()
                self.run_btn.setEnabled(False)
                self.preview_btn.setEnabled(False)
                self.select_all_files_btn.setEnabled(False)
                self.deselect_all_files_btn.setEnabled(False)

    def _parse_prairie_view_xml(self, xml_path, tree=None):
        """Parse a Prairie View XML file to extract metadata and TIFF file paths."""
        if tree is None:
            tree = ET.parse(xml_path)
        root = tree.getroot()
        xml_dir = Path(xml_path).parent

        # Extract framerate from framePeriod in PVStateShard
        framerate = None
        for sv in root.iter("PVStateValue"):
            if sv.get("key") == "framePeriod":
                frame_period = float(sv.get("value"))
                framerate = 1.0 / frame_period
                break

        # Extract micronsPerPixel (dxy)
        dxy = None
        for sv in root.iter("PVStateValue"):
            if sv.get("key") == "micronsPerPixel":
                for iv in sv.findall("IndexedValue"):
                    if iv.get("index") == "XAxis":
                        dxy = float(iv.get("value"))
                        break
                if dxy is not None:
                    break

        # Collect unique TIFF filenames referenced in Frame elements
        tiff_files = []
        seen = set()
        for frame in root.iter("Frame"):
            for f in frame.findall("File"):
                fname = f.get("filename")
                if fname and fname not in seen:
                    seen.add(fname)
                    full_path = xml_dir / fname
                    if full_path.exists():
                        tiff_files.append(str(full_path.resolve()))

        return {
            "framerate": framerate,
            "dxy": dxy,
            "tiff_files": tiff_files,
        }

    def _is_tiff(self, path):
        """Check if file is a TIFF file."""
        return str(path).lower().endswith((".tif", ".tiff"))

    def _frame_to_pixmap(self, frame, frame_min=None, frame_max=None):
        """Normalize a frame to uint8 and return a QPixmap scaled to the video label."""
        if frame_min is None:
            frame_min = np.percentile(frame, 1)
        if frame_max is None:
            frame_max = np.percentile(frame, 99.5)
        frame_clipped = np.clip(frame, frame_min, frame_max)
        if frame_max > frame_min:
            frame_norm = (
                (frame_clipped - frame_min) / (frame_max - frame_min) * 255
            ).astype(np.uint8)
        else:
            frame_norm = np.clip(frame_clipped, 0, 255).astype(np.uint8)
        height, width = frame_norm.shape
        q_image = QtGui.QImage(
            frame_norm.data, width, height, width, QtGui.QImage.Format_Grayscale8
        )
        pixmap = QtGui.QPixmap.fromImage(q_image)
        label_size = self.video_label.size()
        return pixmap.scaled(
            label_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )

    def load_video_preview(self, video_path, max_frames=1000):
        """Load only metadata and first frame for preview; full video loads when Play is pushed."""
        try:
            self.log(f"Loading video preview: {Path(video_path).name}")

            if self._is_tiff(video_path):
                import tifffile

                with tifffile.TiffFile(str(video_path)) as tif:
                    frame_count = len(tif.pages)
                    if frame_count <= 0:
                        self.video_label.setText("Could not get frame count")
                        return
                    frame = tif.pages[0].asarray().astype(np.float32)
                    if len(frame.shape) == 3:
                        frame = np.mean(frame, axis=-1)
            else:
                import cv2

                cap = cv2.VideoCapture(str(video_path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count <= 0:
                    cap.release()
                    self.video_label.setText("Could not get frame count")
                    return
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    self.video_label.setText("Could not load first frame")
                    return
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = frame.astype(np.float32)

            self._first_frame = frame
            self._preview_video_path = str(video_path)
            self._frame_count = min(frame_count, max_frames)  # cap for loading on Play

            self.video_frames = None  # Full video loads on Play
            self.F0 = None
            self.video_min = None
            self.video_max = None
            self.dff_min = None
            self.dff_max = None
            self.display_cache = {}
            self.current_frame_idx = 0
            self.frame_slider.setMaximum(self._frame_count - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(True)
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.show_dff_check.setEnabled(True)
            self.gain_slider.setEnabled(True)

            self.update_frame_display()
            self.log(
                f"Preview ready: {self._frame_count} frames (press Play to load video)"
            )

        except Exception as e:
            self.log(f"Error loading video: {str(e)}")
            self.video_label.setText(f"Error loading video: {str(e)}")

    def _load_full_video(self):
        """Load full video from _preview_video_path and compute F0, min, max (called on Play)."""
        if not self._preview_video_path or self._frame_count <= 0:
            return
        try:
            self.log("Loading video...")
            frames = []
            if self._is_tiff(self._preview_video_path):
                import tifffile

                with tifffile.TiffFile(self._preview_video_path) as tif:
                    for i, page in enumerate(tif.pages):
                        if i >= self._frame_count:
                            break
                        frame = page.asarray().astype(np.float32)
                        if len(frame.shape) == 3:
                            frame = np.mean(frame, axis=-1)
                        frames.append(frame)
            else:
                import cv2

                cap = cv2.VideoCapture(self._preview_video_path)
                for _ in range(self._frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if len(frame.shape) == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(frame.astype(np.float32))
                cap.release()
            if len(frames) == 0:
                self.log("Could not load any frames")
                return
            self.video_frames = np.array(frames)
            self.log("Calculating F0...")
            self.F0 = np.percentile(self.video_frames, 20, axis=0)
            self.F0 = np.maximum(self.F0, 1e-6)
            self.log("Calculating video statistics...")
            self.video_min = np.percentile(self.video_frames, 1)
            self.video_max = np.percentile(self.video_frames, 99.5)
            self.dff_min = 0
            self.dff_max = 100
            self.display_cache = {}
            self.log(f"Loaded {len(self.video_frames)} frames")
        except Exception as e:
            self.log(f"Error loading video: {str(e)}")

    def gain_changed(self, value):
        """Handle gain slider change."""
        self.gain = value / 100.0  # Convert slider value (10-500) to gain (0.1-5.0)
        self.gain_value_label.setText(f"{self.gain:.2f}x")
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
            self.corr_thresh_value_label.setText(f"{value:.2f}")
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
            self.pnr_thresh_value_label.setText(f"{value:.1f}")
            self.update_image_plots()
        finally:
            self._syncing_min_pnr = False

    def _apply_gsig_filt(self, frame):
        """Apply high-pass spatial filter if the checkbox is checked."""
        if self.apply_gsig_filt_check.isChecked():
            gSig_filt = self.params.child("Motion Correction", "gSig_filt").value()
            return high_pass_filter_space(frame, gSig_filt=(gSig_filt, gSig_filt))
        return frame

    def update_frame_display(self):
        """Update the displayed frame (original or dF/F). When video not loaded, show _first_frame."""
        # When full video not loaded, show first frame only (no F0/dF/F)
        if self.video_frames is None or len(self.video_frames) == 0:
            if self._first_frame is not None:
                frame = self._apply_gsig_filt(self._first_frame * self.gain)
                self.video_label.setPixmap(self._frame_to_pixmap(frame))
                self.frame_label.setText(
                    f"Frame: {self.current_frame_idx + 1}/{self._frame_count} (press Play to load)"
                )
            return

        use_dff = self.show_dff_check.isChecked()
        apply_gsig_filt = self.apply_gsig_filt_check.isChecked()
        label_size = self.video_label.size()
        cache_key = (
            self.current_frame_idx,
            use_dff,
            self.gain,
            apply_gsig_filt,
            label_size.width(),
            label_size.height(),
        )

        # Check cache first
        if cache_key in self.display_cache:
            pixmap = self.display_cache[cache_key]
        else:
            if use_dff:
                frame = (
                    100
                    * (self.video_frames[self.current_frame_idx] - self.F0)
                    / self.F0
                )
                frame_min, frame_max = self.dff_min, self.dff_max
            else:
                frame = self.video_frames[self.current_frame_idx]
                frame_min, frame_max = self.video_min, self.video_max

            frame = frame * self.gain
            frame = self._apply_gsig_filt(frame)
            if apply_gsig_filt:
                frame_min, frame_max = None, None  # auto-compute in _frame_to_pixmap

            pixmap = self._frame_to_pixmap(frame, frame_min, frame_max)

            # Cache the scaled pixmap (limit cache size to avoid memory issues)
            if len(self.display_cache) > 100:
                keys_to_remove = list(self.display_cache.keys())[:50]
                for key in keys_to_remove:
                    del self.display_cache[key]

            self.display_cache[cache_key] = pixmap

        self.video_label.setPixmap(pixmap)
        self.frame_label.setText(
            f"Frame: {self.current_frame_idx + 1}/{len(self.video_frames)}"
        )

    def toggle_play(self):
        """Toggle video playback. Load full video and compute F0/min/max on first Play."""
        if self._preview_video_path is None:
            return
        # Load video only when Play is pushed (and compute F0, min, max)
        if self.video_frames is None:
            self._load_full_video()
            if self.video_frames is None:
                return
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText("Play")
        else:
            framerate = self.params.child("Data Parameters", "Frame Rate (Hz)").value()
            interval = int(1000 / framerate)  # milliseconds per frame
            self.timer.start(interval)
            self.is_playing = True
            self.play_btn.setText("Pause")

    def stop_video(self):
        """Stop video playback, clear loaded variables to save memory, reset to first frame."""
        self.timer.stop()
        self.is_playing = False
        self.play_btn.setText("Play")
        self.current_frame_idx = 0
        self.frame_slider.setValue(0)
        self.clear_video_cache()
        self.update_frame_display()

    def clear_video_cache(self):
        """Clear video-related caches."""
        self.video_frames = None
        self.F0 = None
        self.video_min = None
        self.video_max = None
        self.dff_min = None
        self.dff_max = None
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
        self.current_frame_idx = value
        self.update_frame_display()

    def corr_thresh_changed(self, value):
        """Handle correlation threshold slider change - sync with parameter tree."""
        if self._syncing_min_corr:
            return
        min_corr = value / 100.0  # Convert slider value (0-100) to threshold (0.0-1.0)
        self.corr_thresh_value_label.setText(f"{min_corr:.2f}")
        # Update parameter value
        self._syncing_min_corr = True
        try:
            self.params.child("CNMF-E Parameters", "min_corr").setValue(min_corr)
        finally:
            self._syncing_min_corr = False
        self.update_image_plots()

    def pnr_thresh_changed(self, value):
        """Handle PNR threshold slider change - sync with parameter tree."""
        if self._syncing_min_pnr:
            return
        min_pnr = value / 10.0  # Convert slider value (0-200) to threshold (0.0-20.0)
        self.pnr_thresh_value_label.setText(f"{min_pnr:.1f}")
        # Update parameter value
        self._syncing_min_pnr = True
        try:
            self.params.child("CNMF-E Parameters", "min_pnr").setValue(min_pnr)
        finally:
            self._syncing_min_pnr = False
        self.update_image_plots()

    @staticmethod
    def _threshold_image(img, threshold, clip_max):
        """Create an RGB display image with sub-threshold pixels highlighted in red."""
        mask = img < threshold
        clipped = np.clip(img, 0, clip_max)
        gray = (clipped / clip_max * 255).astype(np.uint8)
        display = np.stack([gray, gray, gray], axis=-1)
        display[mask, 0] = 255
        display[mask, 1] = 0
        display[mask, 2] = 0
        return display

    def update_image_plots(self):
        """Update correlation and PNR image plots with thresholding."""
        gSig = self.params.child("CNMF-E Parameters", "gSig").value()

        if self.correlation_image is not None:
            min_corr = self.params.child("CNMF-E Parameters", "min_corr").value()
            self.corr_img_item.setImage(
                self._threshold_image(self.correlation_image, min_corr, 1.0),
                autoLevels=False,
            )
            self.corr_plot.setTitle(
                f"Correlation Image (gSig={gSig}, min_corr={min_corr:.2f})"
            )

        if self.peak_to_noise_ratio is not None:
            min_pnr = self.params.child("CNMF-E Parameters", "min_pnr").value()
            self.pnr_img_item.setImage(
                self._threshold_image(self.peak_to_noise_ratio, min_pnr, 20.0),
                autoLevels=False,
            )
            self.pnr_plot.setTitle(
                f"Peak-to-Noise Ratio (gSig={gSig}, min_pnr={min_pnr:.1f})"
            )

    def preview_parameters(self):
        """Preview parameters using first 500 frames to validate gSig."""
        if not self.data_path or not self.data_path.exists():
            QtWidgets.QMessageBox.warning(
                self, "Error", "Please select a valid data directory"
            )
            return

        # Get selected files
        if not self.selected_files:
            QtWidgets.QMessageBox.warning(
                self, "Error", "Please select at least one file to preview"
            )
            return

        files = natsorted(self.selected_files)

        self.preview_btn.setEnabled(False)
        self.log("\n" + "=" * 50)
        self.log("Previewing Parameters (first 500 frames)")
        self.log("=" * 50)

        try:
            # Get gSig from the appropriate parameter group
            if self.is_2p():
                gSig = self.params.child("CNMF Parameters", "gSig").value()
                self.log(f"Testing CNMF gSig parameter: {gSig} (2p mode)")
            else:
                gSig = self.params.child("CNMF-E Parameters", "gSig").value()
                self.log(f"Testing CNMF-E gSig parameter: {gSig} (1p mode)")

            # Load first 500 frames from first file
            self.log(f"Loading first 500 frames from: {Path(files[0]).name}")
            frames = []
            max_frames = 500

            if self._is_tiff(files[0]):
                import tifffile

                with tifffile.TiffFile(files[0]) as tif:
                    for i, page in enumerate(tif.pages):
                        if i >= max_frames:
                            break
                        frame = page.asarray().astype(np.float32)
                        if len(frame.shape) == 3:
                            frame = np.mean(frame, axis=-1)
                        frames.append(frame)
            else:
                import cv2

                cap = cv2.VideoCapture(files[0])
                for i in range(max_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if len(frame.shape) == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(frame.astype(np.float32))
                cap.release()

            if len(frames) == 0:
                raise ValueError("Could not load any frames from video file")

            images = np.array(frames)  # Shape: (T, H, W)
            self.log(f"Loaded {len(frames)} frames, shape: {images.shape}")

            if self.is_2p():
                # 2p: compute correlation image only (no PNR needed)
                self.log("\nComputing correlation image...")
                correlation_image = cm.local_correlations(images, swap_dim=False)
                correlation_image[np.isnan(correlation_image)] = 0
                self.correlation_image = correlation_image
                self.peak_to_noise_ratio = None

                # Display correlation image
                self.corr_img_item.setImage(correlation_image, autoLevels=True)
                self.corr_plot.setTitle(f"Correlation Image (gSig={gSig})")
                self.pnr_img_item.clear()
                self.pnr_plot.setTitle("(Not used for 2p)")

                corr_mean = np.mean(correlation_image)
                corr_max_val = np.max(correlation_image)
                corr_std = np.std(correlation_image)
                self.log(
                    f"  Correlation image - Mean: {corr_mean:.3f}, Max: {corr_max_val:.3f}, Std: {corr_std:.3f}"
                )
                self.log(
                    "\nPreview complete. Correlation image displayed on the right side."
                )
            else:
                # 1p: compute correlation and PNR images
                self.log("\nComputing correlation and PNR images...")
                correlation_image, peak_to_noise_ratio = (
                    cm.summary_images.correlation_pnr(images, gSig=gSig, swap_dim=False)
                )

                self.correlation_image = correlation_image
                self.peak_to_noise_ratio = peak_to_noise_ratio

                corr_max = float(np.max(correlation_image))
                pnr_max = float(np.max(peak_to_noise_ratio))

                self.corr_thresh_slider.setMaximum(int(corr_max * 100))
                self.pnr_thresh_slider.setMaximum(int(pnr_max * 10))

                min_corr_val = self.params.child(
                    "CNMF-E Parameters", "min_corr"
                ).value()
                min_pnr_val = self.params.child("CNMF-E Parameters", "min_pnr").value()
                self.corr_thresh_slider.setValue(int(min_corr_val * 100))
                self.pnr_thresh_slider.setValue(int(min_pnr_val * 10))
                self.corr_thresh_value_label.setText(f"{min_corr_val:.2f}")
                self.pnr_thresh_value_label.setText(f"{min_pnr_val:.1f}")

                self.update_image_plots()

                corr_mean = np.mean(correlation_image)
                corr_max = np.max(correlation_image)
                corr_std = np.std(correlation_image)
                pnr_mean = np.mean(peak_to_noise_ratio)
                pnr_max = np.max(peak_to_noise_ratio)

                self.log(
                    f"  Correlation image - Mean: {corr_mean:.3f}, Max: {corr_max:.3f}, Std: {corr_std:.3f}"
                )
                self.log(f"  PNR image - Mean: {pnr_mean:.2f}, Max: {pnr_max:.2f}")
                self.log("\nPreview complete. Images displayed on the right side.")

        except Exception as e:
            self.log(f"\nERROR during preview: {str(e)}")
            import traceback

            self.log(traceback.format_exc())
            QtWidgets.QMessageBox.critical(
                self, "Preview Error", f"Preview failed:\n{str(e)}"
            )
        finally:
            self.preview_btn.setEnabled(True)

    def run_pipeline(self):
        """Run the complete CaImAn processing pipeline."""
        if not self.data_path or not self.data_path.exists():
            QtWidgets.QMessageBox.warning(
                self, "Error", "Please select a valid data directory"
            )
            return

        self.run_btn.setEnabled(False)
        # Free preview video memory before running the pipeline
        self.timer.stop()
        self.is_playing = False
        self.play_btn.setText("Play")
        self.clear_video_cache()
        self._first_frame = None
        self._preview_video_path = None
        self.frame_slider.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.video_label.clear()

        start_time = time.time()
        self.log("\n" + "=" * 50)
        self.log("Starting CaImAn Processing Pipeline")
        self.log("=" * 50)

        try:
            # Step 1: Setup cluster
            step_start = time.time()
            self.log("\n[1/5] Setting up cluster...")
            self.setup_cluster()
            step_duration = time.time() - step_start
            self.log(
                f"  ✓ Cluster setup completed in {self._format_duration(step_duration)}"
            )

            # Step 2: Motion correction
            step_start = time.time()
            self.log("\n[2/5] Running motion correction...")
            files_mc = self.run_motion_correction()
            step_duration = time.time() - step_start
            self.log(
                f"  ✓ Motion correction completed in {self._format_duration(step_duration)}"
            )

            # Restart cluster to clear stale worker state
            self.log("Restarting cluster for source extraction...")
            cm.stop_server(dview=self.cluster)
            _, self.cluster, _ = cm.cluster.setup_cluster(
                backend="multiprocessing",
                n_processes=self.n_processes,
                ignore_preexisting=False,
            )

            # Step 3: Source extraction (CNMF-E for 1p, CNMF for 2p)
            step_start = time.time()
            if self.is_2p():
                self.log("\n[3/5] Running CNMF (2p)...")
                self.run_cnmf_2p(files_mc)
                step_duration = time.time() - step_start
                self.log(
                    f"  ✓ CNMF completed in {self._format_duration(step_duration)}"
                )
            else:
                self.log("\n[3/5] Running CNMF-E...")
                self.run_cnmfe(files_mc)
                step_duration = time.time() - step_start
                self.log(
                    f"  ✓ CNMF-E completed in {self._format_duration(step_duration)}"
                )

            # Step 4: Quality control
            step_start = time.time()
            self.log("\n[4/5] Running quality control...")
            self.run_quality_control(files_mc)
            step_duration = time.time() - step_start
            self.log(
                f"  ✓ Quality control completed in {self._format_duration(step_duration)}"
            )

            # Step 5: Save results
            step_start = time.time()
            self.log("\n[5/5] Saving results...")
            self.save_results(files_mc)
            step_duration = time.time() - step_start
            self.log(f"  ✓ Results saved in {self._format_duration(step_duration)}")

            # Calculate and display total duration
            end_time = time.time()
            total_duration = end_time - start_time
            duration_str = self._format_duration(total_duration)

            self.log("\n" + "=" * 50)
            self.log("Processing completed successfully!")
            self.log(f"Total duration: {duration_str} ({total_duration:.2f} seconds)")
            self.log("=" * 50)

            QtWidgets.QMessageBox.information(
                self,
                "Success",
                f"Processing completed!\nDuration: {duration_str}\n\nResults saved to:\n{self.caiman_path}",
            )

            # Automatically load the saved file in the main GUI if parent is MainWindow
            parent = self.parent()
            if parent is not None and hasattr(parent, "load_data"):
                try:
                    self.log(f"\nLoading saved file in main GUI: {self.save_path}")
                    parent.load_data(click=False, filepath=str(self.save_path))
                    self.log("File loaded successfully in main GUI")
                except Exception as e:
                    self.log(f"Warning: Could not auto-load file in main GUI: {str(e)}")

        except Exception as e:
            # Calculate and display duration even on error
            end_time = time.time()
            total_duration = end_time - start_time
            duration_str = self._format_duration(total_duration)

            self.log(f"\nERROR: {str(e)}")
            self.log(f"Processing failed after {duration_str}")
            import traceback

            self.log(traceback.format_exc())
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Processing failed after {duration_str}:\n{str(e)}"
            )
        finally:
            # Cleanup
            if self.cluster is not None:
                self.log("\nCleaning up cluster...")
                cm.stop_server(dview=self.cluster)
                self.cluster = None
            self.run_btn.setEnabled(True)

    def setup_cluster(self):
        """Setup multiprocessing cluster."""
        n_processes = self.params.child(
            "Cluster Parameters", "Number of Processes"
        ).value()
        self.n_processes = n_processes

        # check available cores
        available_cores = os.cpu_count()
        if n_processes > available_cores:
            n_processes = available_cores - 1
            self.params.child("Cluster Parameters", "Number of Processes").setValue(
                n_processes
            )
            self.log(
                f"Number of processes set to {n_processes} (available cores: {available_cores}"
            )

        # check available memory
        available_memory = (
            psutil.virtual_memory().available / 1024**3
        )  # in GB, use free/available memory
        param_group = "CNMF Parameters" if self.is_2p() else "CNMF-E Parameters"
        patch_size = self.params.child(param_group, "patch_size").value()
        stride = self.params.child(param_group, "stride").value()
        _, T = cm.base.movies.get_file_size(self.selected_files)
        memory_usage_per_process = (
            2 * 4 * np.sum(T) * (patch_size + stride) ** 2 / 1024**3
        )  # in GB
        n_process_max = int(np.floor(available_memory / memory_usage_per_process)) - 1
        self.log(f"Available memory: {available_memory:.2f} GB")
        self.log(f"Memory usage per process: {memory_usage_per_process:.2f} GB")
        self.log(
            f"Max number of processes: {n_process_max} (based on available memory)"
        )
        if n_processes > n_process_max:
            self.log(
                f"Available memory is too low for {n_processes} processes. Setting to {n_process_max} processes."
            )
            n_processes = n_process_max
            self.params.child("Cluster Parameters", "Number of Processes").setValue(
                n_processes
            )

        if self.cluster is not None:
            cm.stop_server(dview=self.cluster)

        _, self.cluster, n_processes = cm.cluster.setup_cluster(
            backend="multiprocessing", n_processes=n_processes, ignore_preexisting=False
        )
        self.log(f"Cluster with {n_processes} processes set up.")

    def run_motion_correction(self):
        """Run motion correction."""
        if not self.selected_files:
            raise ValueError("No files selected for processing")

        files = self.selected_files
        self.log(f"Found {len(files)} files")

        framerate = self.params.child("Data Parameters", "Frame Rate (Hz)").value()
        decay_time = self.params.child("Data Parameters", "Decay Time").value()
        max_shifts = self.params.child("Motion Correction", "max_shifts").value()

        if self.is_2p():
            # 2p: piecewise-rigid motion correction (no gSig_filt)
            dxy = self.params.child("Data Parameters", "dxy").value()
            strides_mc = self.params.child("Motion Correction", "strides_mc").value()
            overlaps_mc = self.params.child("Motion Correction", "overlaps_mc").value()

            data_params = {
                "data": {
                    "fnames": files,
                    "fr": framerate,
                    "decay_time": decay_time,
                    "dxy": (dxy, dxy),
                },
                "motion": {
                    "pw_rigid": True,
                    "max_shifts": (max_shifts, max_shifts),
                    "strides": (strides_mc, strides_mc),
                    "overlaps": (overlaps_mc, overlaps_mc),
                    "border_nan": "copy",
                },
            }
        else:
            # 1p: rigid motion correction with gSig_filt
            gSig_filt = self.params.child("Motion Correction", "gSig_filt").value()
            strides_mc = self.params.child("Motion Correction", "strides_mc").value()
            overlaps_mc = self.params.child("Motion Correction", "overlaps_mc").value()
            data_params = {
                "data": {
                    "fnames": files,
                    "fr": framerate,
                    "decay_time": decay_time,
                },
                "motion": {
                    "gSig_filt": (gSig_filt, gSig_filt),
                    "max_shifts": (max_shifts, max_shifts),
                    "strides": (strides_mc, strides_mc),
                    "overlaps": (overlaps_mc, overlaps_mc),
                    "border_nan": "copy",
                },
            }

        parameters = cnmf.params.CNMFParams(params_dict=data_params)

        mot_correct = MotionCorrect(
            files, dview=self.cluster, **parameters.get_group("motion")
        )
        mot_correct.motion_correct(save_movie=True)

        # Save shifts
        base_val = self.params.child("Data Parameters", "Base Name").value()
        base = (base_val or "").strip() if base_val else ""
        base = base or self.data_path.name
        shifts_path = self.caiman_path / f"{base}_shifts_rig.npy"
        np.save(shifts_path, mot_correct.shifts_rig)
        self.log(f"Saved rigid shifts to {shifts_path}")

        # Memory map the motion-corrected file
        border_to_0 = 0 if mot_correct.border_nan == "copy" else mot_correct.border_to_0

        # Use pw_rigid results if available, otherwise rigid
        mc_files = (
            mot_correct.fname_tot_els
            if self.is_2p()
            and hasattr(mot_correct, "fname_tot_els")
            and mot_correct.fname_tot_els
            else mot_correct.fname_tot_rig
        )

        files_mc = cm.save_memmap(
            mc_files,
            base_name=str(self.caiman_path / (self.base_name + "_")),
            order="C",
            border_to_0=border_to_0,
            dview=self.cluster,
        )

        self.log(f"Motion correction completed. Saved to: {files_mc}")
        return files_mc

    def run_cnmfe(self, files_mc):
        """Run CNMF-E processing."""
        # Load motion corrected movie
        Yr, dims, T = cm.load_memmap(files_mc)
        images = Yr.T.reshape((T,) + dims, order="F")

        # Get parameters
        gSig = self.params.child("CNMF-E Parameters", "gSig").value()
        gSiz = self.params.child("CNMF-E Parameters", "gSiz").value()
        min_corr = self.params.child("CNMF-E Parameters", "min_corr").value()
        min_pnr = self.params.child("CNMF-E Parameters", "min_pnr").value()
        merge_thr = self.params.child("CNMF-E Parameters", "merge_thr").value()
        p = self.params.child("CNMF-E Parameters", "p").value()
        patch_size = self.params.child("CNMF-E Parameters", "patch_size").value()
        stride = self.params.child("CNMF-E Parameters", "stride").value()

        # Calculate correlation and PNR images
        self.log("Calculating correlation and PNR images...")
        images_part = images[:1000]
        correlation_image, peak_to_noise_ratio = cm.summary_images.correlation_pnr(
            images_part, gSig=gSig, swap_dim=False
        )

        # Store correlation image for later use
        self.correlation_image = correlation_image
        self.peak_to_noise_ratio = peak_to_noise_ratio
        self.image_max = np.max(images, axis=0)
        self.image_mean = np.mean(images, axis=0)
        self.image_std = np.std(images, axis=0)

        # Update correlation and PNR plots
        self.update_image_plots()

        # Get framerate for data params
        framerate = self.params.child("Data Parameters", "Frame Rate (Hz)").value()
        decay_time = self.params.child("Data Parameters", "Decay Time").value()

        # Setup CNMF-E parameters (include data params with motion-corrected file)
        cnmfe_params = {
            "data": {
                "fnames": self.selected_files,  # Use motion-corrected file
                "fr": framerate,
                "decay_time": decay_time,
            },
            "patch": {
                "del_duplicates": True,
                "low_rank_background": None,
                "nb_patch": 0,
                "rf": patch_size,
                "stride": stride,
            },
            "preprocess": {"p": p},
            "init": {
                "K": None,
                "center_psf": True,
                "method_init": "corr_pnr",
                "nb": 0,
                "normalize_init": False,
                "gSig": (gSig, gSig),
                "gSiz": (gSiz, gSiz),
                "min_corr": min_corr,
                "min_pnr": min_pnr,
                "ring_size_factor": 1.4,
                "tsub": 2,
                "ssub": 1,
                "ssub_B": 2,
            },
            "temporal": {"p": p},
            "merging": {"merge_thr": merge_thr},
        }

        # Create and fit CNMF-E model
        parameters = cnmf.params.CNMFParams(params_dict=cnmfe_params)
        self.cnmfe_model = cnmf.CNMF(
            n_processes=self.n_processes, dview=self.cluster, params=parameters
        )

        self.log("Fitting CNMF-E model (this may take a while)...")
        self.cnmfe_model.fit(images)
        self.log("CNMF-E fitting completed.")

    def run_cnmf_2p(self, files_mc):
        """Run standard CNMF processing for 2-photon data."""
        Yr, dims, T = cm.load_memmap(files_mc)
        images = Yr.T.reshape((T,) + dims, order="F")

        # Get parameters from CNMF Parameters group
        p = self.params.child("CNMF Parameters", "p").value()
        K = self.params.child("CNMF Parameters", "K").value()
        gSig = self.params.child("CNMF Parameters", "gSig").value()
        gSiz = self.params.child("CNMF Parameters", "gSiz").value()
        nb = self.params.child("CNMF Parameters", "nb").value()
        patch_size = self.params.child("CNMF Parameters", "patch_size").value()
        stride = self.params.child("CNMF Parameters", "stride").value()
        merge_thr = self.params.child("CNMF Parameters", "merge_thr").value()
        method_init = self.params.child("CNMF Parameters", "method_init").value()
        framerate = self.params.child("Data Parameters", "Frame Rate (Hz)").value()
        decay_time = self.params.child("Data Parameters", "Decay Time").value()

        # Compute correlation image for quality assessment
        self.log("Computing correlation image...")
        Cns = local_correlations_movie_offline(
            files_mc,
            remove_baseline=True,
            window=1000,
            stride=1000,
            winSize_baseline=100,
            quantil_min_baseline=10,
            dview=self.cluster,
        )
        Cn = Cns.max(axis=0)
        Cn[np.isnan(Cn)] = 0
        self.correlation_image = Cn

        # Store summary images
        self.image_max = np.max(images, axis=0)
        self.image_mean = np.mean(images, axis=0)
        self.image_std = np.std(images, axis=0)

        # Update correlation image plot
        self.corr_img_item.setImage(Cn, autoLevels=True)
        self.corr_plot.setTitle(f"Correlation Image (gSig={gSig})")

        # Setup CNMF parameters
        dxy = self.params.child("Data Parameters", "dxy").value()
        cnmf_dict = {
            "data": {
                "fnames": self.selected_files,
                "fr": framerate,
                "decay_time": decay_time,
                "dxy": (dxy, dxy),
            },
            "patch": {
                "rf": patch_size,
                "stride": stride,
                "nb_patch": nb,
            },
            "preprocess": {"p": p},
            "init": {
                "K": K,
                "gSig": (gSig, gSig),
                "gSiz": (gSiz, gSiz),
                "nb": nb,
                "method_init": method_init,
                "rolling_sum": True,
                "only_init": True,
            },
            "spatial": {"nb": nb},
            "temporal": {"p": p, "nb": nb},
            "merging": {"merge_thr": merge_thr},
        }

        parameters = cnmf.params.CNMFParams(params_dict=cnmf_dict)
        self.cnmfe_model = cnmf.CNMF(
            n_processes=self.n_processes, dview=self.cluster, params=parameters
        )

        # Initial fit
        self.log("Fitting CNMF model (this may take a while)...")
        self.cnmfe_model.fit(images)
        self.log(
            f"Initial CNMF fit completed. Found {self.cnmfe_model.estimates.A.shape[1]} components."
        )

        # Refit on accepted patches to refine and perform deconvolution
        self.log("Refitting CNMF model...")
        self.cnmfe_model = self.cnmfe_model.refit(images, dview=self.cluster)
        self.log(
            f"CNMF refit completed. {self.cnmfe_model.estimates.A.shape[1]} components."
        )

    def run_quality_control(self, files_mc):
        """Run quality control."""
        min_SNR = self.params.child("Quality Control", "min_SNR").value()
        rval_thr = self.params.child("Quality Control", "rval_thr").value()
        use_cnn = self.params.child("Quality Control", "use_cnn").value()

        quality_params = {
            "quality": {
                "min_SNR": min_SNR,
                "rval_thr": rval_thr,
                "use_cnn": use_cnn,
            }
        }
        self.cnmfe_model.params.change_params(params_dict=quality_params)

        Yr, dims, T = cm.load_memmap(files_mc)
        images = Yr.T.reshape((T,) + dims, order="F")

        self.log("Evaluating components...")
        self.cnmfe_model.estimates.evaluate_components(
            images, self.cnmfe_model.params, dview=self.cluster
        )

        if self.cnmfe_model.estimates.F_dff is None:
            self.log("Calculating F_dff...")
            quantileMin = self.params.child("Detrend Parameters", "quantileMin").value()
            frames_window = self.params.child(
                "Detrend Parameters", "frames_window"
            ).value()
            use_residuals = self.params.child(
                "Detrend Parameters", "use_residuals"
            ).value()

            self.cnmfe_model.estimates.detrend_df_f(
                quantileMin=quantileMin,
                frames_window=frames_window,
                use_residuals=use_residuals,
            )

        n_components = len(self.cnmfe_model.estimates.idx_components)
        self.log(f"Quality control completed. {n_components} components passed.")

    def setup_logger_save_folder(self, save_folder):
        """Setup logger to write to save folder location."""
        if save_folder is None:
            return

        save_path = Path(save_folder)
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        # Create logger for save folder
        save_log = logging.getLogger("caiman_runner_save")
        save_log.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        save_log.handlers = []

        # Create log file in save folder
        current_datetime = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
        log_filename = "caiman_runner" + current_datetime + ".log"
        log_path = save_path / log_filename

        # Add file handler
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        save_log.addHandler(handler)

        self.save_logger = save_log
        self.log(f"Logger set up in save folder: {log_path}")

    def save_results(self, files_mc):
        """Save processing results."""
        save_path = self.caiman_path / f"{self.base_name}_data.hdf5"
        self.save_path = save_path

        # Add correlation image if not available
        if (
            not hasattr(self.cnmfe_model, "cn_filter")
            or self.cnmfe_model.cn_filter is None
        ):
            if (
                hasattr(self, "correlation_image")
                and self.correlation_image is not None
                and hasattr(self, "image_max")
                and hasattr(self, "image_mean")
                and hasattr(self, "image_std")
            ):
                self.cnmfe_model.cn_filter = self.correlation_image
                if (
                    hasattr(self, "peak_to_noise_ratio")
                    and self.peak_to_noise_ratio is not None
                ):
                    self.cnmfe_model.pnr = self.peak_to_noise_ratio
                self.cnmfe_model.image_max = self.image_max
                self.cnmfe_model.image_mean = self.image_mean
                self.cnmfe_model.image_std = self.image_std
            else:
                Yr, dims, T = cm.load_memmap(files_mc)
                images = Yr.T.reshape((T,) + dims, order="F")

                if self.is_2p():
                    correlation_image = cm.local_correlations(images, swap_dim=False)
                    correlation_image[np.isnan(correlation_image)] = 0
                    self.cnmfe_model.cn_filter = correlation_image
                else:
                    gSig = self.params.child("CNMF-E Parameters", "gSig").value()
                    correlation_image, peak_to_noise_ratio = (
                        cm.summary_images.correlation_pnr(
                            images, gSig=gSig, swap_dim=False
                        )
                    )
                    self.cnmfe_model.cn_filter = correlation_image
                    self.cnmfe_model.pnr = peak_to_noise_ratio

                self.cnmfe_model.image_max = np.max(images, axis=0)
                self.cnmfe_model.image_mean = np.mean(images, axis=0)
                self.cnmfe_model.image_std = np.std(images, axis=0)

        # Also store on estimates.Cn for notebook compatibility
        self.cnmfe_model.estimates.Cn = self.cnmfe_model.cn_filter

        self.cnmfe_model.save(str(save_path))
        self.log(f"Results saved to: {save_path}")
        self.log(f"Total components: {len(self.cnmfe_model.estimates.idx_components)}")


def caiman_runner_window(parent=None):
    """Open CaImAn runner dialog."""
    win = CaimanRunner(parent)
    win.exec()
