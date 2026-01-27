#!/usr/bin/env python
'''
GUI taking CaImAn processed 1-photon data (hdf5 file) displaying spatial/temporal components
of different groups ('All', 'Accepted', 'Rejected', 'Unassigned') under various mode:
  - 'reset': initialization (show contours of the current group)
  - 'neurons': show colormap of the current group, display mouse clicked multiple components
  - 'correlation': show mouse clicked single componet and all other components
                   whose color is scaled to the pair's temporal correlation
  - 'accepted': display accepted components (keyboard left/right selection)
  - 'neighbors': display neighbors correlation of accepted components (keyboard left/right selection)
  
@author: Hung-Ling
'''
import os
import sys
import json
import logging
import datetime
from pathlib import Path
import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph import FileDialog
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from pyqtgraph.parametertree import Parameter, ParameterTree
from scipy.ndimage import center_of_mass
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi

from .memap_player import memap_window
from .caiman_runner import caiman_runner_window

pg.setConfigOptions(imageAxisOrder='row-major')

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, datapath=None, jsonpath=None):
        super(MainWindow, self).__init__()
        self.resize(1400,1000)
        self.setWindowTitle('Caiman GUI Lite for 1-photon data')
        
        # Icon
        icon_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'Caiman_logo_2.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))
        
        self.statusBar().showMessage('Ready')
        
        ## Discrete colors (cycle of 12) for displaying selected cells
        colors = plt.cm.Set3(np.linspace(0,1,12),bytes=True)[:,:3]  # Shape (12,3) dtype='uint8'
        self.colors = colors.tolist()  # Convert data type to list of int (to circumvent cv2 TypeError) 
        ## Continuous colormap used to setup a colorbar for metric scores
        cmap = plt.cm.jet(np.linspace(0,1,8),bytes=True)[:,:3]
        
        ## ------------ Initialize internal variables -------------------------
        self.loaded = False  # True if the caiman data is loaded
        self.config = None  # Dictionary stores the configuration for NWB file
        self.config_loaded = False  # True if NWB configuration is loaded
        self.mode = 'reset'  # 'reset'|'neurons'|'correlation'|'accepted'|'neighbors'
        self.this_cell = None  # The cell index being selected (by mouse click on img1). None if no cell was selected
        self.selected_cells = []  # List of currently selected cells
        self.neighbor_cells = []  # List of neighbor cells (mode 'neighbors')
        self.last_cell = None  # The cell index of the previously selected cell (used in mode 'accepted'|'neighbors')
        self.yx = np.array([-1,-1])  # Mosue clicked position [y,x] coordinates
        self.logger = None  # Logger instance for save folder
        
        ## ------------ Central Widget Layout --------------------------------
        cw = QtWidgets.QWidget()  # Create a central widget to hold everything
        self.layout = QtWidgets.QGridLayout()  # Create and store the grid layout
        cw.setLayout(self.layout)
        self.setCentralWidget(cw)  
        
        ## ------------ Add Widgets ------------------------------------------
        self.t1 = ParameterTree(showHeader=False)  # For loading data dispaying parameters
        self.t2 = ParameterTree(showHeader=False)  # For action parameters
        self.p1 = pg.PlotWidget()  # For imaging FOV
        self.p2 = pg.PlotWidget()  # For other image (scatter plot...)
        self.p3 = pg.PlotWidget()  # For fluorescence traces
        ## Neuron list table (Phy-like interface)
        self.neuron_table = QtWidgets.QTableWidget()
        self.neuron_table.setColumnCount(7)
        self.neuron_table.setHorizontalHeaderLabels(['ID', 'Rval', 'SNR', 'CNN', 'Area', 'Quality', 'Status'])
        self.neuron_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.neuron_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)  # Allow multiple selection with Ctrl/Shift
        self.neuron_table.setSortingEnabled(True)
        self.neuron_table.itemSelectionChanged.connect(self.on_table_selection_changed)
        self.neuron_table.itemDoubleClicked.connect(self.on_table_double_clicked)
        ## Add widgets to the layout in their proper positions
        self.layout.addWidget(self.t1, 0, 0)  # top-left
        self.layout.addWidget(self.t2, 1, 0)  # bottom-left
        self.layout.addWidget(self.p1, 0, 1)  # top-middle
        self.layout.addWidget(self.p2, 0, 2)   # top-right
        self.layout.addWidget(self.p3, 1, 1, 1, 2)  # bottom-left, spanning 2 columns
        self.layout.addWidget(self.neuron_table, 0, 3, 2, 1)  # right column, spanning 2 rows
        ## Set widget size
        self.t1.setMinimumHeight(320)
        self.t1.setMinimumWidth(270)
        self.t1.setMaximumWidth(360)
        self.t2.setMinimumHeight(320)
        self.t2.setMinimumWidth(270)
        self.t2.setMaximumWidth(360)
        self.neuron_table.setMinimumWidth(300)
        self.neuron_table.setMaximumWidth(400)
        
        ## ------------ Create plot area (ViewBox + axes) --------------------
        self.img1 = pg.ImageItem()
        self.p1.addItem(self.img1)
        self.p1.setAspectLocked()
        self.p1.invertY()  # Set Y axis pointing downward
        self.colorbar = pg.GradientLegend((10,300),(10,10))  # size, offset
        self.colorbar.setParentItem(self.img1)
        gradient = QtGui.QLinearGradient()  # Customize colorbar (jet)
        for i in range(len(cmap)):
            pos = np.linspace(0,1,len(cmap))[i]
            gradient.setColorAt(pos, QtGui.QColor(*cmap[i]))
        self.colorbar.setGradient(gradient)
        self.colorbar.setLabels({'Min':0, 'Max':1})
        self.scatter = pg.ScatterPlotItem(size=12)  # Spot size
        self.scatter.sigClicked.connect(self.scatter_clicked)
        self.p2.addItem(self.scatter)
        self.p2.setLabels(bottom='Rval', left='SNR')
        self.p3.setTitle('Mode: %s' %self.mode)
        self.p3.setLabel('bottom', 'Time (s)')
        self.p3.setMouseEnabled(x=True, y=False)  # Enable only horizontal zoom for displaying traces
        
        ## ------------ Setup munu and parameter tree -------------------------
        self.make_menu()
        self.make_parameter_tree()
        ## ------------ Load data --------------------------------------------
        if datapath is not None:
            self.fname = datapath
            self.load_data(click=False)
        if jsonpath is not None:
            self.fname_json = jsonpath
            self.load_json(click=False)
        ## ------------ Link mouse event -------------------------------------
        self.p1.mousePressEvent = self.mouse_clicked
        ##  A general rule in Qt is that if you override one mouse event handler, you must override all of them ??
        # self.p1.mouseReleaseEvent = lambda *args: None  # "do nothing" function
        # self.p1.mouseMoveEvent = lambda *args: None
        self.p1.scene().sigMouseClicked.connect(self.mouse_clicked)
    
    def make_menu(self):
        ## Run CaImAn processing
        runCaiman = QtGui.QAction('Run CaImAn...', self)
        runCaiman.setShortcut('Ctrl+R')
        runCaiman.setStatusTip('Run CaImAn processing pipeline on raw image files...')
        runCaiman.triggered.connect(lambda: caiman_runner_window(self))
        ## Load caiman hdf5 data
        openHDF5 = QtGui.QAction('Open...', self)
        openHDF5.setShortcut('Ctrl+O')
        openHDF5.setStatusTip('Open caiman hdf5 file')
        openHDF5.triggered.connect(lambda: self.load_data(click=True))
        ## Load json for nwb configuration
        loadJSON = QtGui.QAction('Load json...', self)
        loadJSON.setShortcut('Ctrl+L')
        loadJSON.setStatusTip('Load JSON for NWB file configuration')
        loadJSON.triggered.connect(lambda: self.load_json(click=True))
        ## Save (overwrite original hdf5 file)
        saveData = QtGui.QAction('Save', self)
        saveData.setShortcut('Ctrl+S')
        saveData.setStatusTip('Save caiman hdf5 file')
        saveData.triggered.connect(lambda: self.save_data(new=False))
        ## Save as new file (hdf5 or nwb)
        saveAs = QtGui.QAction('Save as...', self)
        saveAs.setShortcut('Ctrl+Shift+S')
        saveAs.setStatusTip('Save current file as...')
        saveAs.triggered.connect(lambda: self.save_data(new=True))
        ## View memap window
        viewMemap = QtGui.QAction('Movie', self)
        viewMemap.setShortcut('Ctrl+M')
        viewMemap.setStatusTip('Display movie from mmap file...')
        viewMemap.triggered.connect(lambda: memap_window(self))
        ## Exit application
        exitAction = QtGui.QAction('Exit', self)
        exitAction.setShortcut('Ctrl+W')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)
        ## Make main menu
        menu = self.menuBar()
        file_menu = menu.addMenu('&File')
        file_menu.addAction(runCaiman)
        file_menu.addSeparator()
        file_menu.addAction(openHDF5)
        file_menu.addAction(loadJSON)
        file_menu.addSeparator()
        file_menu.addAction(saveData)
        file_menu.addAction(saveAs)
        file_menu.addSeparator()
        file_menu.addAction(exitAction)
        view_menu = menu.addMenu('&View')
        view_menu.addAction(viewMemap)
        
    def make_parameter_tree(self):
        # Setup parameter tree t1 (only if not already set up)
        if not hasattr(self, 'par1') or self.par1 is None:
            param1 = [
                {'name':'NWB config', 'type':'group','children':[
                    {'name':'Sess desc', 'type':'str'},
                    {'name':'Sess start t', 'type':'str'},
                    {'name':'Experimenter', 'type':'str'},
                    {'name':'Exp desc', 'type':'str'}]},
                {'name':'RESET', 'type':'action'},
                {'name':'NEURONS', 'type':'action'},
                {'name':'CORRELATION', 'type':'action'},
                {'name':'Image', 'type':'list', 'values':['Corr','PNR','Max','Mean','Std'], 'value':'PNR'},
                {'name':'Metric', 'type':'list', 'values':['Rval','SNR','Mean paircorr','Max paircorr'], 'value':'Rval'},
                {'name':'Trace', 'type':'list', 'values':['Raw','Denoised','Spike'], 'value':'Raw'},
                {'name':'ACCEPTED', 'type':'action'},
                {'name':'NEIGHBORS', 'type':'action'},
                {'name':'Contour thr', 'type':'float', 'value':0.2, 'limits':(0,1), 'step':0.01},
                {'name':'Contour pix', 'type':'int', 'value':1, 'limits':(1,6), 'step':1},
                {'name':'Dist pix', 'type':'int', 'value':100, 'limits':(0,1000), 'step':5},
                {'name':'Cell ID', 'type':'int', 'value':-1, 'limits':(-1,1000), 'step':1}
            ]
            self.par1 = Parameter.create(name='Parameters Display', type='group', children=param1)
            # Clear tree first to prevent duplicates
            self.t1.clear()
            self.t1.setParameters(self.par1, showTop=True)
            self.par1.child('NWB config').sigTreeStateChanged.connect(self.change_config)
            self.par1.param('RESET').sigActivated.connect(self.reset_button)
            self.par1.param('NEURONS').sigActivated.connect(self.neurons_button)
            self.par1.param('CORRELATION').sigActivated.connect(self.correlation_button)
            self.par1.param('Image').sigValueChanged.connect(self.change_image)
            self.par1.param('Metric').sigValueChanged.connect(self.change_metric)
            self.par1.param('Trace').sigValueChanged.connect(self.draw_trace)
            self.par1.param('ACCEPTED').sigActivated.connect(self.accepted_button)
            self.par1.param('NEIGHBORS').sigActivated.connect(self.neighbors_button)
            self.par1.param('Contour thr').sigValueChanged.connect(self.draw_fov_overall)
            self.par1.param('Contour pix').sigValueChanged.connect(self.draw_fov_overall)
            self.par1.param('Dist pix').sigValueChanged.connect(self.draw_fov_overall)
            self.par1.param('Cell ID').sigValueChanged.connect(self.change_cell)
        
        # Setup parameter tree t2 (only if not already set up)
        if not hasattr(self, 'par2') or self.par2 is None:
            param2 = [
                {'name':'View components', 'type':'list', 'values':['All','Accepted','Rejected','Uncertain','Unassigned'], 'value':'All'},
                {'name':'Filter components', 'type':'bool', 'value':True, 'tip':'Filter components'},          
                {'name':'Quality thr','type':'group','children':[
                    {'name':'Rval high', 'type':'float', 'value':0.85, 'limits':(-1,1), 'step':0.01},
                    {'name':'Rval low', 'type':'float', 'value':0.0, 'limits':(-1,1), 'step':0.01},
                    {'name':'SNR high', 'type':'float', 'value':2, 'limits':(0,20), 'step':0.1},
                    {'name':'SNR low', 'type':'float', 'value':0, 'limits':(0,20), 'step':0.1},
                    {'name':'CNN high', 'type':'float', 'value':0.90, 'limits':(0,1), 'step':0.01},
                    {'name':'CNN low', 'type':'float', 'value':0.2, 'limits':(0,1), 'step':0.01},
                    {'name':'Area low', 'type':'int', 'value':25, 'limits':(0,10000), 'step':1}]},
                {'name':'ADD GROUP', 'type':'action'},
                {'name':'REMOVE GROUP', 'type':'action'},
                {'name':'MERGE', 'type':'action'},
                {'name':'ADD SELECTED', 'type':'action'},
                {'name':'REMOVE SELECTED', 'type':'action'},
                {'name':'Info', 'type':'text'}
            ]
            self.par2 = Parameter.create(name='Parameters Action', type='group', children=param2)
            # Clear tree first to prevent duplicates
            self.t2.clear()
            self.t2.setParameters(self.par2, showTop=True)
            self.par2.param('View components').sigValueChanged.connect(self.change_list)
            self.par2.param('Filter components').sigValueChanged.connect(self.change_list)
            self.par2.child('Quality thr').sigTreeStateChanged.connect(self.change_list)
            self.par2.param('MERGE').sigActivated.connect(self.merge_components)
            self.par2.param('ADD GROUP').sigActivated.connect(self.add_group)
            self.par2.param('REMOVE GROUP').sigActivated.connect(self.remove_group)
            self.par2.param('ADD SELECTED').sigActivated.connect(self.add_selected)
            self.par2.param('REMOVE SELECTED').sigActivated.connect(self.remove_selected)
        
    def load_data(self, click=True, filepath=None):
        self.loaded = False
        if filepath is not None:
            # Load from provided file path
            self.fname = filepath
        elif click:
            fname = FileDialog().getOpenFileName(
                caption='Load CNMF Object', filter='HDF5 (*.h5 *.hdf5)')[0]  # ;;NWB (*.nwb)
            self.fname = fname
        
        if not hasattr(self, 'fname') or self.fname is None or self.fname == '':
            self.statusBar().showMessage('No file specified for loading')
            return
            
        try:
            self.cnmf = load_CNMF(self.fname)
            self.loaded = True
            self.statusBar().showMessage('Loading '+self.fname)
        except Exception:
            self.statusBar().showMessage('Loading '+self.fname+' failed. Try other file...')
        
        if self.loaded:
            dims = self.cnmf.estimates.dims  # (y,x)
            A = self.cnmf.estimates.A  # csc_matrix shape (N,K) where N=xy
            img_components = [A[:,i].reshape(dims,order='F').toarray() for i in range(A.shape[1])]
            self.cms = np.array([center_of_mass(comp) for comp in img_components])  # Shape (K,2) centroid (y,x) of each component
            self.img_components = np.stack(
                [normalize_image(comp) for comp in img_components], axis=0)
            ## Store a background image (initially PNR image)
            self.image = normalize_image(self.cnmf.pnr, stretch_prct=True, rgb=True)
            
            ## Compute pairwise correlation and store the metrics (initially r_values)
            # corr_matrix = np.corrcoef(self.cnmf.estimates.C)  # Denoised traces
            corr_matrix = np.corrcoef(self.cnmf.estimates.C + self.cnmf.estimates.YrA)
            np.fill_diagonal(corr_matrix, np.NaN)
            self.corr_matrix = corr_matrix
            rval = self.cnmf.estimates.r_values
            self.colorbar.setLabels({f'{rval.min():.2f}':0, f'{rval.max():.2f}':1})
            self.metric = (rval-rval.min())/(rval.max()-rval.min())
            
            ## Build accepted, rejected, and uncertain list
            accepted_empty = True
            if hasattr(self.cnmf.estimates, 'accepted_list'):
                accepted_empty = (len(self.cnmf.estimates.accepted_list)==0)  # False if accepted_list already contains neuron indices 
            if accepted_empty:
                self.cnmf.estimates.accepted_list = np.array([], dtype=int)
                self.cnmf.estimates.rejected_list = np.array([], dtype=int)
            if not hasattr(self.cnmf.estimates, 'uncertain_list'):
                self.cnmf.estimates.uncertain_list = np.array([], dtype=int)
            
            # Setup logger at save folder location
            if hasattr(self, 'fname') and self.fname:
                save_folder = os.path.dirname(self.fname)
                self.setup_logger(save_folder)
                if self.logger:
                    self.logger.info(f'Data loaded from: {self.fname}')
            
            self.par2.param('View components').setValue('All', blockSignal=self.change_list)
            self.reset_button()
            self.print_info()
            # Populate neuron table
            self.populate_neuron_table()
            
            ## Set realistic limits
            max_dist = np.sqrt(np.sum(np.array(dims)**2))
            self.par1.param('Dist pix').setLimits((0,int(max_dist)))
            self.par1.param('Cell ID').setLimits((-1,A.shape[1]-1))
            ## Display quality threshold (this may trigger change_list)
            quality = self.cnmf.params.quality
            self.par2.child('Quality thr').param('Rval high').setValue(quality['rval_thr'])
            self.par2.child('Quality thr').param('Rval low').setValue(quality['rval_lowest'])
            self.par2.child('Quality thr').param('SNR high').setValue(quality['min_SNR'])
            self.par2.child('Quality thr').param('SNR low').setValue(quality['SNR_lowest'])
            self.par2.child('Quality thr').param('CNN high').setValue(quality['min_cnn_thr'])
            self.par2.child('Quality thr').param('CNN low').setValue(quality['cnn_lowest'])
            self.par2.param('Filter components').setValue(True)
            self.statusBar().showMessage('Loaded: '+self.fname)
            
    def load_json(self, click=True):
        self.config_loaded = False
        if click:
            fname_json = FileDialog().getOpenFileName(
                caption='Load json file for NWB configuration', filter='JSON (*.json)')[0]
            self.fname_json = fname_json
        try:   
            with open(self.fname_json, 'r') as f:
                self.config = json.load(f)
            ## Set initial NWB metadata using the json file    
            self.par1.child('NWB config').param('Sess desc').setValue(
                self.config['nwbfile']['session_description'])
            self.par1.child('NWB config').param('Sess start t').setValue(
                self.config['nwbfile']['session_start_time'])
            self.par1.child('NWB config').param('Experimenter').setValue(
                self.config['nwbfile']['experimenter'])
            self.par1.child('NWB config').param('Exp desc').setValue(
                self.config['nwbfile']['experiment_description'])
            self.config_loaded = True
        except Exception:
            self.statusBar().showMessage('Loading '+self.fname_json+' failed. Try other file...')
        
    def change_config(self):
        '''Update self.config dictionary from user input
        '''
        if self.config_loaded:
            self.config['nwbfile']['session_description'] = \
                self.par1.child('NWB config').param('Sess desc').value()
            self.config['nwbfile']['session_start_time'] = \
                self.par1.child('NWB config').param('Sess start t').value()
            self.config['nwbfile']['experimenter'] = \
                self.par1.child('NWB config').param('Experimenter').value()
            self.config['nwbfile']['experiment_description'] = \
                self.par1.child('NWB config').param('Exp desc').value()

    # %% Various setup
    def reset_init(self, accepted=False):
        self.p1.setTitle('FOV')
        self.p2.setTitle('Metrics')
        self.p3.clearPlots()
        self.p3.setTitle('Mode: %s' %self.mode)
        if accepted:
            accepted_list = self.cnmf.estimates.accepted_list
            if len(accepted_list) > 0:
                self.this_cell = accepted_list[0]
                self.selected_cells = [self.this_cell]
                self.neighbor_cells = []
                self.last_cell = accepted_list[0]
                self.par1.param('Cell ID').setValue(self.this_cell)
        else:
            self.this_cell = None
            self.selected_cells = []
            self.neighbor_cells = []
            self.last_cell = None
            self.par1.param('Cell ID').setValue(-1, blockSignal=self.change_cell)
            
    def reset_button(self):
        self.mode = 'reset'
        self.reset_init()
        self.draw_contours()
        self.p1.autoRange()
        self.draw_scatter()
        self.p2.autoRange()
        
    def neurons_button(self):
        self.mode = 'neurons'
        self.reset_init()
        self.draw_colormap()
        self.draw_scatter()
        
    def correlation_button(self):
        self.mode = 'correlation'
        self.reset_init()
        self.draw_contours()
        self.draw_scatter()
    
    def accepted_button(self):
        self.mode = 'accepted'
        self.reset_init(accepted=True)
        self.par2.param('Filter components').setValue(False)
        self.par2.param('View components').setValue('Accepted')
        self.draw_fov_overall()
        self.draw_scatter()
        self.draw_trace()
    
    def neighbors_button(self):
        self.mode = 'neighbors'
        self.reset_init(accepted=True)
        self.par2.param('Filter components').setValue(False)
        self.par2.param('View components').setValue('Accepted')
        accepted_list = self.cnmf.estimates.accepted_list
        if len(accepted_list) > 0:
            radius = self.par1.param('Dist pix').value() 
            distances = np.sqrt(np.sum(
                (self.cms[self.this_cell] - self.cms[accepted_list])**2, axis=1))
            self.neighbor_cells = np.setdiff1d(
                accepted_list[distances<radius], self.this_cell)
        self.draw_fov_overall()
        self.draw_scatter()
        self.draw_trace()
        
    def print_info(self):
        K = self.cnmf.estimates.C.shape[0]
        idx_components = self.cnmf.estimates.idx_components
        accepted_list = self.cnmf.estimates.accepted_list
        rejected_list = self.cnmf.estimates.rejected_list
        uncertain_list = getattr(self.cnmf.estimates, 'uncertain_list', np.array([], dtype=int))
        self.par2.param('Info').setValue(
            os.path.split(self.fname)[-1] + '\n' 
            + '-'*32 + '\n' 
            + f'Total components: {K}\n'
            + f'Current group: {len(idx_components)}\n'
            + f'Accepted: {len(accepted_list)}\n'
            + f'Rejected: {len(rejected_list)}\n'
            + f'Uncertain: {len(uncertain_list)}\n')  # + str([*accepted_list])
    
    def populate_neuron_table(self):
        """Populate the neuron table with all neurons and their parameters."""
        if not self.loaded:
            return
        
        K = self.cnmf.estimates.C.shape[0]
        rval = self.cnmf.estimates.r_values
        snr = self.cnmf.estimates.SNR_comp
        # Get CNN predictions if available
        cnn_preds = getattr(self.cnmf.estimates, 'cnn_preds', None)
        has_cnn = cnn_preds is not None and len(cnn_preds) > 0
        accepted_list = self.cnmf.estimates.accepted_list
        rejected_list = self.cnmf.estimates.rejected_list
        uncertain_list = getattr(self.cnmf.estimates, 'uncertain_list', np.array([], dtype=int))
        
        # Calculate contour area (number of non-zero pixels in spatial footprint)
        A = self.cnmf.estimates.A  # Sparse matrix shape (N, K)
        self.cnmf.estimates.areas = np.array([A[:, i].nnz for i in range(K)])  # Number of non-zero elements per component
        
        # Disable sorting temporarily to avoid issues during population
        self.neuron_table.setSortingEnabled(False)
        self.neuron_table.setRowCount(K)
        
        for i in range(K):
            # ID - store as numeric type in EditRole for proper sorting
            id_item = QtWidgets.QTableWidgetItem()
            id_item.setData(QtCore.Qt.EditRole, i)  # Store as int - QVariant remembers type
            id_item.setData(QtCore.Qt.DisplayRole, str(i))  # Display as string
            id_item.setData(QtCore.Qt.UserRole, i)  # Store neuron ID
            id_item.setFlags(id_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.neuron_table.setItem(i, 0, id_item)
            
            # Rval - store as numeric type in EditRole for proper sorting
            rval_item = QtWidgets.QTableWidgetItem()
            rval_item.setData(QtCore.Qt.EditRole, float(rval[i]))  # Store as float - QVariant remembers type
            rval_item.setData(QtCore.Qt.DisplayRole, f'{rval[i]:.3f}')  # Display as formatted string
            rval_item.setData(QtCore.Qt.UserRole, rval[i])
            rval_item.setFlags(rval_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.neuron_table.setItem(i, 1, rval_item)
            
            # SNR - store as numeric type in EditRole for proper sorting
            snr_item = QtWidgets.QTableWidgetItem()
            snr_item.setData(QtCore.Qt.EditRole, float(snr[i]))  # Store as float - QVariant remembers type
            snr_item.setData(QtCore.Qt.DisplayRole, f'{snr[i]:.2f}')  # Display as formatted string
            snr_item.setData(QtCore.Qt.UserRole, snr[i])
            snr_item.setFlags(snr_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.neuron_table.setItem(i, 2, snr_item)
            
            # CNN - store as numeric type in EditRole for proper sorting
            cnn_item = QtWidgets.QTableWidgetItem()
            if has_cnn and i < len(cnn_preds):
                cnn_value = float(cnn_preds[i])
                cnn_item.setData(QtCore.Qt.EditRole, cnn_value)  # Store as float - QVariant remembers type
                cnn_item.setData(QtCore.Qt.DisplayRole, f'{cnn_value:.3f}')  # Display as formatted string
                cnn_item.setData(QtCore.Qt.UserRole, cnn_value)
            else:
                cnn_item.setData(QtCore.Qt.EditRole, -1.0)  # Store as float - QVariant remembers type (N/A comes last)
                cnn_item.setData(QtCore.Qt.DisplayRole, 'N/A')  # Display as string
                cnn_item.setData(QtCore.Qt.UserRole, np.nan)
            cnn_item.setFlags(cnn_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.neuron_table.setItem(i, 3, cnn_item)
            
            # Area (contour area size) - store as numeric type in EditRole for proper sorting
            area_value = int(self.cnmf.estimates.areas[i])
            area_item = QtWidgets.QTableWidgetItem()
            area_item.setData(QtCore.Qt.EditRole, area_value)  # Store as int - QVariant remembers type
            area_item.setData(QtCore.Qt.DisplayRole, str(area_value))  # Display as string
            area_item.setData(QtCore.Qt.UserRole, area_value)
            area_item.setFlags(area_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.neuron_table.setItem(i, 4, area_item)
            
            # Quality (based on idx_components or idx_components_bad)
            idx_components = getattr(self.cnmf.estimates, 'idx_components', None)
            idx_components_bad = getattr(self.cnmf.estimates, 'idx_components_bad', None)
            if idx_components is not None and i in idx_components:
                quality = 'Good'
                quality_color = QtGui.QColor(200, 255, 200)  # Light green
            elif idx_components_bad is not None and i in idx_components_bad:
                quality = 'Bad'
                quality_color = QtGui.QColor(255, 200, 200)  # Light red
            else:
                quality = 'Unknown'
                quality_color = QtGui.QColor(240, 240, 240)  # Light gray
            
            quality_item = QtWidgets.QTableWidgetItem(quality)
            quality_item.setData(QtCore.Qt.UserRole, i)
            quality_item.setFlags(quality_item.flags() & ~QtCore.Qt.ItemIsEditable)
            quality_item.setBackground(quality_color)
            quality_item.setForeground(QtGui.QColor(0, 0, 0))  # Black text color
            self.neuron_table.setItem(i, 5, quality_item)
            
            # Status
            if i in accepted_list:
                status = 'Good'
                color = QtGui.QColor(200, 255, 200)  # Light green
            elif i in rejected_list:
                status = 'Noise'
                color = QtGui.QColor(255, 200, 200)  # Light red
            elif i in uncertain_list:
                status = 'Uncertain'
                color = QtGui.QColor(255, 255, 200)  # Light yellow
            else:
                status = 'Unassigned'
                color = QtGui.QColor(240, 240, 240)  # Light gray
            
            status_item = QtWidgets.QTableWidgetItem(status)
            status_item.setData(QtCore.Qt.UserRole, i)
            status_item.setFlags(status_item.flags() & ~QtCore.Qt.ItemIsEditable)
            status_item.setBackground(color)
            status_item.setForeground(QtGui.QColor(0, 0, 0))  # Black text color
            self.neuron_table.setItem(i, 6, status_item)
        
        # Resize columns to content
        self.neuron_table.resizeColumnsToContents()
        # Re-enable sorting
        self.neuron_table.setSortingEnabled(True)
        
        # Apply current view mode filter
        if hasattr(self, 'par2') and self.par2 is not None:
            select_mode = self.par2.param('View components').value()
            accepted_list = self.cnmf.estimates.accepted_list
            rejected_list = self.cnmf.estimates.rejected_list
            uncertain_list = getattr(self.cnmf.estimates, 'uncertain_list', np.array([], dtype=int))
            self.filter_neuron_table(select_mode, accepted_list, rejected_list, uncertain_list)
    
    def on_table_selection_changed(self):
        """Handle table selection change - sync with current neuron selection."""
        if not self.loaded:
            return
        
        # Get neuron IDs directly from selected items' UserRole data
        # This works correctly even when table is sorted/reordered
        selected_neuron_ids = []
        selected_items = self.neuron_table.selectedItems()
        
        # Get unique neuron IDs from selected items (check all columns to get unique rows)
        seen_rows = set()
        for item in selected_items:
            row = item.row()
            if row not in seen_rows:
                seen_rows.add(row)
                # Get neuron ID from the ID column (column 0) of this row
                # The UserRole data contains the actual neuron ID, not affected by sorting
                id_item = self.neuron_table.item(row, 0)
                if id_item:
                    neuron_id = id_item.data(QtCore.Qt.UserRole)
                    if neuron_id is not None:
                        selected_neuron_ids.append(int(neuron_id))
        
        if selected_neuron_ids:
                # Update selected cells
                self.selected_cells = selected_neuron_ids
                # Set this_cell to the first selected (or last if only one)
                self.this_cell = selected_neuron_ids[0] if len(selected_neuron_ids) == 1 else selected_neuron_ids[-1]
                
                # Update UI
                if len(selected_neuron_ids) == 1:
                    self.par1.param('Cell ID').setValue(self.this_cell, blockSignal=self.change_cell)
                    self.p1.setTitle('Component %d' % self.this_cell)
                    self.p2.setTitle('Rval: %.3f SNR: %.2f'
                                     % (self.cnmf.estimates.r_values[self.this_cell],
                                        self.cnmf.estimates.SNR_comp[self.this_cell]))
                else:
                    self.par1.param('Cell ID').setValue(self.this_cell, blockSignal=self.change_cell)
                    self.p1.setTitle(f'Components: {len(selected_neuron_ids)} selected')
                    avg_rval = np.mean([self.cnmf.estimates.r_values[i] for i in selected_neuron_ids])
                    avg_snr = np.mean([self.cnmf.estimates.SNR_comp[i] for i in selected_neuron_ids])
                    self.p2.setTitle(f'Avg Rval: {avg_rval:.3f} Avg SNR: {avg_snr:.2f}')
                
                # Update displays
                self.draw_fov_overall()
                self.draw_scatter()
                self.draw_trace()
        else:
            # No selection - clear
            self.this_cell = None
            self.selected_cells = []
            self.par1.param('Cell ID').setValue(-1, blockSignal=self.change_cell)
            self.p1.setTitle('')
            self.p2.setTitle('')
            self.draw_fov_overall()
            self.draw_scatter()
            self.draw_trace()
    
    def on_table_double_clicked(self, item):
        """Handle double click on table item - focus on that neuron."""
        row = item.row()
        neuron_id_item = self.neuron_table.item(row, 0)
        if neuron_id_item:
            neuron_id = neuron_id_item.data(QtCore.Qt.UserRole)
            if neuron_id is not None:
                self.this_cell = int(neuron_id)
                self.selected_cells = [self.this_cell]
                self.update_selection(self.this_cell)
    
    def sync_table_selection(self):
        """Sync table selection with current neuron selection."""
        if not self.loaded:
            return
        
        # Block signals to prevent recursive updates
        self.neuron_table.blockSignals(True)
        self.neuron_table.clearSelection()
        
        if self.selected_cells:
            # Find rows with matching neuron IDs
            rows_to_select = []
            for row in range(self.neuron_table.rowCount()):
                id_item = self.neuron_table.item(row, 0)
                if id_item:
                    neuron_id = id_item.data(QtCore.Qt.UserRole)
                    if neuron_id in self.selected_cells:
                        rows_to_select.append(row)
            
            # Select all matching rows
            for row in rows_to_select:
                self.neuron_table.selectRow(row)
            
            # Scroll to first selected item if any
            if rows_to_select and self.this_cell is not None:
                for row in range(self.neuron_table.rowCount()):
                    id_item = self.neuron_table.item(row, 0)
                    if id_item and id_item.data(QtCore.Qt.UserRole) == self.this_cell:
                        self.neuron_table.scrollToItem(id_item)
                        break
        
        self.neuron_table.blockSignals(False)
    
    def find_next_unlabeled_neuron(self, start_from=None):
        """Find the next unlabeled (unassigned) neuron after the current one, based on table row order."""
        if not self.loaded:
            return None
        
        K = self.cnmf.estimates.C.shape[0]
        accepted_list = self.cnmf.estimates.accepted_list
        rejected_list = self.cnmf.estimates.rejected_list
        uncertain_list = getattr(self.cnmf.estimates, 'uncertain_list', np.array([], dtype=int))
        
        # Get all labeled neurons
        all_labeled = np.union1d(np.union1d(accepted_list, rejected_list), uncertain_list)
        # Get unlabeled neurons
        unlabeled_set = set(np.setdiff1d(np.arange(K), all_labeled))
        
        if len(unlabeled_set) == 0:
            return None  # No unlabeled neurons
        
        # Find the current row in the table
        current_row = None
        if start_from is not None:
            # Find the row containing the start_from neuron
            for row in range(self.neuron_table.rowCount()):
                id_item = self.neuron_table.item(row, 0)
                if id_item:
                    neuron_id = id_item.data(QtCore.Qt.UserRole)
                    if neuron_id == start_from:
                        current_row = row
                        break
        elif self.this_cell is not None:
            # Find the row containing the current cell
            for row in range(self.neuron_table.rowCount()):
                id_item = self.neuron_table.item(row, 0)
                if id_item:
                    neuron_id = id_item.data(QtCore.Qt.UserRole)
                    if neuron_id == self.this_cell:
                        current_row = row
                        break
        
        # Search from the next row after current_row (or from start if current_row is None)
        start_row = (current_row + 1) if current_row is not None else 0
        
        # Look for next unlabeled neuron in table rows (respecting current sort order)
        for row in range(start_row, self.neuron_table.rowCount()):
            id_item = self.neuron_table.item(row, 0)
            if id_item:
                neuron_id = id_item.data(QtCore.Qt.UserRole)
                if neuron_id is not None and neuron_id in unlabeled_set:
                    return int(neuron_id)
        
        # If not found, wrap around and search from the beginning
        for row in range(0, start_row):
            id_item = self.neuron_table.item(row, 0)
            if id_item:
                neuron_id = id_item.data(QtCore.Qt.UserRole)
                if neuron_id is not None and neuron_id in unlabeled_set:
                    return int(neuron_id)
        
        return None
    
    def mark_neuron_good(self):
        """Mark selected neuron(s) as good (accepted)."""
        if not self.loaded or not self.selected_cells:
            return
        
        # Store current cell for finding next unlabeled
        current_cell = self.this_cell if self.this_cell is not None else self.selected_cells[0]
        
        self.cnmf.estimates.accepted_list = np.union1d(
            self.cnmf.estimates.accepted_list, self.selected_cells)
        self.cnmf.estimates.rejected_list = np.setdiff1d(
            self.cnmf.estimates.rejected_list, self.selected_cells)
        if hasattr(self.cnmf.estimates, 'uncertain_list'):
            self.cnmf.estimates.uncertain_list = np.setdiff1d(
                self.cnmf.estimates.uncertain_list, self.selected_cells)
        self.update_neuron_table_status()
        self.change_list(None, None)
        if self.logger:
            self.logger.info(f'Marked neurons {self.selected_cells} as Good')
        
        # Move to next unlabeled neuron
        next_unlabeled = self.find_next_unlabeled_neuron(start_from=current_cell)
        if next_unlabeled is not None:
            self.this_cell = next_unlabeled
            self.selected_cells = [next_unlabeled]
            self.par1.param('Cell ID').setValue(next_unlabeled, blockSignal=self.change_cell)
            self.p1.setTitle('Component %d' % next_unlabeled)
            self.p2.setTitle('Rval: %.3f SNR: %.2f'
                             % (self.cnmf.estimates.r_values[next_unlabeled],
                                self.cnmf.estimates.SNR_comp[next_unlabeled]))
            self.sync_table_selection()
            self.draw_fov_overall()
            self.draw_scatter()
            self.draw_trace()
    
    def mark_neuron_noise(self):
        """Mark selected neuron(s) as noise (rejected)."""
        if not self.loaded or not self.selected_cells:
            return
        
        # Store current cell for finding next unlabeled
        current_cell = self.this_cell if self.this_cell is not None else self.selected_cells[0]
        
        self.cnmf.estimates.rejected_list = np.union1d(
            self.cnmf.estimates.rejected_list, self.selected_cells)
        self.cnmf.estimates.accepted_list = np.setdiff1d(
            self.cnmf.estimates.accepted_list, self.selected_cells)
        if hasattr(self.cnmf.estimates, 'uncertain_list'):
            self.cnmf.estimates.uncertain_list = np.setdiff1d(
                self.cnmf.estimates.uncertain_list, self.selected_cells)
        self.update_neuron_table_status()
        self.change_list(None, None)
        if self.logger:
            self.logger.info(f'Marked neurons {self.selected_cells} as Noise')
        
        # Move to next unlabeled neuron
        next_unlabeled = self.find_next_unlabeled_neuron(start_from=current_cell)
        if next_unlabeled is not None:
            self.this_cell = next_unlabeled
            self.selected_cells = [next_unlabeled]
            self.par1.param('Cell ID').setValue(next_unlabeled, blockSignal=self.change_cell)
            self.p1.setTitle('Component %d' % next_unlabeled)
            self.p2.setTitle('Rval: %.3f SNR: %.2f'
                             % (self.cnmf.estimates.r_values[next_unlabeled],
                                self.cnmf.estimates.SNR_comp[next_unlabeled]))
            self.sync_table_selection()
            self.draw_fov_overall()
            self.draw_scatter()
            self.draw_trace()
    
    def mark_neuron_uncertain(self):
        """Mark selected neuron(s) as uncertain."""
        if not self.loaded or not self.selected_cells:
            return
        
        # Store current cell for finding next unlabeled
        current_cell = self.this_cell if self.this_cell is not None else self.selected_cells[0]
        
        if not hasattr(self.cnmf.estimates, 'uncertain_list'):
            self.cnmf.estimates.uncertain_list = np.array([], dtype=int)
        self.cnmf.estimates.uncertain_list = np.union1d(
            self.cnmf.estimates.uncertain_list, self.selected_cells)
        self.cnmf.estimates.accepted_list = np.setdiff1d(
            self.cnmf.estimates.accepted_list, self.selected_cells)
        self.cnmf.estimates.rejected_list = np.setdiff1d(
            self.cnmf.estimates.rejected_list, self.selected_cells)
        self.update_neuron_table_status()
        self.change_list(None, None)
        if self.logger:
            self.logger.info(f'Marked neurons {self.selected_cells} as Uncertain')
        
        # Move to next unlabeled neuron
        next_unlabeled = self.find_next_unlabeled_neuron(start_from=current_cell)
        if next_unlabeled is not None:
            self.this_cell = next_unlabeled
            self.selected_cells = [next_unlabeled]
            self.par1.param('Cell ID').setValue(next_unlabeled, blockSignal=self.change_cell)
            self.p1.setTitle('Component %d' % next_unlabeled)
            self.p2.setTitle('Rval: %.3f SNR: %.2f'
                             % (self.cnmf.estimates.r_values[next_unlabeled],
                                self.cnmf.estimates.SNR_comp[next_unlabeled]))
            self.sync_table_selection()
            self.draw_fov_overall()
            self.draw_scatter()
            self.draw_trace()
    
    def update_neuron_table_status(self):
        """Update status column in neuron table."""
        if not self.loaded:
            return
        
        accepted_list = self.cnmf.estimates.accepted_list
        rejected_list = self.cnmf.estimates.rejected_list
        uncertain_list = getattr(self.cnmf.estimates, 'uncertain_list', np.array([], dtype=int))
        K = self.cnmf.estimates.C.shape[0]
        
        # Disable sorting temporarily
        self.neuron_table.setSortingEnabled(False)
        
        # Iterate through all rows and update status based on neuron ID stored in UserRole
        # This works correctly even when table is sorted
        for row in range(self.neuron_table.rowCount()):
            # Get the neuron ID from the ID column (column 0)
            id_item = self.neuron_table.item(row, 0)
            if id_item:
                neuron_id = id_item.data(QtCore.Qt.UserRole)
                if neuron_id is not None:
                    neuron_id = int(neuron_id)
                    # Get the status item for this row (column 6, after Quality)
                    status_item = self.neuron_table.item(row, 6)
                    if status_item:
                        if neuron_id in accepted_list:
                            status = 'Good'
                            color = QtGui.QColor(200, 255, 200)  # Light green
                        elif neuron_id in rejected_list:
                            status = 'Noise'
                            color = QtGui.QColor(255, 200, 200)  # Light red
                        elif neuron_id in uncertain_list:
                            status = 'Uncertain'
                            color = QtGui.QColor(255, 255, 200)  # Light yellow
                        else:
                            status = 'Unassigned'
                            color = QtGui.QColor(240, 240, 240)  # Light gray
                        
                        status_item.setText(status)
                        status_item.setBackground(color)
                        status_item.setForeground(QtGui.QColor(0, 0, 0))  # Black text color
        
        # Re-enable sorting
        self.neuron_table.setSortingEnabled(True)
        self.print_info()
    
    def update_neuron_table_quality(self):
        """Update quality column in neuron table based on idx_components and idx_components_bad."""
        if not self.loaded:
            return
        
        idx_components = getattr(self.cnmf.estimates, 'idx_components', None)
        idx_components_bad = getattr(self.cnmf.estimates, 'idx_components_bad', None)
        
        if idx_components is None and idx_components_bad is None:
            return  # Quality not yet determined
        
        # Disable sorting temporarily
        self.neuron_table.setSortingEnabled(False)
        
        # Iterate through all rows and update quality based on neuron ID stored in UserRole
        # This works correctly even when table is sorted
        for row in range(self.neuron_table.rowCount()):
            # Get the neuron ID from the ID column (column 0)
            id_item = self.neuron_table.item(row, 0)
            if id_item:
                neuron_id = id_item.data(QtCore.Qt.UserRole)
                if neuron_id is not None:
                    neuron_id = int(neuron_id)
                    # Get the quality item for this row (column 5)
                    quality_item = self.neuron_table.item(row, 5)
                    if quality_item:
                        if idx_components is not None and neuron_id in idx_components:
                            quality = 'Good'
                            quality_color = QtGui.QColor(200, 255, 200)  # Light green
                        elif idx_components_bad is not None and neuron_id in idx_components_bad:
                            quality = 'Bad'
                            quality_color = QtGui.QColor(255, 200, 200)  # Light red
                        else:
                            quality = 'Unknown'
                            quality_color = QtGui.QColor(240, 240, 240)  # Light gray
                        
                        quality_item.setText(quality)
                        quality_item.setBackground(quality_color)
                        quality_item.setForeground(QtGui.QColor(0, 0, 0))  # Black text color
        
        # Re-enable sorting
        self.neuron_table.setSortingEnabled(True)
    
    def filter_neuron_table(self, select_mode, accepted_list, rejected_list, uncertain_list):
        """Filter neuron table rows based on view mode (show/hide rows)."""
        if not self.loaded:
            return
        
        # Disable sorting temporarily to avoid issues during filtering
        self.neuron_table.setSortingEnabled(False)
        
        # Determine which neurons should be visible based on view mode
        K = self.neuron_table.rowCount()
        
        for row in range(K):
            # Get the neuron ID from the ID column (column 0)
            id_item = self.neuron_table.item(row, 0)
            if id_item:
                neuron_id = id_item.data(QtCore.Qt.UserRole)
                if neuron_id is not None:
                    neuron_id = int(neuron_id)
                    
                    # Determine visibility based on view mode
                    if select_mode == 'All':
                        visible = True
                    elif select_mode == 'Accepted':
                        visible = neuron_id in accepted_list
                    elif select_mode == 'Rejected':
                        visible = neuron_id in rejected_list
                    elif select_mode == 'Uncertain':
                        visible = neuron_id in uncertain_list
                    elif select_mode == 'Unassigned':
                        all_labeled = np.union1d(np.union1d(accepted_list, rejected_list), uncertain_list)
                        visible = neuron_id not in all_labeled
                    else:
                        visible = True
                    
                    # Show or hide the row
                    self.neuron_table.setRowHidden(row, not visible)
        
        # Re-enable sorting
        self.neuron_table.setSortingEnabled(True)
            
    def change_image(self):
        img_to_plot = self.par1.param('Image').value()
        if img_to_plot=='PNR':
            self.image = normalize_image(self.cnmf.pnr, stretch_prct=True, rgb=True)
        elif img_to_plot=='Corr':
            self.image = normalize_image(self.cnmf.cn_filter, stretch_prct=True, rgb=True)
        elif hasattr(self.cnmf,'image_max') and img_to_plot=='Max':
            self.image = normalize_image(self.cnmf.image_max, stretch_prct=True, rgb=True)
        elif hasattr(self.cnmf,'image_mean') and img_to_plot=='Mean':
            self.image = normalize_image(self.cnmf.image_mean, stretch_prct=True, rgb=True)
        elif hasattr(self.cnmf,'image_std') and img_to_plot=='Std':
            self.image = normalize_image(self.cnmf.image_std, stretch_prct=True, rgb=True)
        self.draw_fov_overall()
    
    def change_metric(self, plot=True):
        metric = self.par1.param('Metric').value()
        if metric=='Rval':
            scores = self.cnmf.estimates.r_values
        elif metric=='SNR':
            scores = self.cnmf.estimates.SNR_comp
        elif metric=='Mean paircorr':
            scores = np.nanmean(self.corr_matrix, axis=1)
        elif metric=='Max paircorr':
            scores = np.nanmax(self.corr_matrix, axis=1)
        self.colorbar.setLabels({f'{scores.min():.2f}':0, f'{scores.max():.2f}':1})
        self.metric = (scores-scores.min())/(scores.max()-scores.min())  # Normalized to [0,1]
        if plot:
            self.draw_fov_overall()
            self.draw_scatter()
        
    # %% Plotting        
    def draw_contours(self):
        '''Draw contours of all cells in the current group (idx_components) with color scaled to the metric
        '''
        thisImg = self.image.copy()
        thrsh = self.par1.param('Contour thr').value()  # Contour threshold (0,1)
        thick = self.par1.param('Contour pix').value()  # Contour thickness (int pixel)
        idx_components = self.cnmf.estimates.idx_components
        if len(idx_components) > 0:
            colors = plt.cm.jet(self.metric, bytes=True)[:,:3].tolist()  # All colors
            for idx in idx_components:
                img = self.img_components[idx]
                contour = cv2.findContours(cv2.threshold(img,int(255*thrsh),255,0)[1],
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                cv2.drawContours(thisImg, contour, -1, colors[idx], thick)
        self.img1.setImage(thisImg, autoLevels=False)
    
    def draw_colormap(self):
        '''Draw colormap of all cells in the current group (idx_components) with color scaled to the metric
        '''
        thisImg = self.image[:,:,0].copy()  # dtype=np.uint8
        dims = self.cnmf.estimates.dims
        idx_components = self.cnmf.estimates.idx_components
        img_rgb = color_cells(thisImg, self.cnmf.estimates.A, dims,
                              self.metric, idx_components)
        self.img1.setImage(img_rgb, autoLevels=False)
        
    def draw_fov_update(self):
        '''
        Update FOV by drawing contours of the selected cells (mouse clicked)
        and colormap of the other cells in the current group (idx_components).
        Mode 'neurons' or 'correlation'
        '''
        thisImg = self.image[:,:,0].copy()
        idx_components = self.cnmf.estimates.idx_components
        if len(idx_components) > 0:
            dims = self.cnmf.estimates.dims
            idx_remain = np.setdiff1d(idx_components, self.selected_cells)
            if self.mode == 'neurons':
                thisImg = color_cells(thisImg, self.cnmf.estimates.A, dims,
                                      self.metric, idx_remain)
            elif self.mode == 'correlation':
                scores = self.corr_matrix[self.this_cell]
                self.colorbar.setLabels({f'{np.nanmin(scores):.2f}':0, f'{np.nanmax(scores):.2f}':1})
                scores = (scores-np.nanmin(scores))/(np.nanmax(scores)-np.nanmin(scores))
                thisImg = color_cells(thisImg, self.cnmf.estimates.A, dims,
                                      scores, idx_remain)
            thrsh = self.par1.param('Contour thr').value()
            thick = self.par1.param('Contour pix').value()
            for i, idx in enumerate(self.selected_cells):
                ii = i % len(self.colors)  # Remainder
                img = self.img_components[idx]
                contour = cv2.findContours(cv2.threshold(img,int(255*thrsh),255,0)[1],
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                cv2.drawContours(thisImg, contour, -1, self.colors[ii], thick)
        else:
            thisImg = np.dstack([thisImg]*3)
        self.img1.setImage(thisImg, autoLevels=False)

    def draw_fov_keyupdate(self):
        ''' Mode 'accepted' or 'neighbors'
        Draw the contour of this_cell and colormap of other cells in the accepted list.
        '''
        thisImg = self.image[:,:,0].copy()
        accepted_list = self.cnmf.estimates.accepted_list
        if len(accepted_list) > 0:
            dims = self.cnmf.estimates.dims
            thrsh = self.par1.param('Contour thr').value()
            thick = self.par1.param('Contour pix').value()
            if self.mode == 'accepted':
                idx_remain = np.setdiff1d(accepted_list, self.this_cell)        
                thisImg = color_cells(thisImg, self.cnmf.estimates.A, dims,
                                      self.metric, idx_remain)
            elif self.mode == 'neighbors':
                scores = self.corr_matrix[self.this_cell]
                self.colorbar.setLabels({f'{np.nanmin(scores):.2f}':0, f'{np.nanmax(scores):.2f}':1})
                scores = (scores-np.nanmin(scores))/(np.nanmax(scores)-np.nanmin(scores))
                ## Draw colormap of other cells
                thisImg = color_cells(thisImg, self.cnmf.estimates.A, dims,
                                      scores, self.neighbor_cells)
            ## Draw contour of this_cell
            img = self.img_components[self.this_cell]
            contour = cv2.findContours(cv2.threshold(img,int(255*thrsh),255,0)[1],
                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(thisImg, contour, -1, (255,0,255), thick)  # self.colors[3] -> red (plt.cm.Set3)
        else:
            thisImg = np.dstack([thisImg]*3)
        self.img1.setImage(thisImg, autoLevels=False)
    
    def draw_fov_overall(self):
        if self.mode == 'reset':
            self.draw_contours()
        elif self.mode == 'neurons':
            if self.this_cell is None:  # No specific neuron has been selected
                self.draw_colormap()
            else:  # One neuron has been selected by mouse click
                self.draw_fov_update()
        elif self.mode == 'correlation':
            if self.this_cell is None:
                self.draw_contours()
            else:
                self.draw_fov_update()
        elif self.mode in {'accepted', 'neighbors'}:
            if self.this_cell is None:
                self.draw_contours()
            else:
                self.draw_fov_keyupdate()

    def draw_scatter(self):
        '''Scatter plot of neuron quality (Rval & SNR) and color spots with the specified metric
        '''
        self.scatter.clear()
        idx_components = self.cnmf.estimates.idx_components
        rval = self.cnmf.estimates.r_values
        snr = self.cnmf.estimates.SNR_comp
        rgba = plt.cm.jet(self.metric, bytes=True, alpha=0.8)  # Shape (K,4) dtype=np.uint8
        idx_other = np.setdiff1d(np.arange(len(snr)), idx_components)
        rgba[idx_other] = np.array([128,128,128,204])  # Gray for other cells
        spots = []
        for i in range(len(rval)):
            spots.append({'pos':(rval[i], snr[i]), 'data':i,
                          'brush':rgba[i]})
        self.scatter.setData(spots)
        for i in self.selected_cells:
            self.scatter.points()[i].setPen('w', width=2)
            
    def draw_trace(self):
        '''Plot fluorescence traces of selected cells with predefined color cycle (self.colors -> plt.cm.Set3)
        '''
        self.p3.clearPlots()
        if not self.loaded or not self.selected_cells:
            return
        
        fr = self.cnmf.params.data['fr']
        T = self.cnmf.estimates.C.shape[1]
        trace = self.par1.param('Trace').value()
        
        # Handle multiple selected cells - draw traces for all selected
        if len(self.selected_cells) > 0:
            for i, idx in enumerate(self.selected_cells):
                if trace == 'Raw':
                    f = self.cnmf.estimates.C[idx] + self.cnmf.estimates.YrA[idx]
                elif trace == 'Denoised':
                    f = self.cnmf.estimates.C[idx]
                elif trace == 'Spike':
                    f = self.cnmf.estimates.S[idx]
                
                if self.mode in {'neurons', 'correlation'}:
                    if self.mode == 'neurons':
                        ii = i % len(self.colors)  # Remainder (cyclic colors if selected more than 12 cells)
                        self.p3.plot(np.arange(T)/fr, i+f/f.max(), pen=self.colors[ii])  # Normalized trace and shifted vertically 
                    elif self.mode == 'correlation':
                        if len(self.selected_cells) == 1:
                            self.p3.plot(np.arange(T)/fr, f, pen=self.colors[0])  # Original intensity
                        else:
                            # For multiple in correlation mode, show normalized and shifted
                            self.p3.plot(np.arange(T)/fr, i+f/f.max(), pen=self.colors[i % len(self.colors)])
                elif self.mode in {'accepted', 'neighbors'}:
                    if len(self.selected_cells) == 1:
                        # Single selection - original behavior
                        if self.mode == 'accepted':
                            self.p3.plot(np.arange(T)/fr, f, pen='m')
                        elif self.mode == 'neighbors':
                            self.p3.plot(np.arange(T)/fr, f/f.max(), pen='m')
                            scores = self.corr_matrix[self.this_cell]
                            scores = (scores-np.nanmin(scores))/(np.nanmax(scores)-np.nanmin(scores))
                            colors = plt.cm.jet(scores, bytes=True)[:,:3].tolist()
                            if len(self.neighbor_cells) > 0:
                                orders = np.argsort(scores[self.neighbor_cells])[::-1]  # Sort from high to low correlation
                                for j, neighbor_idx in enumerate(self.neighbor_cells[orders]):  # High correlation component is close to the selected cell
                                    if trace == 'Raw':
                                        f_neighbor = self.cnmf.estimates.C[neighbor_idx] + self.cnmf.estimates.YrA[neighbor_idx]
                                    elif trace == 'Denoised':
                                        f_neighbor = self.cnmf.estimates.C[neighbor_idx]
                                    elif trace == 'Spike':
                                        f_neighbor = self.cnmf.estimates.S[neighbor_idx]
                                    self.p3.plot(np.arange(T)/fr, j+1+f_neighbor/f_neighbor.max(), pen=colors[neighbor_idx])
                    else:
                        # Multiple selections - show all traces
                        ii = i % len(self.colors)
                        self.p3.plot(np.arange(T)/fr, i+f/f.max(), pen=self.colors[ii])
                else:
                    # Default mode - show normalized and shifted
                    ii = i % len(self.colors)
                    self.p3.plot(np.arange(T)/fr, i+f/f.max(), pen=self.colors[ii])
                
    # %% Mouse and keyboard interaction
    def update_selection(self, this_cell):
        '''Update this_cell, selected_cells and displaying items after mouse clicked on img1 or scatter plot item.
        '''
        if self.mode == 'neurons':
            if this_cell in self.selected_cells:  # Already selected, remove this_cell
                self.selected_cells.remove(this_cell)  # Click twice removes selected cell
                self.scatter.points()[this_cell].resetPen()
                if len(self.selected_cells) > 0:
                    self.this_cell = self.selected_cells[-1]
                else:
                    self.this_cell = None
            else:  # Not been selected, append this_cell
                self.selected_cells.append(this_cell)  # Multiple cells are selected
                self.scatter.points()[this_cell].setPen('w', width=2)
                self.this_cell = this_cell
        elif self.mode == 'correlation':
            for i in self.selected_cells:  # There should be only one cell or empty in the list
                self.scatter.points()[i].resetPen()
            self.scatter.points()[this_cell].setPen('w', width=2)
            self.selected_cells = [this_cell]  # Override as only one cell selected is possible in correlation mode
            self.this_cell = this_cell
            
        if self.this_cell is None:
            self.p1.setTitle('')
            self.p2.setTitle('')
            self.par1.param('Cell ID').setValue(-1, blockSignal=self.change_cell)
            # No selection - clear table selection too
            self.neuron_table.clearSelection()
        else:
            # Update title and info based on number of selected cells
            if len(self.selected_cells) == 1:
                self.p1.setTitle('Component %d' %self.this_cell)  # *self.yx
                self.p2.setTitle('Rval: %.3f SNR: %.2f'
                                 %(self.cnmf.estimates.r_values[self.this_cell],
                                   self.cnmf.estimates.SNR_comp[self.this_cell]))
            else:
                # Multiple cells selected
                self.p1.setTitle(f'Components: {len(self.selected_cells)} selected')
                avg_rval = np.mean([self.cnmf.estimates.r_values[i] for i in self.selected_cells])
                avg_snr = np.mean([self.cnmf.estimates.SNR_comp[i] for i in self.selected_cells])
                self.p2.setTitle(f'Avg Rval: {avg_rval:.3f} Avg SNR: {avg_snr:.2f}')
            self.par1.param('Cell ID').setValue(self.this_cell,
                                                blockSignal=self.change_cell)
            # Sync table selection - this will select all rows for selected_cells
            self.sync_table_selection()
        self.draw_fov_overall()
        self.draw_scatter()
        self.draw_trace()
            
    def mouse_clicked(self, event):
        '''Determine the mouse clicked position (y,x) on img1 to infer this_cell
        '''
        if self.mode in {'neurons', 'correlation'}:
            dims = self.cnmf.estimates.dims
            # Handle both QMouseEvent (from mousePressEvent) and scene mouse events
            if hasattr(event, 'scenePos'):
                # Event from sigMouseClicked - has scenePos()
                scene_pos = event.scenePos()
            else:
                # Event from mousePressEvent - need to map to scene coordinates
                # Use position() instead of deprecated pos()
                if hasattr(event, 'position'):
                    widget_pos = event.position()
                else:
                    widget_pos = event.pos()  # Fallback for older PyQt versions
                scene_pos = self.p1.mapToScene(widget_pos.toPoint() if hasattr(widget_pos, 'toPoint') else widget_pos)
            pos = self.img1.mapFromScene(scene_pos)
            x, y = pos.x(), pos.y()
            if x>=0 and x<=dims[1] and y>=0 and y<=dims[0]:  # Do nothing if user clicks outside the image
                # i = int(np.clip(y, 0, dims[0] - 1))  # Value outside the interval are clipped to the edges
                # j = int(np.clip(x, 0, dims[1] - 1))
                self.yx = np.array([y, x])
                idx_components = self.cnmf.estimates.idx_components
                # Check if there are any components before calculating distances
                if len(idx_components) == 0:
                    return  # No components to select
                distances = np.sum((self.yx - self.cms[idx_components])**2, axis=1)  # **0.5
                this_cell = idx_components[np.argmin(distances)]  # components_to_plot
                self.update_selection(this_cell)
    
    def scatter_clicked(self, plot, points):
        '''Get the cell ID of the clicked spot on the scatter plot. Argument points is a list of points under the clicked mouse curser.
        '''
        if self.mode in {'neurons', 'correlation'}:
            this_cell = points[0].data()
            self.update_selection(this_cell)
        
    def keyPressEvent(self, event):
        '''Override the existing method to activate left/right key to scroll through the cell ID
        and Alt+G/N/M for marking neurons
        '''
        # Handle ESC key to unselect all neurons
        if event.key() == QtCore.Qt.Key_Escape:
            # Reset scatter plot pens for currently selected cells
            if self.loaded and self.selected_cells:
                for i in self.selected_cells:
                    try:
                        self.scatter.points()[i].resetPen()
                    except (IndexError, AttributeError):
                        pass  # Scatter point might not exist
            self.this_cell = None
            self.selected_cells = []
            self.par1.param('Cell ID').setValue(-1, blockSignal=self.change_cell)
            self.p1.setTitle('')
            self.p2.setTitle('')
            self.neuron_table.clearSelection()
            self.draw_fov_overall()
            self.draw_scatter()
            self.draw_trace()
            return
        
        # Handle Alt+G (Good), Alt+N (Noise), Alt+M (Uncertain) shortcuts
        if event.modifiers() == QtCore.Qt.AltModifier:
            if event.key() == QtCore.Qt.Key_G:
                self.mark_neuron_good()
                return
            elif event.key() == QtCore.Qt.Key_N:
                self.mark_neuron_noise()
                return
            elif event.key() == QtCore.Qt.Key_M:
                self.mark_neuron_uncertain()
                return
        
        if self.mode in {'accepted', 'neighbors'}:
            accepted_list = self.cnmf.estimates.accepted_list
            K2 = len(accepted_list)
            if K2 > 0 and self.this_cell is not None:
                # if event.modifiers() !=  QtCore.Qt.ShiftModifier:
                self.last_cell = self.this_cell
                last_i = np.where(accepted_list==self.last_cell)[0].item()
                if event.key() == QtCore.Qt.Key_Left:
                    this_i = np.clip(last_i-1, 0, K2-1)
                    self.this_cell = accepted_list[this_i]
                elif event.key() == QtCore.Qt.Key_Right:
                    this_i = np.clip(last_i+1, 0, K2-1)
                    self.this_cell = accepted_list[this_i]
                self.selected_cells = [self.this_cell]
                self.par1.param('Cell ID').setValue(
                    self.this_cell, blockSignal=self.change_cell)
                self.p1.setTitle('Component %d' %self.this_cell)
                self.p2.setTitle('Rval: %.3f SNR: %.2f'
                                 %(self.cnmf.estimates.r_values[self.this_cell],
                                   self.cnmf.estimates.SNR_comp[self.this_cell]))
                self.scatter.points()[self.last_cell].resetPen()
                self.scatter.points()[self.this_cell].setPen('w', width=2)
                # Sync table selection
                self.sync_table_selection()
                ## Compute neighbor cells
                if self.mode == 'neighbors':
                    radius = self.par1.param('Dist pix').value() 
                    distances = np.sqrt(np.sum(
                        (self.cms[self.this_cell] - self.cms[accepted_list])**2, axis=1))
                    self.neighbor_cells = np.setdiff1d(
                        accepted_list[distances<radius], self.this_cell)
                self.draw_fov_keyupdate()
                self.draw_trace()
    
    def change_cell(self):
        '''
        Act when the Cell ID parameter is changed by user (update only single cell selection).
        Block this signal when this_cell is determined by mouse clicked and keyboard interaction!!
        '''
        self.this_cell = self.par1.param('Cell ID').value()
        self.p1.setTitle('Component %d' %self.this_cell)
        self.p2.setTitle('Rval: %.3f SNR: %.2f'
                         %(self.cnmf.estimates.r_values[self.this_cell],
                           self.cnmf.estimates.SNR_comp[self.this_cell]))
        for i in self.selected_cells:  # There should be only one cell or empty in the list
            self.scatter.points()[i].resetPen()
        self.scatter.points()[self.this_cell].setPen('w', width=2)
        self.selected_cells = [self.this_cell]
        # Sync table selection
        self.sync_table_selection()
        ## Compute neighbor cells
        if self.mode == 'neighbors':
            accepted_list = self.cnmf.estimates.accepted_list
            radius = self.par1.param('Dist pix').value() 
            distances = np.sqrt(np.sum(
                (self.cms[self.this_cell] - self.cms[accepted_list])**2, axis=1))
            self.neighbor_cells = np.setdiff1d(
                accepted_list[distances<radius], self.this_cell)
        self.draw_fov_overall()
        self.draw_trace()
        
    # %% Merge components
    def merge_components(self):
        '''Merge components in the selected_cells list and update cnmf object.
        '''
        if not self.loaded:
            self.statusBar().showMessage('Please load a data file first')
            return
        if len(self.selected_cells) > 1:
            K = self.cnmf.estimates.C.shape[0]
            K1 = K - len(self.selected_cells) + 1  # Number of total components after merge
            A_merge = self.cnmf.estimates.A[:,self.selected_cells]
            C_merge = self.cnmf.estimates.C[self.selected_cells,:] + \
                self.cnmf.estimates.YrA[self.selected_cells,:]
            computedA, computedC = merge_iteration(A_merge, C_merge)
            deconvC, bl, c1, g, sn, sp, lam = constrained_foopsi(
                computedC, g=None, **self.cnmf.params.get_group('temporal'))
            keep = np.setdiff1d(np.arange(K), self.selected_cells)
            ## Update estimates
            self.cnmf.estimates.A = sparse.hstack([self.cnmf.estimates.A[:,keep], computedA])
            self.cnmf.estimates.C = np.vstack([self.cnmf.estimates.C[keep,:], deconvC])
            self.cnmf.estimates.YrA = np.vstack([self.cnmf.estimates.YrA[keep,:], computedC-deconvC])
            self.cnmf.estimates.S = np.vstack([self.cnmf.estimates.S[keep,:], sp])
            self.cnmf.estimates.bl = np.hstack([self.cnmf.estimates.bl[keep], bl])
            self.cnmf.estimates.c1 = np.hstack([self.cnmf.estimates.c1[keep], c1])
            self.cnmf.estimates.sn = np.hstack([self.cnmf.estimates.sn[keep], sn])
            self.cnmf.estimates.g = np.vstack([self.cnmf.estimates.g[keep], g])  # Shape (K1,p) where p=1 or 2
            self.cnmf.estimates.nr = K1
            ## Update metrics (approximately, not re-evaluated -> need C-memmap)
            rval = np.mean(self.cnmf.estimates.r_values[self.selected_cells])
            snr = np.max(self.cnmf.estimates.SNR_comp[self.selected_cells])
            self.cnmf.estimates.r_values = np.hstack([self.cnmf.estimates.r_values[keep], rval])
            self.cnmf.estimates.SNR_comp = np.hstack([self.cnmf.estimates.SNR_comp[keep], snr])
            ## Update img_components, center of mass, metrics (pairwise correlation...)
            dims = self.cnmf.estimates.dims
            img_merged = computedA.reshape(dims, order='F').toarray()
            self.img_components = np.concatenate(
                [self.img_components[keep,:,:], normalize_image(img_merged)[np.newaxis,:,:]], axis=0)
            self.cms = np.vstack([self.cms[keep,:], np.array(center_of_mass(img_merged))])  # Shape (K1,2)
            self.corr_matrix = np.corrcoef(self.cnmf.estimates.C)
            np.fill_diagonal(self.corr_matrix, np.NaN)
            ## Update accepted list, figures...
            self.cnmf.estimates.accepted_list = update_list(K, self.cnmf.estimates.accepted_list, self.selected_cells)
            self.cnmf.estimates.rejected_list = update_list(K, self.cnmf.estimates.rejected_list, self.selected_cells)
            if hasattr(self.cnmf.estimates, 'uncertain_list'):
                self.cnmf.estimates.uncertain_list = update_list(K, self.cnmf.estimates.uncertain_list, self.selected_cells)
            self.this_cell = K1-1  # The merged component
            self.selected_cells = [K1-1]
            self.change_metric(plot=False)
            self.change_list(None, None)
            # Update table after merge (need to repopulate since neuron count changed)
            if self.loaded:
                self.populate_neuron_table()
            
    # %% Change idx_components, accepted_list and save data
    def add_group(self):
        '''Add all current components to the accepted list
        '''
        if not self.loaded:
            self.statusBar().showMessage('Please load a data file first')
            return
        self.cnmf.estimates.accepted_list = \
            np.union1d(self.cnmf.estimates.accepted_list, self.cnmf.estimates.idx_components)  # union of two arrays
        self.cnmf.estimates.rejected_list = \
            np.setdiff1d(self.cnmf.estimates.rejected_list, self.cnmf.estimates.idx_components)  # unique values in arg1 that are not in arg2
        if hasattr(self.cnmf.estimates, 'uncertain_list'):
            self.cnmf.estimates.uncertain_list = \
                np.setdiff1d(self.cnmf.estimates.uncertain_list, self.cnmf.estimates.idx_components)
        self.update_neuron_table_status()
        self.change_list(None, None)
        
    def remove_group(self):
        '''Remove all current components from the accepted list and put them into the rejected list
        '''
        if not self.loaded:
            self.statusBar().showMessage('Please load a data file first')
            return
        self.cnmf.estimates.rejected_list = \
            np.union1d(self.cnmf.estimates.rejected_list, self.cnmf.estimates.idx_components)
        self.cnmf.estimates.accepted_list = \
            np.setdiff1d(self.cnmf.estimates.accepted_list, self.cnmf.estimates.idx_components)
        if hasattr(self.cnmf.estimates, 'uncertain_list'):
            self.cnmf.estimates.uncertain_list = \
                np.setdiff1d(self.cnmf.estimates.uncertain_list, self.cnmf.estimates.idx_components)
        self.update_neuron_table_status()
        self.change_list(None, None)
        
    def add_selected(self):
        '''Add the current selected component to the accepted list
        '''
        if not self.loaded:
            self.statusBar().showMessage('Please load a data file first')
            return
        self.cnmf.estimates.accepted_list = \
            np.union1d(self.cnmf.estimates.accepted_list, self.selected_cells)
        self.cnmf.estimates.rejected_list = \
            np.setdiff1d(self.cnmf.estimates.rejected_list, self.selected_cells)
        if hasattr(self.cnmf.estimates, 'uncertain_list'):
            self.cnmf.estimates.uncertain_list = \
                np.setdiff1d(self.cnmf.estimates.uncertain_list, self.selected_cells)
        self.update_neuron_table_status()
        self.this_cell = None
        self.selected_cells = []
        self.change_list(None, None)
        
    def remove_selected(self):
        '''Remove the current selected component from the accepted list and put it into the rejected list
        '''
        if not self.loaded:
            self.statusBar().showMessage('Please load a data file first')
            return
        self.cnmf.estimates.rejected_list = \
            np.union1d(self.cnmf.estimates.rejected_list, self.selected_cells)
        self.cnmf.estimates.accepted_list = \
            np.setdiff1d(self.cnmf.estimates.accepted_list, self.selected_cells)
        if hasattr(self.cnmf.estimates, 'uncertain_list'):
            self.cnmf.estimates.uncertain_list = \
                np.setdiff1d(self.cnmf.estimates.uncertain_list, self.selected_cells)
        self.update_neuron_table_status()
        if self.mode in {'neurons', 'correlation'}:
            self.this_cell = None
            self.p1.setTitle('')
            self.p2.setTitle('')
        elif self.mode in {'accepted','neighbors'}:
            if self.this_cell != self.last_cell:
                self.this_cell = self.last_cell  ## Keep this_cell visible as last_cell
            else:
                self.this_cell = self.cnmf.estimates.accepted_list[0]
            self.p1.setTitle('Component %d' %self.this_cell)
            self.p2.setTitle('Rval: %.3f SNR: %.2f'
                             %(self.cnmf.estimates.r_values[self.this_cell],
                               self.cnmf.estimates.SNR_comp[self.this_cell]))
        self.selected_cells = []
        self.change_list(None, None)
        
    def change_list(self, param, changes):  # param, changes are positional (required!!) arguments connected to the Parameter class, not used here
        '''Act when quality thresholds are modified or Filter components, View components state is changed
        '''
        K = self.cnmf.estimates.C.shape[0]
        accepted_list = self.cnmf.estimates.accepted_list
        rejected_list = self.cnmf.estimates.rejected_list
        uncertain_list = getattr(self.cnmf.estimates, 'uncertain_list', np.array([], dtype=int))
        if self.par2.param('Filter components').value():
            set_par = self.par2.child('Quality thr').getValues()
            par_dict = {'rval_thr': set_par['Rval high'][0],
                        'rval_lowest': set_par['Rval low'][0],
                        'min_SNR': set_par['SNR high'][0],
                        'SNR_lowest': set_par['SNR low'][0],
                        'min_cnn_thr': set_par['CNN high'][0],
                        'cnn_lowest': set_par['CNN low'][0]}
            self.cnmf.params.quality.update(par_dict)  # Renew caiman thresholds
            ## ======== Filter Components ============ ##
            ## Rval/SNR/CNN low and high thresholds
            rval = self.cnmf.estimates.r_values
            snr = self.cnmf.estimates.SNR_comp
            if hasattr(self.cnmf.estimates, 'cnn_preds'):
                cnn = self.cnmf.estimates.cnn_preds
            else:
                cnn = np.ones(K)
            areas = self.cnmf.estimates.areas
            low_thr = (rval>=set_par['Rval low'][0]) & (snr>=set_par['SNR low'][0]) & (areas>=set_par['Area low'][0]) & (cnn>=set_par['CNN low'][0])
            high_thr = (rval>=set_par['Rval high'][0]) | (snr>=set_par['SNR high'][0]) | (cnn>=set_par['CNN high'][0])
            good_idx = np.arange(K)[low_thr & high_thr]
        else:  # Not to filter components (Original Caiman GUI: filter with default setting -> this makes thing complicated...)
            good_idx = np.arange(K)
        select_mode = self.par2.param('View components').value()  # 'All'|'Accepted'|'Rejected'|'Uncertain'|'Unassigned'
        if select_mode == 'Accepted':
            good_idx_selected = np.intersect1d(good_idx, accepted_list)
        elif select_mode == 'Rejected':
            good_idx_selected = np.intersect1d(good_idx, rejected_list)
        elif select_mode == 'Uncertain':
            good_idx_selected = np.intersect1d(good_idx, uncertain_list)
        elif select_mode == 'Unassigned':
            good_idx_selected = np.setdiff1d(good_idx,
                                    np.union1d(np.union1d(rejected_list, accepted_list), uncertain_list))
        else:  # 'All'
            good_idx_selected = good_idx

        bad_idx = np.setdiff1d(np.arange(K), good_idx)
        self.cnmf.estimates.idx_components = good_idx
        self.cnmf.estimates.idx_components_bad = bad_idx
        # Update quality column in table
        self.update_neuron_table_quality()
        # Filter table rows based on view mode
        self.filter_neuron_table(select_mode, accepted_list, rejected_list, uncertain_list)
        self.draw_fov_overall()
        self.draw_scatter()
        self.draw_trace()
        self.print_info()
        
    def setup_logger(self, save_folder):
        """Setup logger to write to save folder location."""
        if save_folder is None or save_folder == '':
            return
        
        save_path = Path(save_folder)
        if not save_path.exists():
            return
        
        # Create logger
        logger_name = 'caiman_gui'
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers = []
        
        # Create log file in save folder
        current_datetime = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
        log_filename = 'caiman_gui' + current_datetime + '.log'
        log_path = save_path / log_filename
        
        # Add file handler
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        self.logger = logger
        logger.info(f'Logger initialized. Log file: {log_path}')
    
    def save_data(self, new=True):
        if new:
            fname_save = FileDialog().getSaveFileName(filter='HDF5 (*.hdf5);;NWB (*.nwb)')[0]
        else:
            fname_save = self.fname
        if os.path.splitext(fname_save)[1] == '.hdf5':
            self.cnmf.save(fname_save)
        elif os.path.splitext(fname_save)[1] == '.nwb':
            import nwb
            nwb.save_nwb(self.cnmf, fname_save, self.config, raw_data_file=None)
        
        # Setup logger in save folder
        if fname_save:
            save_folder = os.path.dirname(fname_save)
            self.setup_logger(save_folder)
            if self.logger:
                self.logger.info(f'Data saved to: {fname_save}')
        
        self.statusBar().showMessage('Saved: '+fname_save)
        
# %% Useful functions
def normalize_image(img, stretch_prct=False, prct=(1,99), rgb=False):
    '''
    Normalize image to 'uint8' for use in OpenCV

    Parameters
    ----------
    img : numpy 2D array
        Input image.
    stretch_prct : bool
        Whether to apply a percentile stretching. The default is False (min-max streching).
    prct : tuple of two values between 0 and 100
        The low and high percentile used if stretch_prct. The default is (1,99)
    rgb : bool
        Whether to return a RGB stack. The default is False.

    Returns
    -------
    img2 : numpy 2D or 3D array
        Normalized image, shape (h,w) or (h,w,3).
    '''
    if stretch_prct:
        min_, max_ = np.percentile(img, prct)
        img2 = np.clip((img.copy()-min_)/(max_-min_)*255,0,255).astype('uint8')
    else:  # min/max normalization
        min_, max_ = img.min(), img.max()
        img2 = ((img.copy()-min_)/(max_-min_)*255).astype('uint8')
    if rgb:
        img2 = np.dstack([img2]*3)
    return img2

def color_cells(background, A, dims, scores, list_cells):
    '''
    Overlay a gray scale background image with a list of colored cells. 

    Parameters
    ----------
    background : numpy 2d array, shape (y,x)
    A : scipy.sparse.csc_matrix, shape (N,K) where N=xy and K total number of components
    dims : list or tuple, (y,x) pixels of the FOV
    scores : numpy 1d array, shape (K,)
        Metric used to color cells
    list_cells : list or numpy 1d array
        List of cells to color

    Returns
    -------
    img_rgb : numpy 3d array, shape (y,x,3)
        Background image with colored cells
    '''
    if len(list_cells) > 0:
        # scores = (scores-scores.min())/(scores.max()-scores.min())
        rgb = plt.cm.jet(scores)[:,:3]  # 0 to 1
        hsv = mcolors.rgb_to_hsv(rgb)  # 0 to 1
        hue_list = (hsv[:,0]*179).astype(np.uint8)  # OpenCV hue range is [0,179]
        H = np.zeros(np.prod(dims), dtype=np.uint8)
        S = np.zeros(np.prod(dims), dtype=np.uint8)
        for idx in list_cells:
            img = A[:,idx]  # csc_matrix ('data','indices','indptr') here column vector
            weight = (img.data/np.max(img.data)*255).astype(np.uint8)
            overlay = (weight > S[img.indices])  # Overlay only in region where this component has larger weight than previous
            pos = img.indices[overlay]  # Position (i.e. indices) where to overlay
            H[pos] = hue_list[idx]  # Color the most weighted cell
            S[pos] = weight[overlay]
        H = H.reshape(dims, order='F')
        S = S.reshape(dims, order='F')  # 0 saturation -> white
        img_hsv = np.dstack([H,S,background])  # 0 value (lightness) -> black
        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    else:
        img_rgb = np.dstack([background]*3)
    return img_rgb

def merge_iteration(A_merge, C_merge):
    '''
    Perform rank 1 nonnegative matrix factorization to merge components.
    
    Parameters
    ----------
    A_merge : scipy.sparse.csc_matrix
        Matrix of spatial components to merge, shape (N,r)
        where N=xy the number of pixels, r>=2 the number of components to merge.
    C_merge : numpy.ndarray, shape (r,T)
        Array of temporal components to merge.

    Returns
    -------
    computedA : scipy.csc_matrix, shape (N,1)
        Merged spatial component
    computedC : numpy.ndarray, shape (T,)
        Merged temporal component
    '''
    ## Start by initializing a merged spatial component (computedA) calculated as
    ## a weighted sum of all spatial components to merge (A_merge)
    ## Weights are choosed to scale with the corresponding temporal component C_merge,
    ## and are normalized to 1 (norm2)
    ## Note that norm2 of computedA is not necessarilly 1 due to overlapped A_merge
    C2 = np.mean(C_merge**2, axis=1)  # Shape (r,) sum of squares
    nC = np.sqrt(C2/C2.sum())  # Shape (r,) weight of temporal component such that nC**2 is summed to 1
    computedA = A_merge.dot(nC)  # Initialized A, shape (N,) Note that matrix vector product dot() returns numpy.ndarray !!
    ## Updata merged temporal component (computedC) and computedA with 10 iterations
    for _ in range(10):
        computedC = (A_merge.T.dot(computedA)).dot(C_merge) / (computedA.dot(computedA))
        computedA = A_merge.dot(C_merge.dot(computedC)) / (computedC.dot(computedC))
        computedA = np.maximum(computedA, 0)  # Spatial component is nonnegative
    
    normA = np.sqrt(computedA.dot(computedA))
    computedA /= normA  # Make norm2 of computedA 1
    computedC *= normA
    return sparse.csr_matrix(computedA).T, computedC

def update_list(K, list_idx, merge_idx):
    '''
    Parameters
    ----------
    K : int
        Number of components before merging operation.
    list_idx : numpy 1D array of int
        List of component indices before merging operation.
    merge_idx : numpy 1D array of int
        List of indices of components to merge.

    Returns
    -------
    list_merged : numpy 1D array of int
        The component indices after merging for the same group of components represented by list_idx.
        The merged component appends to the last one and is in the list if all the parent components (merged)
        were in the list
    '''
    keep = np.ones(K, dtype=bool)  # Components to keep
    keep[merge_idx] = False
    list_tmp = np.setdiff1d(list_idx, merge_idx)
    bool_tmp = np.zeros(K, dtype=bool)
    bool_tmp[list_tmp] = True
    bool_tmp = bool_tmp[keep]
    if all([i in list_idx for i in merge_idx]):
        bool_tmp = np.hstack([bool_tmp, True])
    else:
        bool_tmp = np.hstack([bool_tmp, False])
    list_merged = np.where(bool_tmp)[0]
    return list_merged
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
