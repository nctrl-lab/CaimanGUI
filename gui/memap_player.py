"""
Window displaying motion-corrected movie (F-order mmap file).
When opened from the parent GUI with an HDF file loaded, available F-order mmap
files in the same directory are listed.
- Video modes:
    1) Raw video
    2) dF/F 
        - F is calculated from moving median
    3) Background-subtracted video
        - Y - b0 - Bf
        - = Y - b0 - W * (Y - AC - b0))
- Options:
    - Show neuron contours

@author: Hung-Ling
"""
import os
import sys
import numpy as np
import cv2
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from natsort import natsorted
from caiman.mmapping import load_memmap

pg.setConfigOptions(imageAxisOrder='row-major')


def find_mmap_files(directory, order='F'):
    """
    Scan a directory for .mmap files whose filename indicates F-order
    (CaImAn convention: ..._d1_X_d2_Y_d3_Z_order_F_frames_T.mmap).

    Returns
    -------
    list of str
        Full paths of F-order mmap files.
    """
    if not directory or not os.path.isdir(directory):
        return []

    result = []
    for name in os.listdir(directory):
        if not name.lower().endswith('.mmap'):
            continue
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        try:
            filename = os.path.splitext(name)[0]
            fpart = filename.split('_')
            if len(fpart) < 9:
                continue
            if fpart[-3].upper() == order:
                result.append(path)
        except (IndexError, ValueError):
            continue
    return natsorted(result)


class MemapPlayer(QtWidgets.QMainWindow):
    VIDEO_RAW = 'Raw'
    VIDEO_DFF = 'dF/F'
    VIDEO_BGSUB = 'Background subtracted'

    def __init__(self, parent=None):
        super(MemapPlayer, self).__init__(parent)
        self.resize(800, 800)
        self.setWindowTitle('Movie player')
        self.layout = QtWidgets.QGridLayout()
        cw = QtWidgets.QWidget()
        cw.setLayout(self.layout)
        self.setCentralWidget(cw)

        self.loaded = False
        self.cframe = 0
        self.fps = 20.0
        self.prct = [1, 99.9]
        self.dframe = 10
        self.this_cell = -1
        self.Yr = None
        self.dims = None
        self.nframe = 0
        self._dff_baseline = None
        self._video_mode = self.VIDEO_RAW

        # -------- Mmap file selection (dropdown + OPEN) --------
        self.layout.addWidget(QtWidgets.QLabel('Files:'), 0, 0, 1, 1)
        self.mmapCombo = QtWidgets.QComboBox()
        self.mmapCombo.setMinimumWidth(200)
        self.mmapCombo.currentIndexChanged.connect(self.on_mmap_combo_changed)
        self.layout.addWidget(self.mmapCombo, 0, 1, 1, 4)
        openButton = QtWidgets.QPushButton('OPEN...')
        openButton.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        openButton.clicked.connect(self.open_memap)
        self.layout.addWidget(openButton, 0, 5, 1, 1)
        self.fileLabel = QtWidgets.QLabel('No file loaded...')
        self.layout.addWidget(self.fileLabel, 0, 6, 1, 10)

        # Populate mmap combo from parent if available (load first file after all widgets exist)
        self._parent = parent
        if parent is not None and getattr(parent, 'available_mmap_files', None):
            for path in parent.available_mmap_files:
                self.mmapCombo.addItem(os.path.basename(path), path)
            if self.mmapCombo.count() > 0:
                self.mmapCombo.setCurrentIndex(0)

        # -------- Video mode: Raw / dF/F / Background subtracted --------
        self.layout.addWidget(QtWidgets.QLabel('Video:'), 1, 0, 1, 1)
        self.videoModeCombo = QtWidgets.QComboBox()
        self.videoModeCombo.addItems([self.VIDEO_RAW, self.VIDEO_DFF, self.VIDEO_BGSUB])
        self.videoModeCombo.currentTextChanged.connect(self.on_video_mode_changed)
        self.layout.addWidget(self.videoModeCombo, 1, 1, 1, 2)

        # -------- Save as video --------
        self.saveVideoButton = QtWidgets.QPushButton('Save as video...')
        self.saveVideoButton.clicked.connect(self.save_as_video)
        self.saveVideoButton.setEnabled(False)
        self.layout.addWidget(self.saveVideoButton, 1, 3, 1, 2)

        # -------- Show contours --------
        self.showContoursCheck = QtWidgets.QCheckBox('Show contours')
        self.showContoursCheck.stateChanged.connect(self.on_show_contours_changed)
        self.layout.addWidget(self.showContoursCheck, 1, 5, 1, 1)
        self.layout.addWidget(QtWidgets.QLabel('Component:'), 1, 6, 1, 1)
        self.component = QtWidgets.QSpinBox()
        self.component.setRange(-1, 9999)
        self.component.setValue(-1)
        self.component.setSpecialValueText('None')
        self.component.valueChanged.connect(self.change_params)
        self.layout.addWidget(self.component, 1, 7, 1, 1)

        # -------- Frame rate --------
        self.layout.addWidget(QtWidgets.QLabel('Frame rate:'), 1, 8, 1, 1)
        self.rate = QtWidgets.QLineEdit()
        self.rate.setFixedWidth(50)
        self.rate.setAlignment(QtCore.Qt.AlignCenter)
        self.rate.setText(str(self.fps))
        self.rate.textChanged.connect(self.change_params)
        self.layout.addWidget(self.rate, 1, 9, 1, 1)

        # -------- Graphics --------
        graph = pg.GraphicsLayoutWidget()
        self.p1 = graph.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        self.img = pg.ImageItem()
        self.contour = pg.IsocurveItem(level=32, pen='m')
        self.contour.setParentItem(self.img)
        self.contour.setZValue(10)
        self.contour.setVisible(False)
        self.p1.addItem(self.img)
        self.p2 = graph.addPlot(row=1, col=0)
        self.p2.addLegend()
        self.p2.setMouseEnabled(x=True, y=False)
        self.vline = pg.InfiniteLine(angle=90, movable=True)
        self.vline.setValue(0)
        self.vline.sigPositionChanged.connect(self.go_to_frame)
        self.p2.addItem(self.vline, ignoreBounds=True)
        graph.ci.layout.setRowStretchFactor(0, 2)
        self.layout.addWidget(graph, 2, 0, 1, 16)

        # -------- Parent data (contours, traces, background) --------
        if parent is not None and hasattr(parent, 'cnmf'):
            self.img_components = getattr(parent, 'img_components', None)
            self.C = getattr(parent.cnmf, 'estimates', None)
            if self.C is not None:
                self.C = getattr(self.C, 'C', None)
            self.shifts = getattr(parent.cnmf, 'shifts_rig', None)
            self._estimates_b = getattr(parent.cnmf.estimates, 'b', None)
            self._estimates_f = getattr(parent.cnmf.estimates, 'f', None)
            if self.img_components is not None:
                self.showContoursCheck.setEnabled(True)
            if self.C is not None:
                self.component.setMaximum(max(9999, self.C.shape[0] - 1))
            if self._estimates_b is None or self._estimates_f is None:
                idx = self.videoModeCombo.findText(self.VIDEO_BGSUB)
                if idx >= 0:
                    self.videoModeCombo.model().item(idx).setEnabled(False)
        else:
            self.img_components = None
            self.C = None
            self.shifts = None
            self._estimates_b = None
            self._estimates_f = None
            self.showContoursCheck.setEnabled(False)

        self.plot_trace()
        self.plot_contour()

        # -------- Playback --------
        iconSize = QtCore.QSize(24, 24)
        self.playButton = QtWidgets.QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.playButton.setIconSize(iconSize)
        self.playButton.setCheckable(True)
        self.playButton.setEnabled(False)
        self.pauseButton = QtWidgets.QToolButton()
        self.pauseButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
        self.pauseButton.setIconSize(iconSize)
        self.pauseButton.setCheckable(True)
        self.pauseButton.setEnabled(False)
        self.frameSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frameSlider.setTracking(False)
        self.layout.addWidget(self.playButton, 3, 0, 1, 1)
        self.layout.addWidget(self.pauseButton, 3, 1, 1, 1)
        self.layout.addWidget(self.frameSlider, 3, 2, 1, 12)
        self.layout.addWidget(QtWidgets.QLabel('Time:'), 4, 0, 1, 1)
        self.elapsedTime = QtWidgets.QLabel('0:00.0')
        self.layout.addWidget(self.elapsedTime, 4, 1, 1, 2)
        self.playButton.clicked.connect(self.play)
        self.pauseButton.clicked.connect(self.pause)
        self.frameSlider.valueChanged.connect(self.change_slider)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)

        # Load first mmap from combo if we populated it from parent
        if self.mmapCombo.count() > 0:
            path = self.mmapCombo.itemData(0)
            if path:
                self._load_mmap_path(path)

    def on_mmap_combo_changed(self):
        idx = self.mmapCombo.currentIndex()
        if idx < 0:
            return
        path = self.mmapCombo.itemData(idx)
        if path:
            self._load_mmap_path(path)

    def _load_mmap_path(self, fmemap):
        self._dff_baseline = None
        try:
            self.Yr, self.dims, self.nframe = load_memmap(fmemap)
            self.loaded = True
            self.fileLabel.setText('Loaded: ' + os.path.basename(fmemap))
        except Exception:
            self.loaded = False
            self.fileLabel.setText('Load failed: ' + os.path.basename(fmemap))
            self.Yr = None
            self.nframe = 0
        if self.loaded:
            self.saveVideoButton.setEnabled(True)
            self.playButton.setEnabled(True)
            self.frameSlider.setMinimum(0)
            self.frameSlider.setMaximum(max(0, self.nframe - 1))
            self.cframe = 0
            self.vline.setValue(0)
            self.frameSlider.setValue(0)
            self.go_to_frame()

    def open_memap(self):
        fd = os.path.expanduser('~/caiman_data/temp')
        if self._parent and getattr(self._parent, 'fname', None):
            fd = os.path.dirname(self._parent.fname)
        fmemap, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load memory-mapped file', fd, 'MMAP (*.mmap)')
        if not fmemap:
            return
        # Add to combo if not already present
        existing = [self.mmapCombo.itemData(i) for i in range(self.mmapCombo.count())]
        if fmemap not in existing:
            self.mmapCombo.addItem(os.path.basename(fmemap), fmemap)
        self.mmapCombo.setCurrentIndex(self.mmapCombo.findData(fmemap))
        self._load_mmap_path(fmemap)

    def on_video_mode_changed(self, text):
        self._video_mode = text
        self._dff_baseline = None
        if self.loaded:
            self.go_to_frame()

    def on_show_contours_changed(self):
        show = self.showContoursCheck.isChecked()
        self.contour.setVisible(show and self.this_cell >= 0)
        self.plot_contour()

    def change_params(self):
        if self.rate.text():
            try:
                self.fps = float(self.rate.text())
            except ValueError:
                pass
        self.this_cell = self.component.value()
        self.contour.setVisible(self.showContoursCheck.isChecked() and self.this_cell >= 0)
        self.plot_contour()
        self.plot_trace()
        if self.loaded and self.playButton.isEnabled():
            self.go_to_frame()

    def change_slider(self):
        if int(self.vline.value()) != self.frameSlider.value():
            self.vline.setValue(self.frameSlider.value())

    def _get_frame_raw(self, frame_idx):
        """Return one frame (y, x) from mmap."""
        if self.Yr is None:
            return None
        data = self.Yr[:, frame_idx]
        return data.reshape(self.dims, order='F').astype(np.float32)

    def _compute_dff_baseline(self):
        if self._dff_baseline is not None:
            return
        # Percentile (8th) over time per pixel; use a sample if T is large
        T = self.nframe
        step = max(1, T // 500)
        idx = np.arange(0, T, step)
        frames = np.array([self._get_frame_raw(i) for i in idx])
        if frames is None or len(frames) == 0:
            self._dff_baseline = None
            return
        self._dff_baseline = np.percentile(frames, 8, axis=0).astype(np.float32)
        self._dff_baseline[self._dff_baseline < 1e-6] = 1e-6

    def _get_background_frame(self, t):
        if self._estimates_b is None or self._estimates_f is None:
            return None
        b, f = self._estimates_b, self._estimates_f
        if hasattr(b, 'toarray'):
            b = b.toarray()
        if b.ndim == 1:
            bg = (b * f[:, t]).reshape(self.dims, order='F')
        else:
            bg = (b.dot(f[:, t])).reshape(self.dims, order='F')
        return bg.astype(np.float32)

    def _process_frame(self, frame_idx):
        """Get one displayable frame and apply video mode (raw/dF/F/bg sub)."""
        if not self.loaded or self.Yr is None:
            return None
        frame = self._get_frame_raw(frame_idx)
        if frame is None:
            return None
        if self._video_mode == self.VIDEO_DFF:
            self._compute_dff_baseline()
            if self._dff_baseline is not None:
                frame = (frame - self._dff_baseline) / self._dff_baseline
        elif self._video_mode == self.VIDEO_BGSUB:
            t = int(np.clip(frame_idx, 0, self.nframe - 1))
            bg = self._get_background_frame(t)
            if bg is not None:
                frame = frame - bg
        return frame

    def _frame_to_display(self, frame):
        """Stretch contrast and convert to uint8 for display."""
        if frame is None:
            return None
        min_ = np.percentile(frame, self.prct[0]) if self.prct[0] > 0 else np.min(frame)
        max_ = np.percentile(frame, self.prct[1]) if self.prct[1] < 100 else np.max(frame)
        if max_ <= min_:
            max_ = min_ + 1
        out = np.clip(255 * (frame - min_) / (max_ - min_), 0, 255).astype(np.uint8)
        return out

    def next_frame(self):
        self.cframe += 1
        if self.cframe < self.nframe:
            frame = self._process_frame(self.cframe)
            if frame is not None:
                frame = self._frame_to_display(frame)
                self.img.setImage(frame)
            self.vline.setValue(self.cframe)
            self.frameSlider.setValue(self.cframe)
            sec = self.cframe / self.fps
            self.elapsedTime.setText(f'{int(sec) // 60}:{int(sec) % 60:02d}.{int((sec % 1) * 10)}')
        else:
            self.timer.stop()

    def go_to_frame(self):
        if not self.playButton.isEnabled():
            return
        self.cframe = int(np.clip(self.vline.value(), 0, self.nframe - 1))
        self.frameSlider.setValue(self.cframe)
        frame = self._process_frame(self.cframe)
        if frame is not None:
            frame = self._frame_to_display(frame)
            self.img.setImage(frame)
        sec = self.cframe / self.fps
        self.elapsedTime.setText(f'{int(sec) // 60}:{int(sec) % 60:02d}.{int((sec % 1) * 10)}')

    def play(self):
        if self.cframe < self.nframe - 1:
            self.playButton.setEnabled(False)
            self.pauseButton.setEnabled(True)
            self.frameSlider.setEnabled(False)
            self.timer.start(0)

    def pause(self):
        self.timer.stop()
        self.playButton.setEnabled(True)
        self.pauseButton.setEnabled(True)
        self.frameSlider.setEnabled(True)

    def keyPressEvent(self, event):
        if self.playButton.isEnabled():
            if event.key() == QtCore.Qt.Key_Left:
                self.cframe = np.clip(self.cframe - self.dframe, 0, self.nframe - 1)
                self.frameSlider.setValue(self.cframe)
                self.vline.setValue(self.cframe)
            elif event.key() == QtCore.Qt.Key_Right:
                self.cframe = np.clip(self.cframe + self.dframe, 0, self.nframe - 1)
                self.frameSlider.setValue(self.cframe)
                self.vline.setValue(self.cframe)

    def plot_trace(self):
        self.p2.clearPlots()
        if hasattr(self, 'shifts') and self.shifts is not None:
            self.p2.plot(self.shifts[:, 0], pen=(0, 128, 255), name='y shift')
            self.p2.plot(self.shifts[:, 1], pen=(102, 204, 0), name='x shift')
        if self.this_cell >= 0 and self.C is not None:
            fluor = self.C[self.this_cell] / (np.max(self.C[self.this_cell]) + 1e-12) * 10 - 10
            self.p2.plot(fluor, pen=(255, 51, 153), name='fluor')

    def plot_contour(self):
        if self.this_cell >= 0 and self.img_components is not None:
            self.contour.setData(self.img_components[self.this_cell].astype(np.float32))
            self.contour.setVisible(self.showContoursCheck.isChecked())
        else:
            self.contour.setVisible(False)

    def save_as_video(self):
        if not self.loaded or self.Yr is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save video', '', 'AVI (*.avi);;MP4 (*.mp4)')
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        fourcc = cv2.VideoWriter_fourcc(*'XVID') if ext == '.avi' else cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, self.fps, (self.dims[1], self.dims[0]), False)
        if not out.isOpened():
            self.fileLabel.setText('Could not create video file')
            return
        self.fileLabel.setText('Writing video...')
        QtWidgets.QApplication.processEvents()
        for t in range(self.nframe):
            frame = self._process_frame(t)
            if frame is not None:
                disp = self._frame_to_display(frame)
                out.write(cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR))
        out.release()
        self.fileLabel.setText('Saved: ' + os.path.basename(path))


def memap_window(parent):
    win = MemapPlayer(parent)
    win.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MemapPlayer()
    win.show()
    sys.exit(app.exec())
