"""CSV Table Viewer with Filtering - PySide6 GUI"""
import os
import sys
import json
import csv
import io
import re
import atexit
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTableView,
    QVBoxLayout, QHBoxLayout, QLineEdit, QMessageBox, QComboBox, QFileDialog,
    QListWidget, QListWidgetItem, QSizePolicy, QMenu, QHeaderView, QAbstractItemView,
    QProgressBar, QDialog
)
from PySide6.QtGui import QPalette, QColor, QAction, QFontMetrics
from PySide6.QtCore import Qt, QThread, QObject, Signal, QAbstractTableModel, QModelIndex

NAV_BUTTON_STYLE = """
    QPushButton {
        background-color: #00284d;
        color: #f5f5f5;
        border-radius: 6px;
        padding: 6px 12px;
    }
    QPushButton:hover {
        background-color: #004080;
    }
"""



# =====================================================================
# CONFIGURATION & APP DATA PATHS (same pattern as the diary app)
# =====================================================================
APP_NAME = "CSVViewer"

if sys.platform == "win32":
    DATA_FOLDER = os.path.join(os.environ['APPDATA'], APP_NAME)
else:
    DATA_FOLDER = os.path.expanduser(f"~/.{APP_NAME.lower()}")

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

RECENT_FILE = os.path.join(DATA_FOLDER, "recent_files.json")
MAX_RECENT = 10


def load_recent():
    """Load the list of recently opened CSV paths."""
    if os.path.exists(RECENT_FILE):
        try:
            with open(RECENT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return []
    return []


def save_recent(paths):
    """Persist the list of recently opened CSV paths."""
    try:
        with open(RECENT_FILE, "w", encoding="utf-8") as f:
            json.dump(paths[:MAX_RECENT], f)
    except OSError as e:
        print(f"Error saving recent files: {e}")


def _warm_pandas():
    """Pool initializer: import pandas once per worker process up front, so the
    (fairly heavy) import cost is paid once per process rather than on every
    file load."""
    import pandas as pd  # noqa: F401


_PROCESS_POOL = None


def _get_process_pool():
    """Lazily create one persistent process pool for the app's lifetime, so
    repeated file loads don't keep paying process-spawn + pandas-import cost."""
    global _PROCESS_POOL
    if _PROCESS_POOL is None:
        workers = min(os.cpu_count() or 1, 16)
        _PROCESS_POOL = ProcessPoolExecutor(max_workers=workers, initializer=_warm_pandas)
        atexit.register(_shutdown_process_pool)
    return _PROCESS_POOL


def _shutdown_process_pool():
    global _PROCESS_POOL
    if _PROCESS_POOL is not None:
        _PROCESS_POOL.shutdown(wait=False, cancel_futures=True)
        _PROCESS_POOL = None


def _count_lines(path):
    """Fast newline count via raw binary reads - used only to give the progress
    bar an accurate denominator, far cheaper than actually parsing the file."""
    count = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                return count
            count += chunk.count(b"\n")


def _is_number(value):
    """True if a string parses as a float - used to decide numeric vs text sort."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def _parse_chunk(path, start, end, encoding, headers):
    """Parse one byte range of a CSV file into rows using pandas' C parser.
    Runs in a worker process; must be a module-level function so it can be
    pickled for ProcessPoolExecutor.
    """
    import pandas as pd
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    text = data.decode(encoding, errors="replace")
    df = pd.read_csv(
        io.StringIO(text), header=None, names=headers,
        dtype=str, keep_default_na=False, na_filter=False, engine="c",
    )
    return df.values.tolist()


def _compute_chunk_boundaries(path, header_end, num_chunks):
    """Split the file (after the header) into byte ranges aligned to line breaks."""
    file_size = os.path.getsize(path)
    remaining = file_size - header_end
    if remaining <= 0 or num_chunks <= 1:
        return [(header_end, file_size)]
    approx = remaining // num_chunks
    boundaries = [header_end]
    with open(path, "rb") as f:
        for i in range(1, num_chunks):
            target = header_end + approx * i
            f.seek(target)
            f.readline()  # consume the partial line so the next chunk starts clean
            pos = f.tell()
            if pos > boundaries[-1]:
                boundaries.append(pos)
    boundaries.append(file_size)
    ranges = []
    for i in range(len(boundaries) - 1):
        if boundaries[i] < boundaries[i + 1]:
            ranges.append((boundaries[i], boundaries[i + 1]))
    return ranges


class CSVTableModel(QAbstractTableModel):
    """Virtualized table model: only cells actually on screen are ever touched,
    so setting even a huge dataset is instant instead of building one item per cell."""

    HIGHLIGHT_COLOR = QColor(178, 34, 34)  # firebrick - flagged/matched rows

    def __init__(self, headers=None, rows=None, parent=None):
        super().__init__(parent)
        self._headers = headers or []
        self._rows = rows or []
        self._highlight_terms = set()
        self._highlighted_rows = set()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            row = self._rows[index.row()]
            col = index.column()
            return row[col] if col < len(row) else ""
        if role == Qt.BackgroundRole and index.row() in self._highlighted_rows:
            return self.HIGHLIGHT_COLOR
        if role == Qt.ForegroundRole and index.row() in self._highlighted_rows:
            return QColor(255, 255, 255)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self._headers[section] if section < len(self._headers) else ""
        return str(section + 1)

    def row_at(self, row_idx):
        return self._rows[row_idx]

    def set_data(self, headers, rows):
        self.beginResetModel()
        self._headers = headers
        self._rows = rows
        self._recompute_highlights()
        self.endResetModel()

    def set_highlight_terms(self, terms):
        """terms: a set of lowercase substrings. Any row containing one of them
        in any column is flagged, without removing other rows from the view -
        keeps suspicious traffic visible in its surrounding context."""
        self._highlight_terms = terms
        self._recompute_highlights()
        if self._rows and self._headers:
            top_left = self.index(0, 0)
            bottom_right = self.index(len(self._rows) - 1, len(self._headers) - 1)
            self.dataChanged.emit(top_left, bottom_right, [Qt.BackgroundRole, Qt.ForegroundRole])

    def _recompute_highlights(self):
        if not self._highlight_terms:
            self._highlighted_rows = set()
            return
        terms = self._highlight_terms
        highlighted = set()
        for i, row in enumerate(self._rows):
            for value in row:
                value_lower = value.lower()
                if any(term in value_lower for term in terms):
                    highlighted.add(i)
                    break
        self._highlighted_rows = highlighted


class CSVLoadWorker(QObject):
    """Reads a CSV file on a background thread and reports progress by bytes read."""
    progress = Signal(int)
    # No payload here on purpose: Qt marshals signal arguments across threads
    # through its meta-object system, and for a huge nested list (millions of
    # rows) that marshaling can take far longer than the actual CSV parsing.
    # The parsed data is stashed on self.headers/self.rows instead, and the
    # main thread reads it directly once this fires.
    finished = Signal()
    error = Signal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path
        self._cancelled = False
        self.headers = []
        self.rows = []

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            file_size = os.path.getsize(self.path)
            cpu_count = os.cpu_count() or 1
            parallel_threshold = 10 * 1024 * 1024  # 10 MB
            use_parallel = file_size >= parallel_threshold and cpu_count > 1 and not self._file_has_quotes()
            if use_parallel:
                try:
                    self._run_parallel(file_size, cpu_count)
                    return
                except Exception:
                    # Any problem spawning/using worker processes falls back to
                    # the always-correct single-process reader.
                    if self._cancelled:
                        return
            self._run_sequential(file_size)
        except OSError as e:
            self.error.emit(str(e))
        except csv.Error as e:
            self.error.emit(str(e))

    def _file_has_quotes(self):
        """Quick binary scan: if there's no quote character anywhere, no field can
        contain an embedded newline, so splitting the file by byte offset is safe."""
        try:
            with open(self.path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        return False
                    if b'"' in chunk:
                        return True
        except OSError:
            return True  # be conservative and use the safe sequential path

    def _run_sequential(self, file_size):
        """Single-process read using pandas' C parser (roughly 2x+ faster than
        the pure-python csv module), used for small files or files with quoted
        fields (where embedded newlines make byte-based splitting unsafe)."""
        total_lines = _count_lines(self.path) if file_size else 0
        headers = None
        rows = []
        processed = 0
        last_pct = -1
        chunk_reader = pd.read_csv(
            self.path, dtype=str, keep_default_na=False, na_filter=False,
            engine="c", encoding="utf-8-sig", chunksize=50000,
        )
        for chunk_df in chunk_reader:
            if self._cancelled:
                return
            if headers is None:
                headers = list(chunk_df.columns)
            chunk_rows = chunk_df.values.tolist()
            rows.extend(chunk_rows)
            processed += len(chunk_rows)
            if total_lines:
                pct = min(int(processed / total_lines * 100), 99)
                if pct != last_pct:
                    last_pct = pct
                    self.progress.emit(pct)
        self.headers = headers or []
        self.rows = rows
        self.progress.emit(100)
        self.finished.emit()

    def _run_parallel(self, file_size, cpu_count):
        """Split the file into byte-aligned chunks and parse them in parallel
        worker processes using pandas' C parser (real multi-core speedup;
        Python threads can't do this for CPU-bound parsing because of the GIL).
        Uses a persistent, pre-warmed pool so pandas' import cost is only paid
        once per app session, not on every file load."""
        encoding = "utf-8-sig"
        with open(self.path, "rb") as f:
            header_line = f.readline()
            header_end = f.tell()
        headers = next(csv.reader(io.StringIO(header_line.decode(encoding, errors="replace"))))

        pool = _get_process_pool()
        num_chunks = min(cpu_count, 16)
        ranges = _compute_chunk_boundaries(self.path, header_end, num_chunks)
        if len(ranges) <= 1:
            self._run_sequential(file_size)
            return

        rows_by_index = {}
        total = len(ranges)
        completed = 0
        future_map = {
            pool.submit(_parse_chunk, self.path, start, end, encoding, headers): idx
            for idx, (start, end) in enumerate(ranges)
        }
        for future in as_completed(future_map):
            if self._cancelled:
                for f in future_map:
                    f.cancel()
                return
            idx = future_map[future]
            rows_by_index[idx] = future.result()
            completed += 1
            self.progress.emit(min(int(completed / total * 99), 99))

        if self._cancelled:
            return
        rows = []
        for idx in range(total):
            rows.extend(rows_by_index[idx])
        self.headers = headers
        self.rows = rows
        self.progress.emit(100)
        self.finished.emit()


class CSVViewerApp(QWidget):
    """Main application window for the CSV viewer."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Table Viewer")
        self.resize(1100, 750)
        self.dark_mode = True
        self.headers = []
        self.rows = []            # all rows loaded from the current file
        self.active_filters = []  # list of (col_index, col_name, mode, text)
        self.sort_column = None
        self.sort_ascending = True
        self.last_filtered_rows = []
        self.current_path = None
        self.recent_files = load_recent()
        self.load_thread = None
        self.load_worker = None
        self.init_ui()
        self.toggle_mode()  # applies the initial (dark) styling
        self.refresh_recent_menu()

    def init_ui(self):
        """Build the interface."""
        self.label_title = QLabel("CSV Table Viewer")
        self.label_title.setStyleSheet("font-weight: bold; font-size: 20px;")

        self.btn_open = QPushButton("Open CSV")
        self.btn_open.clicked.connect(self.open_file)

        self.btn_recent = QPushButton("Recent Files \u25be")
        self.recent_menu = QMenu(self.btn_recent)
        self.btn_recent.setMenu(self.recent_menu)

        self.btn_toggle = QPushButton("Toggle Light/Dark")
        self.btn_toggle.clicked.connect(self.toggle_mode)

        top_hbox = QHBoxLayout()
        top_hbox.addWidget(self.label_title)
        top_hbox.addStretch()
        top_hbox.addWidget(self.btn_open)
        top_hbox.addWidget(self.btn_recent)
        top_hbox.addWidget(self.btn_toggle)

        # --- Filter bar ---
        self.combo_column = QComboBox()
        self.combo_column.setMinimumWidth(160)
        self.combo_filter_mode = QComboBox()
        self.combo_filter_mode.addItems([
            "Contains", "Not Contains", "Exact Match", "Regex", "In List (comma-separated)"
        ])
        self.combo_filter_mode.setMinimumWidth(170)
        self.filter_text = QLineEdit()
        self.filter_text.setPlaceholderText("Filter value...")
        self.filter_text.returnPressed.connect(self.add_filter)
        self.btn_add_filter = QPushButton("Add Filter")
        self.btn_add_filter.clicked.connect(self.add_filter)
        self.btn_clear_filters = QPushButton("Clear Filters")
        self.btn_clear_filters.clicked.connect(self.clear_filters)
        self.combo_combine = QComboBox()
        self.combo_combine.addItems(["Match ALL filters (AND)", "Match ANY filter (OR)"])
        self.combo_combine.setMinimumWidth(190)
        self.combo_combine.currentIndexChanged.connect(self.apply_filters)

        filter_hbox = QHBoxLayout()
        filter_hbox.addWidget(QLabel("Column:"))
        filter_hbox.addWidget(self.combo_column)
        filter_hbox.addWidget(self.combo_filter_mode)
        filter_hbox.addWidget(self.filter_text)
        filter_hbox.addWidget(self.btn_add_filter)
        filter_hbox.addWidget(self.btn_clear_filters)
        filter_hbox.addWidget(self.combo_combine)

        # --- Preset bar: reuse a detection rule set across similar log files ---
        self.btn_save_preset = QPushButton("Save Filters...")
        self.btn_save_preset.clicked.connect(self.save_filter_preset)
        self.btn_load_preset = QPushButton("Load Filters...")
        self.btn_load_preset.clicked.connect(self.load_filter_preset)

        # --- Highlight bar: flag rows in place without filtering others out ---
        self.filter_highlight = QLineEdit()
        self.filter_highlight.setPlaceholderText("Highlight values (comma-separated IOCs)...")
        self.filter_highlight.returnPressed.connect(self.apply_highlight)
        self.btn_apply_highlight = QPushButton("Highlight")
        self.btn_apply_highlight.clicked.connect(self.apply_highlight)
        self.btn_clear_highlight = QPushButton("Clear Highlight")
        self.btn_clear_highlight.clicked.connect(self.clear_highlight)

        preset_hbox = QHBoxLayout()
        preset_hbox.addWidget(self.filter_highlight, stretch=1)
        preset_hbox.addWidget(self.btn_apply_highlight)
        preset_hbox.addWidget(self.btn_clear_highlight)
        preset_hbox.addWidget(self.btn_save_preset)
        preset_hbox.addWidget(self.btn_load_preset)

        # --- Analysis bar: sort hint, export, and frequency/outlier detection ---
        self.btn_export = QPushButton("Export Filtered to CSV")
        self.btn_export.clicked.connect(self.export_filtered)
        self.btn_value_counts = QPushButton("Value Counts...")
        self.btn_value_counts.clicked.connect(self.show_value_counts)
        self.btn_copy_selected = QPushButton("Copy Selected Rows")
        self.btn_copy_selected.clicked.connect(self.copy_selected_rows)
        analysis_hbox = QHBoxLayout()
        analysis_hbox.addWidget(QLabel("Click a column header to sort it \u2014 click again to reverse."))
        analysis_hbox.addStretch()
        analysis_hbox.addWidget(self.btn_copy_selected)
        analysis_hbox.addWidget(self.btn_value_counts)
        analysis_hbox.addWidget(self.btn_export)

        # Active filter chips (double-click a chip to remove it)
        self.list_filters = QListWidget()
        self.list_filters.setFixedHeight(60)
        self.list_filters.setFlow(QListWidget.LeftToRight)
        self.list_filters.setWrapping(True)
        self.list_filters.itemDoubleClicked.connect(self.remove_filter)

        # --- Table (virtualized: only visible cells are ever rendered) ---
        self.table_model = CSVTableModel()
        self.table = QTableView()
        self.table.setModel(self.table_model)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.horizontalHeader().sectionClicked.connect(self.sort_by_column)

        self.label_status = QLabel("No file loaded")

        # --- Progress bar (hidden until a load/render is in progress) ---
        self.label_progress = QLabel("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.btn_cancel_load = QPushButton("Cancel")
        self.btn_cancel_load.clicked.connect(self.cancel_load)
        progress_hbox = QHBoxLayout()
        progress_hbox.addWidget(self.label_progress)
        progress_hbox.addWidget(self.progress_bar, stretch=1)
        progress_hbox.addWidget(self.btn_cancel_load)
        self.label_progress.hide()
        self.progress_bar.hide()
        self.btn_cancel_load.hide()

        main_vbox = QVBoxLayout()
        main_vbox.addLayout(top_hbox)
        main_vbox.addLayout(filter_hbox)
        main_vbox.addLayout(preset_hbox)
        main_vbox.addWidget(QLabel("Active filters (double-click to remove):"))
        main_vbox.addWidget(self.list_filters)
        main_vbox.addLayout(analysis_hbox)
        main_vbox.addWidget(self.table, stretch=1)
        main_vbox.addLayout(progress_hbox)
        main_vbox.addWidget(self.label_status)
        self.setLayout(main_vbox)

    # ---------------- File handling ----------------
    def open_file(self):
        """Open a file picker and load the chosen CSV."""
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)")
        if path:
            self.load_csv(path)

    def load_csv(self, path):
        """Kick off a background read of the CSV so the UI stays responsive."""
        if self.load_thread is not None:
            self.show_info("A file is already loading. Please wait or cancel it first.")
            return
        self.set_loading_ui(True, "Reading file...")
        self._pending_path = path

        self.load_thread = QThread(self)
        self.load_worker = CSVLoadWorker(path)
        self.load_worker.moveToThread(self.load_thread)
        self.load_thread.started.connect(self.load_worker.run)
        self.load_worker.progress.connect(self.on_load_progress)
        self.load_worker.finished.connect(self.on_load_finished)
        self.load_worker.error.connect(self.on_load_error)
        self.load_thread.start()

    def cancel_load(self):
        """Cancel an in-progress file read."""
        if self.load_worker is not None:
            self.load_worker.cancel()
        self.label_progress.setText("Cancelling...")

    def on_load_progress(self, pct):
        """Update the progress bar as the worker reads the file."""
        self.progress_bar.setValue(pct)

    def on_load_error(self, message):
        """Handle a read failure from the worker thread."""
        self.teardown_load_thread()
        self.set_loading_ui(False)
        QMessageBox.critical(self, "Error", f"Could not read file:\n{message}")

    def on_load_finished(self):
        """Data is ready on the worker (self.load_worker.headers/.rows) - read it
        directly rather than via signal arguments (see CSVLoadWorker for why)."""
        headers = self.load_worker.headers
        rows = self.load_worker.rows
        self.teardown_load_thread()
        path = self._pending_path
        if not headers:
            self.set_loading_ui(False)
            QMessageBox.information(self, "Empty File", "The selected CSV file is empty.")
            return

        self.headers = headers
        self.rows = rows
        self.current_path = path
        self.active_filters = []
        self.sort_column = None
        self.sort_ascending = True
        self.combo_combine.setCurrentIndex(0)
        self.filter_highlight.clear()
        self.list_filters.clear()
        self.combo_column.clear()
        self.combo_column.addItem("(All Columns)")
        self.combo_column.addItems(self.headers)
        self.add_to_recent(path)
        self.apply_filters()
        self.table_model.set_highlight_terms(set())
        self.set_loading_ui(False)

    def teardown_load_thread(self):
        """Cleanly stop and release the background thread/worker."""
        if self.load_thread is not None:
            self.load_thread.quit()
            self.load_thread.wait()
            self.load_thread.deleteLater()
        self.load_thread = None
        self.load_worker = None

    def set_loading_ui(self, loading, phase_text=""):
        """Show/hide the progress bar and disable controls while busy."""
        self.progress_bar.setVisible(loading)
        self.label_progress.setVisible(loading)
        self.btn_cancel_load.setVisible(loading)
        self.label_progress.setText(phase_text)
        self.progress_bar.setValue(0)
        for widget in (self.btn_open, self.btn_recent, self.btn_add_filter,
                       self.btn_clear_filters, self.filter_text, self.combo_column,
                       self.combo_filter_mode, self.combo_combine, self.btn_export,
                       self.btn_value_counts, self.btn_copy_selected, self.filter_highlight,
                       self.btn_apply_highlight, self.btn_clear_highlight,
                       self.btn_save_preset, self.btn_load_preset):
            widget.setEnabled(not loading)
        QApplication.processEvents()

    def add_to_recent(self, path):
        """Push a path to the top of the recent-files cache."""
        if path in self.recent_files:
            self.recent_files.remove(path)
        self.recent_files.insert(0, path)
        self.recent_files = self.recent_files[:MAX_RECENT]
        save_recent(self.recent_files)
        self.refresh_recent_menu()

    def refresh_recent_menu(self):
        """Rebuild the Recent Files dropdown menu."""
        self.recent_menu.clear()
        if not self.recent_files:
            action = QAction("No recent files", self)
            action.setEnabled(False)
            self.recent_menu.addAction(action)
            return
        for path in self.recent_files:
            display = path if len(path) < 60 else f"...{path[-57:]}"
            action = QAction(display, self)
            action.triggered.connect(lambda checked=False, p=path: self.open_recent(p))
            self.recent_menu.addAction(action)
        self.recent_menu.addSeparator()
        clear_action = QAction("Clear Recent Files", self)
        clear_action.triggered.connect(self.clear_recent)
        self.recent_menu.addAction(clear_action)

    def open_recent(self, path):
        """Open a file chosen from the recent-files menu."""
        if not os.path.exists(path):
            QMessageBox.warning(self, "File Not Found", f"This file no longer exists:\n{path}")
            self.recent_files.remove(path)
            save_recent(self.recent_files)
            self.refresh_recent_menu()
            return
        self.load_csv(path)

    def clear_recent(self):
        """Empty the recent-files cache."""
        self.recent_files = []
        save_recent(self.recent_files)
        self.refresh_recent_menu()

    # ---------------- Filtering ----------------
    def add_filter(self):
        """Add a filter (with mode) to the active filter list."""
        if not self.headers:
            self.show_info("Load a CSV file first.")
            return
        text = self.filter_text.text().strip()
        if not text:
            return
        mode = self.combo_filter_mode.currentText()
        if mode == "Regex":
            try:
                re.compile(text)
            except re.error as e:
                self.show_info(f"That's not a valid regular expression:\n{e}")
                return
        combo_idx = self.combo_column.currentIndex()
        col_index = combo_idx - 1  # -1 = "(All Columns)", 0.. = a real column
        col_name = self.combo_column.currentText()
        self.active_filters.append((col_index, col_name, mode, text))
        item = QListWidgetItem(f'{col_name} [{mode}]: "{text}"  \u2715')
        self.list_filters.addItem(item)
        self.filter_text.clear()
        self.apply_filters()

    def remove_filter(self, item):
        """Remove a filter chip on double-click."""
        row = self.list_filters.row(item)
        self.list_filters.takeItem(row)
        del self.active_filters[row]
        self.apply_filters()

    def clear_filters(self):
        """Remove all active filters."""
        self.active_filters = []
        self.list_filters.clear()
        self.apply_filters()

    def apply_filters(self):
        """Re-run every active filter (combined per the AND/OR selector) and refresh the table."""
        if not self.active_filters:
            filtered = self.rows
        elif self.combo_combine.currentIndex() == 0:  # Match ALL (AND)
            filtered = self.rows
            for col_index, _col_name, mode, text in self.active_filters:
                filtered = [row for row in filtered if self._row_matches(row, col_index, mode, text)]
        else:  # Match ANY (OR)
            filtered = [
                row for row in self.rows
                if any(self._row_matches(row, ci, m, t) for ci, _, m, t in self.active_filters)
            ]

        if self.sort_column is not None:
            filtered = self._sorted_rows(filtered, self.sort_column, self.sort_ascending)

        self.last_filtered_rows = filtered
        self.populate_table(filtered)
        file_label = os.path.basename(self.current_path) if self.current_path else "No file"
        self.label_status.setText(f"{file_label} \u2014 {len(filtered)} of {len(self.rows)} rows shown")

    @staticmethod
    def _row_matches(row, col_index, mode, text):
        """Test one row against one filter. col_index == -1 means search every column."""
        values = row if col_index == -1 else [row[col_index]] if col_index < len(row) else [""]
        text_lower = text.lower()

        if mode == "Contains":
            return any(text_lower in v.lower() for v in values)
        if mode == "Not Contains":
            return all(text_lower not in v.lower() for v in values)
        if mode == "Exact Match":
            return any(v.lower() == text_lower for v in values)
        if mode == "Regex":
            pattern = re.compile(text, re.IGNORECASE)
            return any(pattern.search(v) for v in values)
        if mode == "In List (comma-separated)":
            wanted = {part.strip().lower() for part in text.split(",") if part.strip()}
            return any(v.strip().lower() in wanted for v in values)
        return True

    def sort_by_column(self, section_index):
        """Toggle ascending/descending sort when a header is clicked."""
        if section_index == self.sort_column:
            self.sort_ascending = not self.sort_ascending
        else:
            self.sort_column = section_index
            self.sort_ascending = True
        self.apply_filters()

    @staticmethod
    def _sorted_rows(rows, col_index, ascending):
        """Sort rows by one column, numerically when the column's values look
        numeric, otherwise as case-insensitive text. Blanks always sort last,
        in either direction, since an empty value isn't meaningfully high or low."""
        def get(row):
            return row[col_index] if col_index < len(row) else ""

        non_blank = [row for row in rows if get(row) != ""]
        blank = [row for row in rows if get(row) == ""]

        sample = [get(row) for row in non_blank[:200]]
        numeric = bool(sample) and all(_is_number(v) for v in sample)

        if numeric:
            def numeric_key(row):
                try:
                    return float(get(row))
                except ValueError:
                    return float("inf")
            non_blank.sort(key=numeric_key, reverse=not ascending)
        else:
            non_blank.sort(key=lambda row: get(row).lower(), reverse=not ascending)

        return non_blank + blank

    def populate_table(self, rows):
        """Push rows into the virtualized model. This is effectively instant no
        matter how many rows there are, since the view only ever asks the model
        for the cells it's actually about to paint."""
        self.table_model.set_data(self.headers, rows)
        if len(rows) <= 5000:
            self.table.resizeColumnsToContents()
        else:
            # resizeColumnsToContents() measures every row's content, which gets
            # slow at huge sizes. Size columns from the header text instead and
            # let the user drag-resize any column that needs more room.
            metrics = QFontMetrics(self.table.font())
            for col, name in enumerate(self.headers):
                width = metrics.horizontalAdvance(str(name)) + 40
                self.table.setColumnWidth(col, max(80, min(width, 300)))

    def copy_selected_rows(self):
        """Copy the selected table rows to the clipboard as tab-separated text,
        ready to paste into a ticket, email, or spreadsheet."""
        selection = self.table.selectionModel()
        if not selection or not selection.hasSelection():
            self.show_info("Select one or more rows first.")
            return
        row_indices = sorted({idx.row() for idx in selection.selectedRows()})
        lines = ["\t".join(self.headers)]
        for r in row_indices:
            row_data = self.table_model.row_at(r)
            lines.append("\t".join(str(v) for v in row_data))
        QApplication.clipboard().setText("\n".join(lines))
        self.show_info(f"Copied {len(row_indices)} row(s) to the clipboard.")

    # ---------------- Highlighting (flag IOC matches in place) ----------------
    def apply_highlight(self):
        """Flag rows containing any of the given terms, without hiding the rest -
        useful for spotting known-bad indicators in their surrounding traffic."""
        if not self.headers:
            self.show_info("Load a CSV file first.")
            return
        text = self.filter_highlight.text().strip()
        terms = {part.strip().lower() for part in text.split(",") if part.strip()}
        self.table_model.set_highlight_terms(terms)

    def clear_highlight(self):
        """Remove all row highlighting."""
        self.filter_highlight.clear()
        self.table_model.set_highlight_terms(set())

    # ---------------- Filter presets (reuse a rule set across similar logs) ----------------
    def save_filter_preset(self):
        """Save the active filters, combine mode, and highlight terms to a JSON
        file, so the same detection rule set can be reapplied to future logs
        with the same schema."""
        if not self.active_filters and not self.filter_highlight.text().strip():
            self.show_info("No filters or highlight terms to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Filter Preset", "", "Filter Presets (*.json)")
        if not path:
            return
        data = {
            "combine_mode": self.combo_combine.currentIndex(),
            "filters": [list(f) for f in self.active_filters],
            "highlight_terms": self.filter_highlight.text(),
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            QMessageBox.critical(self, "Save Failed", f"Could not save preset:\n{e}")
            return
        self.show_info(f"Saved filter preset to:\n{path}")

    def load_filter_preset(self):
        """Load a previously saved filter/highlight rule set."""
        path, _ = QFileDialog.getOpenFileName(self, "Load Filter Preset", "", "Filter Presets (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            QMessageBox.critical(self, "Load Failed", f"Could not read preset:\n{e}")
            return
        if not self.headers:
            self.show_info("Load a CSV file first, then load the preset.")
            return

        self.combo_combine.setCurrentIndex(data.get("combine_mode", 0))
        self.active_filters = [tuple(f) for f in data.get("filters", [])]
        self.list_filters.clear()
        for col_index, col_name, mode, text in self.active_filters:
            item = QListWidgetItem(f'{col_name} [{mode}]: "{text}"  \u2715')
            self.list_filters.addItem(item)
        self.filter_highlight.setText(data.get("highlight_terms", ""))
        self.apply_highlight()
        self.apply_filters()

    def show_info(self, message):
        """Show an informational message box."""
        QMessageBox.information(self, "Info", message)

    # ---------------- Export & analysis ----------------
    def export_filtered(self):
        """Write the currently filtered/sorted rows to a new CSV file."""
        if not self.headers:
            self.show_info("Load a CSV file first.")
            return
        if not self.last_filtered_rows:
            self.show_info("There are no rows to export with the current filters.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Filtered Rows", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
                writer.writerows(self.last_filtered_rows)
        except OSError as e:
            QMessageBox.critical(self, "Export Failed", f"Could not write file:\n{e}")
            return
        self.show_info(f"Exported {len(self.last_filtered_rows)} rows to:\n{path}")

    def show_value_counts(self):
        """Show the most frequent values in the selected column, among the
        currently filtered rows. A single value dominating the count (e.g. one
        source IP hammering a host) is a classic scanning/beaconing signature."""
        if not self.headers:
            self.show_info("Load a CSV file first.")
            return
        combo_idx = self.combo_column.currentIndex()
        col_index = combo_idx - 1
        if col_index == -1:
            self.show_info("Pick a specific column (not \"(All Columns)\") to count its values.")
            return
        col_name = self.combo_column.currentText()
        source_rows = self.last_filtered_rows if self.active_filters or self.sort_column is not None else self.rows
        counter = Counter(row[col_index] for row in source_rows if col_index < len(row))
        if not counter:
            self.show_info("No values to count.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Value Counts \u2014 {col_name}")
        dialog.resize(420, 500)
        layout = QVBoxLayout()
        total_rows = sum(counter.values())
        layout.addWidget(QLabel(f"{len(counter)} unique values across {total_rows} rows. Top 200 shown, most frequent first:"))
        list_widget = QListWidget()
        for value, count in counter.most_common(200):
            pct = (count / total_rows * 100) if total_rows else 0
            display = value if value else "(blank)"
            list_widget.addItem(f"{count:>8}  ({pct:5.1f}%)   {display}")
        layout.addWidget(list_widget)
        btn_close = QPushButton("Close")
        btn_close.setStyleSheet(NAV_BUTTON_STYLE)
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close)
        dialog.setLayout(layout)
        dialog.exec()

    # ---------------- Styling (same palette as the diary app) ----------------
    def toggle_mode(self):
        """Toggle between light and dark mode."""
        qt_app = QApplication.instance()
        nav_button_style = NAV_BUTTON_STYLE
        buttons = [self.btn_open, self.btn_recent, self.btn_toggle, self.btn_add_filter,
                   self.btn_clear_filters, self.btn_cancel_load, self.btn_export, self.btn_value_counts,
                   self.btn_copy_selected, self.btn_apply_highlight, self.btn_clear_highlight,
                   self.btn_save_preset, self.btn_load_preset]

        if self.dark_mode:
            # switching TO light mode
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(190, 195, 190))
            palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
            palette.setColor(QPalette.Base, QColor(160, 168, 168))
            palette.setColor(QPalette.AlternateBase, QColor(190, 195, 190))
            palette.setColor(QPalette.ToolTipBase, QColor(140, 140, 140))
            palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
            palette.setColor(QPalette.Text, QColor(0, 0, 0))
            palette.setColor(QPalette.Button, QColor(0, 40, 77))
            palette.setColor(QPalette.ButtonText, QColor(245, 245, 245))
            palette.setColor(QPalette.BrightText, QColor(0, 0, 0))
            palette.setColor(QPalette.Link, QColor(0, 100, 220))
            palette.setColor(QPalette.Highlight, QColor(51, 153, 255))
            palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
            qt_app.setPalette(palette)
            for btn in buttons:
                btn.setStyleSheet(nav_button_style)
            self.label_title.setStyleSheet("font-weight: bold; font-size: 20px; color: black;")
            table_style = "color: black; background-color: rgb(160,168,168); gridline-color: rgb(120,128,128);"
            self.table.setStyleSheet(table_style)
            self.list_filters.setStyleSheet(table_style)
            self.dark_mode = False
        else:
            # switching TO dark mode
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(38, 40, 44))
            palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
            palette.setColor(QPalette.Base, QColor(28, 30, 34))
            palette.setColor(QPalette.AlternateBase, QColor(48, 50, 55))
            palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
            palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
            palette.setColor(QPalette.Text, QColor(220, 220, 220))
            palette.setColor(QPalette.Button, QColor(0, 40, 77))
            palette.setColor(QPalette.ButtonText, QColor(245, 245, 245))
            palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
            palette.setColor(QPalette.Link, QColor(85, 170, 255))
            palette.setColor(QPalette.Highlight, QColor(0, 85, 255))
            palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
            qt_app.setPalette(palette)
            for btn in buttons:
                btn.setStyleSheet(nav_button_style)
            self.label_title.setStyleSheet("font-weight: bold; font-size: 20px; color: white;")
            table_style = "color: white; background-color: rgb(28,30,34); gridline-color: rgb(60,60,65);"
            self.table.setStyleSheet(table_style)
            self.list_filters.setStyleSheet(table_style)
            self.dark_mode = True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CSVViewerApp()
    window.show()
    sys.exit(app.exec())