"""MCP server control panel for ArcGeek Calculator.

The server itself is the plugin side of qgis-mcp by Nicolas Karasiak
(https://github.com/nkarasiak/qgis-mcp), vendored in ``qgis_mcp_server/``.
This module only provides the dialog that starts and stops it.
"""

import json
import sys

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QFont
from qgis.PyQt.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)

from .qgis_mcp_server.constants import DEFAULT_PORT, UPSTREAM_VERSION
from .qgis_mcp_server.registry import COMMANDS
from .qgis_mcp_server.server import QgisMCPServer

# Pinned to the tag the vendored plugin side was copied from: uvx would
# otherwise fetch upstream's latest server, which can call commands or pass
# parameters this copy does not have yet.
MCP_SERVER_SOURCE = f"git+https://github.com/nkarasiak/qgis-mcp@v{UPSTREAM_VERSION}"


class QgisMCPDialog(QDialog):
    """Simple dialog for ArcGeekCalculator MCP integration.

    Used by ArcGeekCalculator's plugin.py to provide a toggle-able
    MCP server control panel embedded in the main plugin.
    """
    closed = pyqtSignal()

    def __init__(self, iface):
        super().__init__(iface.mainWindow())
        self.setWindowTitle("ArcGeek MCP Server")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Tool)
        self.iface = iface
        self.server = None
        self.setMinimumWidth(320)
        self._setup_ui()

    def _setup_ui(self):
        align_center = Qt.AlignmentFlag.AlignCenter

        layout = QVBoxLayout(self)

        info = QLabel(
            "<b>QGIS MCP Server</b><br>"
            f"<small>Powered by <i>nkarasiak/qgis-mcp</i> {UPSTREAM_VERSION} "
            f"&mdash; {len(COMMANDS)} commands</small>"
        )
        info.setAlignment(align_center)
        layout.addWidget(info)

        layout.addWidget(QLabel("Server Port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(DEFAULT_PORT)
        layout.addWidget(self.port_spin)

        self.start_button = QPushButton("Start Server")
        self.start_button.clicked.connect(self.start_server)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Server")
        self.stop_button.clicked.connect(self.stop_server)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        self.status_label = QLabel("Server: Stopped")
        self.status_label.setAlignment(align_center)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        cfg_btn = QPushButton("How to connect (Claude / Antigravity / Cursor / OpenCode)")
        cfg_btn.clicked.connect(self._show_config_dialog)
        layout.addWidget(cfg_btn)

    def _show_config_dialog(self):
        if sys.platform == "win32":
            uv_install = 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
            cfg_path = r"%APPDATA%\Claude\claude_desktop_config.json"
            os_name = "Windows"
        elif sys.platform == "darwin":
            uv_install = "brew install uv"
            cfg_path = "~/Library/Application Support/Claude/claude_desktop_config.json"
            os_name = "macOS"
        else:
            uv_install = "curl -LsSf https://astral.sh/uv/install.sh | sh"
            cfg_path = "~/.config/Claude/claude_desktop_config.json"
            os_name = "Linux"

        config_json = json.dumps({
            "mcpServers": {
                "qgis": {
                    "command": "uvx",
                    "args": ["--from", MCP_SERVER_SOURCE, "qgis-mcp-server"]
                }
            }
        }, indent=2)

        dlg = QDialog(self)
        dlg.setWindowTitle("Connect Claude / Antigravity / Cursor / OpenCode -- MCP Setup")
        dlg.setMinimumWidth(560)
        dlg_layout = QVBoxLayout(dlg)

        header = QLabel(
            "<b>Works with:</b> Claude Code, Claude Desktop, Antigravity, Cursor, OpenCode, "
            "and any MCP-compatible client."
        )
        header.setWordWrap(True)
        dlg_layout.addWidget(header)

        dlg_layout.addWidget(QLabel(f"<br><b>Step 1 &mdash; Install uv on {os_name}:</b>"))
        uv_row = QHBoxLayout()
        uv_edit = QLineEdit(uv_install)
        uv_edit.setReadOnly(True)
        uv_edit.setFont(QFont("Courier New", 8))
        copy_uv_btn = QPushButton("Copy")
        copy_uv_btn.setFixedWidth(55)
        copy_uv_btn.clicked.connect(lambda: QApplication.clipboard().setText(uv_install))
        uv_row.addWidget(uv_edit)
        uv_row.addWidget(copy_uv_btn)
        dlg_layout.addLayout(uv_row)

        dlg_layout.addWidget(QLabel(
            f"<br><b>Step 2 &mdash; Add this to your MCP config file:</b>"
            f"<br><code>{cfg_path}</code>"
        ))
        code_edit = QTextEdit()
        code_edit.setReadOnly(True)
        code_edit.setFont(QFont("Courier New", 9))
        code_edit.setFixedHeight(145)
        code_edit.setPlainText(config_json)
        dlg_layout.addWidget(code_edit)

        note = QLabel(
            "<small>Already configured with an older ArcGeek version? Replace the "
            "<code>--from</code> value with the one above and restart your AI client.</small>"
        )
        note.setWordWrap(True)
        dlg_layout.addWidget(note)

        btn_row = QHBoxLayout()
        copy_btn = QPushButton("Copy JSON config")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(config_json))
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(copy_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        dlg_layout.addLayout(btn_row)

        dlg.exec()

    def start_server(self):
        if not self.server:
            self.server = QgisMCPServer(port=self.port_spin.value(), iface=self.iface)
        if self.server.start():
            self._refresh_ui()
        else:
            reason = self.server.start_error or "unknown error"
            self.server = None
            self._refresh_ui()
            self.status_label.setText(f"Server: failed to start - {reason}")

    def stop_server(self):
        if self.server:
            self.server.stop()
            self.server = None
        self._refresh_ui()

    def _refresh_ui(self):
        """Sync button/label state with actual server state."""
        running = self.server is not None and self.server.running
        if running:
            self.status_label.setText(f"Server: Running on port {self.server.port}")
        else:
            self.status_label.setText("Server: Stopped")
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.port_spin.setEnabled(not running)

    def showEvent(self, event):
        """Refresh UI state every time the dialog is shown."""
        super().showEvent(event)
        self._refresh_ui()

    def closeEvent(self, event):
        """Hide the dialog instead of closing it — server keeps running."""
        event.ignore()
        self.hide()
