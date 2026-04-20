import os
import io
import sys
import json
import shutil
import socket
import traceback
from qgis.core import *
from qgis.gui import *
from qgis.PyQt.QtCore import QObject, pyqtSignal, QTimer, Qt, QSize
from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                  QPushButton, QSpinBox, QWidget, QLineEdit,
                                  QTextEdit, QApplication)
from qgis.PyQt.QtGui import QIcon, QColor, QFont  # QFont used in config dialog

from qgis.PyQt.QtWidgets import QAction
from qgis.utils import active_plugins

SERVER_DIR = os.path.join(os.path.expanduser("~"), ".qgis_mcp_server")
_SRC_DIR   = os.path.join(os.path.dirname(__file__), "server_mcp")


def _setup_server_files():
    """Copy server files to ~/.qgis_mcp_server (first-time only). No uv sync needed."""
    server_py = os.path.join(SERVER_DIR, "qgis_mcp_server.py")
    if os.path.isfile(server_py):
        return True, None

    os.makedirs(SERVER_DIR, exist_ok=True)
    for fname in ("qgis_mcp_server.py", "pyproject.toml", "uv.lock"):
        shutil.copy2(os.path.join(_SRC_DIR, fname), SERVER_DIR)
    return True, None


def _build_config_json():
    """Return the Claude Desktop config JSON string with the correct server path."""
    uv_name = "uv.exe" if sys.platform == "win32" else "uv"
    uv = shutil.which("uv") or os.path.join(os.path.expanduser("~"), ".local", "bin", uv_name)
    config = {
        "mcpServers": {
            "qgis": {
                "command": uv,
                "args": ["--directory", SERVER_DIR, "run", "qgis_mcp_server.py"]
            }
        }
    }
    return json.dumps(config, indent=2)

# Compatibilidad de enums QGIS 3 / QGIS 4
try:
    VECTOR_LAYER_TYPE = Qgis.LayerType.Vector
    RASTER_LAYER_TYPE = Qgis.LayerType.Raster
except AttributeError:
    VECTOR_LAYER_TYPE = QgsMapLayer.VectorLayer
    RASTER_LAYER_TYPE = QgsMapLayer.RasterLayer

try:
    MSG_CRITICAL = Qgis.MessageLevel.Critical
    MSG_WARNING = Qgis.MessageLevel.Warning
    MSG_INFO = Qgis.MessageLevel.Info
except AttributeError:
    MSG_CRITICAL = Qgis.Critical
    MSG_WARNING = Qgis.Warning
    MSG_INFO = Qgis.Info

class QgisMCPServer(QObject):
    """Server class to handle socket connections and execute QGIS commands"""
    
    def __init__(self, host='localhost', port=9876, iface=None):
        super().__init__()
        self.host = host
        self.port = port
        self.iface = iface
        self.running = False
        self.socket = None
        self.client = None
        self.buffer = b''
        self.timer = None
    
    def start(self):
        """Start the server"""
        self.running = True
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.socket.setblocking(False)
            
            # Create a timer to process server operations
            self.timer = QTimer()
            self.timer.timeout.connect(self.process_server)
            self.timer.start(100)  # 100ms interval
            
            QgsMessageLog.logMessage(f"QGIS MCP server started on {self.host}:{self.port}", "QGIS MCP")
            return True
        except Exception as e:
            QgsMessageLog.logMessage(f"Failed to start server: {str(e)}", "QGIS MCP", MSG_CRITICAL)
            self.stop()
            return False
            
    def stop(self):
        """Stop the server"""
        self.running = False
        
        if self.timer:
            self.timer.stop()
            self.timer = None
            
        if self.socket:
            self.socket.close()
        if self.client:
            self.client.close()
            
        self.socket = None
        self.client = None
        QgsMessageLog.logMessage("QGIS MCP server stopped", "QGIS MCP")
    
    def process_server(self):
        """Process server operations (called by timer)"""
        if not self.running:
            return
            
        try:
            # Accept new connections
            if not self.client and self.socket:
                try:
                    self.client, address = self.socket.accept()
                    self.client.setblocking(False)
                    QgsMessageLog.logMessage(f"Connected to client: {address}", "QGIS MCP")
                except BlockingIOError:
                    pass  # No connection waiting
                except Exception as e:
                    QgsMessageLog.logMessage(f"Error accepting connection: {str(e)}", "QGIS MCP", MSG_WARNING)
                
            # Process existing connection
            if self.client:
                try:
                    # Try to receive data
                    try:
                        data = self.client.recv(8192)
                        if data:
                            self.buffer += data
                            # Try to process complete messages
                            try:
                                # Attempt to parse the buffer as JSON
                                command = json.loads(self.buffer.decode('utf-8'))
                                # If successful, clear the buffer and process command
                                self.buffer = b''
                                response = self.execute_command(command)
                                response_json = json.dumps(response)
                                self.client.sendall(response_json.encode('utf-8'))
                            except json.JSONDecodeError:
                                # Incomplete data, keep in buffer
                                pass
                        else:
                            # Connection closed by client
                            QgsMessageLog.logMessage("Client disconnected", "QGIS MCP")
                            self.client.close()
                            self.client = None
                            self.buffer = b''
                    except BlockingIOError:
                        pass  # No data available
                    except Exception as e:
                        QgsMessageLog.logMessage(f"Error receiving data: {str(e)}", "QGIS MCP", MSG_WARNING)
                        self.client.close()
                        self.client = None
                        self.buffer = b''
                        
                except Exception as e:
                    QgsMessageLog.logMessage(f"Error with client: {str(e)}", "QGIS MCP", MSG_WARNING)
                    if self.client:
                        self.client.close()
                        self.client = None
                    self.buffer = b''
                    
        except Exception as e:
            QgsMessageLog.logMessage(f"Server error: {str(e)}", "QGIS MCP", MSG_CRITICAL)

    def execute_command(self, command):
        """Execute a command"""
        try:
            cmd_type = command.get("type")
            params = command.get("params", {})
            
            handlers = {
                "ping": self.ping,
                "get_qgis_info": self.get_qgis_info,
                "load_project": self.load_project,
                "get_project_info": self.get_project_info,
                "execute_code": self.execute_code,
                "add_vector_layer": self.add_vector_layer,
                "add_raster_layer": self.add_raster_layer,
                "get_layers": self.get_layers,
                "remove_layer": self.remove_layer,
                "zoom_to_layer": self.zoom_to_layer,
                "get_layer_features": self.get_layer_features,
                "execute_processing": self.execute_processing,
                "save_project": self.save_project,
                "render_map": self.render_map,
                "create_new_project": self.create_new_project,
            }
            
            handler = handlers.get(cmd_type)
            if handler:
                try:
                    QgsMessageLog.logMessage(f"Executing handler for {cmd_type}", "QGIS MCP")
                    result = handler(**params)
                    QgsMessageLog.logMessage(f"Handler execution complete", "QGIS MCP")
                    return {"status": "success", "result": result}
                except Exception as e:
                    QgsMessageLog.logMessage(f"Error in handler: {str(e)}", "QGIS MCP", MSG_CRITICAL)
                    traceback.print_exc()
                    return {"status": "error", "message": str(e)}
            else:
                return {"status": "error", "message": f"Unknown command type: {cmd_type}"}
                
        except Exception as e:
            QgsMessageLog.logMessage(f"Error executing command: {str(e)}", "QGIS MCP", MSG_CRITICAL)
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
    
    # Command handlers
    def ping(self, **kwargs):
        """Simple ping command"""
        return {"pong": True}
    
    def get_qgis_info(self, **kwargs):
        """Get basic QGIS information"""
        return {
            "qgis_version": Qgis.version(),
            "profile_folder": QgsApplication.qgisSettingsDirPath(),
            "plugins_count": len(active_plugins)
        }
    
    def get_project_info(self, **kwargs):
        """Get information about the current QGIS project"""
        project = QgsProject.instance()
        
        # Get basic project information
        info = {
            "filename": project.fileName(),
            "title": project.title(),
            "layer_count": len(project.mapLayers()),
            "crs": project.crs().authid(),
            "layers": []
        }
        
        # Add basic layer information (limit to 10 layers for performance)
        layers = list(project.mapLayers().values())
        for i, layer in enumerate(layers):
            if i >= 10:  # Limit to 10 layers
                break
                
            layer_info = {
                "id": layer.id(),
                "name": layer.name(),
                "type": self._get_layer_type(layer),
                "visible": layer.isValid() and project.layerTreeRoot().findLayer(layer.id()).isVisible()
            }
            info["layers"].append(layer_info)
        
        return info
    
    def _get_layer_type(self, layer):
        """Helper to get layer type as string"""
        if layer.type() == VECTOR_LAYER_TYPE:
            return f"vector_{layer.geometryType()}"
        elif layer.type() == RASTER_LAYER_TYPE:
            return "raster"
        else:
            return str(layer.type())
    
    def execute_code(self, code, **kwargs):
        """Execute arbitrary PyQGIS code"""

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Store original stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        try:
            # Redirect stdout and stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Create a local namespace for execution
            namespace = {
                "qgis": Qgis,
                "QgsProject": QgsProject,
                "iface": self.iface,
                "QgsApplication": QgsApplication,
                "QgsVectorLayer": QgsVectorLayer,
                "QgsRasterLayer": QgsRasterLayer,
                "QgsCoordinateReferenceSystem": QgsCoordinateReferenceSystem
            }
            
            # Execute the code
            exec(code, namespace)  # nosec B102
            
            # Restore stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
            return {
                "executed": True,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue()
            }
        except Exception as e:
            # Generate full traceback
            error_traceback = traceback.format_exc()
            
            # Restore stdout and stderr in case of exception
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
            return {
                "executed": False,
                "error": str(e),
                "traceback": error_traceback,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue()
            }
    
    def add_vector_layer(self, path, name=None, provider="ogr", **kwargs):
        """Add a vector layer to the project"""
        if not name:
            name = os.path.basename(path)
            
        # Create the layer
        layer = QgsVectorLayer(path, name, provider)
        
        if not layer.isValid():
            raise Exception(f"Layer is not valid: {path}")
        
        # Add to project
        QgsProject.instance().addMapLayer(layer)
        
        return {
            "id": layer.id(),
            "name": layer.name(),
            "type": self._get_layer_type(layer),
            "feature_count": layer.featureCount()
        }
    
    def add_raster_layer(self, path, name=None, provider="gdal", **kwargs):
        """Add a raster layer to the project"""
        if not name:
            name = os.path.basename(path)
            
        # Create the layer
        layer = QgsRasterLayer(path, name, provider)
        
        if not layer.isValid():
            raise Exception(f"Layer is not valid: {path}")
        
        # Add to project
        QgsProject.instance().addMapLayer(layer)
        
        return {
            "id": layer.id(),
            "name": layer.name(),
            "type": "raster",
            "width": layer.width(),
            "height": layer.height()
        }
    
    def get_layers(self, **kwargs):
        """Get all layers in the project"""
        project = QgsProject.instance()
        layers = []
        
        for layer_id, layer in project.mapLayers().items():
            layer_info = {
                "id": layer_id,
                "name": layer.name(),
                "type": self._get_layer_type(layer),
                "visible": project.layerTreeRoot().findLayer(layer_id).isVisible()
            }
            
            # Add type-specific information
            if layer.type() == VECTOR_LAYER_TYPE:
                layer_info.update({
                    "feature_count": layer.featureCount(),
                    "geometry_type": int(layer.geometryType())
                })
            elif layer.type() == RASTER_LAYER_TYPE:
                layer_info.update({
                    "width": layer.width(),
                    "height": layer.height()
                })
                
            layers.append(layer_info)
        
        return layers
    
    def remove_layer(self, layer_id, **kwargs):
        """Remove a layer from the project"""
        project = QgsProject.instance()
        
        if layer_id in project.mapLayers():
            project.removeMapLayer(layer_id)
            return {"removed": layer_id}
        else:
            raise Exception(f"Layer not found: {layer_id}")
    
    def zoom_to_layer(self, layer_id, **kwargs):
        """Zoom to a layer's extent"""
        project = QgsProject.instance()
        
        if layer_id in project.mapLayers():
            layer = project.mapLayer(layer_id)
            self.iface.setActiveLayer(layer)
            self.iface.zoomToActiveLayer()
            return {"zoomed_to": layer_id}
        else:
            raise Exception(f"Layer not found: {layer_id}")
    
    def get_layer_features(self, layer_id, limit=10, **kwargs):
        """Get features from a vector layer"""
        project = QgsProject.instance()
        
        if layer_id in project.mapLayers():
            layer = project.mapLayer(layer_id)
            
            if layer.type() != VECTOR_LAYER_TYPE:
                raise Exception(f"Layer is not a vector layer: {layer_id}")
            
            features = []
            for i, feature in enumerate(layer.getFeatures()):
                if i >= limit:
                    break
                    
                # Extract attributes
                attrs = {}
                for field in layer.fields():
                    attrs[field.name()] = feature.attribute(field.name())
                
                # Extract geometry if available
                geom = None
                if feature.hasGeometry():
                    geom = {
                        "type": int(feature.geometry().type()),
                        "wkt": feature.geometry().asWkt(precision=4)
                    }
                
                features.append({
                    "id": feature.id(),
                    "attributes": attrs,
                    "geometry": geom
                })
            
            return {
                "layer_id": layer_id,
                "feature_count": layer.featureCount(),
                "features": features,
                "fields": [field.name() for field in layer.fields()]
            }
        else:
            raise Exception(f"Layer not found: {layer_id}")
    
    def execute_processing(self, algorithm, parameters, **kwargs):
        """Execute a processing algorithm"""
        try:
            import processing
            result = processing.run(algorithm, parameters)
            return {
                "algorithm": algorithm,
                "result": {k: str(v) for k, v in result.items()}  # Convert values to strings for JSON
            }
        except Exception as e:
            raise Exception(f"Processing error: {str(e)}")
    
    def save_project(self, path=None, **kwargs):
        """Save the current project"""
        project = QgsProject.instance()
        
        if not path and not project.fileName():
            raise Exception("No project path specified and no current project path")
        
        save_path = path if path else project.fileName()
        if project.write(save_path):
            return {"saved": save_path}
        else:
            raise Exception(f"Failed to save project to {save_path}")
    
    def load_project(self, path, **kwargs):
        """Load a project"""
        project = QgsProject.instance()
        
        if project.read(path):
            self.iface.mapCanvas().refresh()
            return {
                "loaded": path,
                "layer_count": len(project.mapLayers())
            }
        else:
            raise Exception(f"Failed to load project from {path}")
    
    def create_new_project(self, path, **kwargs):
        """
        Creates a new QGIS project and saves it at the specified path.
        If a project is already loaded, it clears it before creating the new one.
        
        :param project_path: Full path where the project will be saved
                            (e.g., 'C:/path/to/project.qgz')
        """
        project = QgsProject.instance()
        
        if project.fileName():
            project.clear()
        
        project.setFileName(path)
        self.iface.mapCanvas().refresh()
        
        # Save the project
        if project.write():
            return {
                "created": f"Project created and saved successfully at: {path}",
                "layer_count": len(project.mapLayers())
            }
        else:
            raise Exception(f"Failed to save project to {path}")
    
    def render_map(self, path, width=800, height=600, **kwargs):
        """Render the current map view to an image"""
        try:
            # Create map settings
            ms = QgsMapSettings()
            
            # Set layers to render
            layers = list(QgsProject.instance().mapLayers().values())
            ms.setLayers(layers)
            
            # Set map canvas properties
            rect = self.iface.mapCanvas().extent()
            ms.setExtent(rect)
            ms.setOutputSize(QSize(width, height))
            ms.setBackgroundColor(QColor(255, 255, 255))
            ms.setOutputDpi(96)
            
            # Create the render
            render = QgsMapRendererParallelJob(ms)
            
            # Start rendering
            render.start()
            render.waitForFinished()
            
            # Get the image and save
            img = render.renderedImage()
            if img.save(path):
                return {
                    "rendered": True,
                    "path": path,
                    "width": width,
                    "height": height
                }
            else:
                raise Exception(f"Failed to save rendered image to {path}")
                
        except Exception as e:
            raise Exception(f"Render error: {str(e)}")


class QgisMCPDialog(QDialog):
    """Dialog for the QGIS MCP plugin"""
    closed = pyqtSignal()
    
    def __init__(self, iface):
        super().__init__(iface.mainWindow())
        self.setWindowTitle("MCP Server")
        # Compatibilidad Qt5 / Qt6 para WindowFlags
        if hasattr(Qt, "WindowType") and hasattr(Qt.WindowType, "Tool"):
            tool_flag = Qt.WindowType.Tool
        else:
            tool_flag = Qt.Tool
        self.setWindowFlags(self.windowFlags() | tool_flag)
        self.iface = iface
        self.server = None
        self.setMinimumWidth(300)
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the dialog UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Compatibilidad Qt5 / Qt6 para AlignCenter
        try:
            align_center = Qt.AlignmentFlag.AlignCenter
        except AttributeError:
            align_center = Qt.AlignCenter

        info_label = QLabel("<b>QGIS MCP Server</b><br>Original project: <i>QGISMCP</i> by jjsantos01")
        info_label.setAlignment(align_center)
        layout.addWidget(info_label)
        
        # Add port selection
        layout.addWidget(QLabel("Server Port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setMinimum(1024)
        self.port_spin.setMaximum(65535)
        self.port_spin.setValue(9876)
        layout.addWidget(self.port_spin)
        
        # Add server control buttons
        self.start_button = QPushButton("Start Server")
        self.start_button.clicked.connect(self.start_server)
        layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Server")
        self.stop_button.clicked.connect(self.stop_server)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)
        
        # Add status label
        self.status_label = QLabel("Server: Stopped")
        self.status_label.setAlignment(align_center)
        layout.addWidget(self.status_label)

        cfg_btn = QPushButton("How to connect Claude Code / Antigravity / Cursor")
        cfg_btn.clicked.connect(self._show_config_dialog)
        layout.addWidget(cfg_btn)

        self.setup_label = QLabel("")
        self.setup_label.setAlignment(align_center)
        layout.addWidget(self.setup_label)
        self._refresh_setup_label()

    def _refresh_setup_label(self):
        ready = os.path.isfile(os.path.join(SERVER_DIR, "qgis_mcp_server.py"))
        if ready:
            self.setup_label.setText('<span style="color:green">Server files ready</span>')
        else:
            self.setup_label.setText('<span style="color:orange">Files will be copied on first Start</span>')

    def _show_config_dialog(self):
        if sys.platform == "win32":
            uv_install = (
                'powershell -ExecutionPolicy ByPass -c '
                '"irm https://astral.sh/uv/install.ps1 | iex"'
            )
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

        dlg = QDialog(self)
        dlg.setWindowTitle("Connect Claude Code / Antigravity / Cursor — MCP Setup")
        dlg.setMinimumWidth(520)
        layout = QVBoxLayout(dlg)

        header = QLabel(
            "<b>Works with:</b> Claude Code, Claude Desktop, Antigravity, Cursor, "
            "and any MCP-compatible client."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        layout.addWidget(QLabel(f"<b>Step 1 — Install uv on {os_name}:</b>"))
        uv_row = QHBoxLayout()
        uv_edit = QLineEdit(uv_install)
        uv_edit.setReadOnly(True)
        uv_edit.setFont(QFont("Courier New", 8))
        copy_uv_btn = QPushButton("Copy")
        copy_uv_btn.setFixedWidth(55)
        copy_uv_btn.clicked.connect(lambda: QApplication.clipboard().setText(uv_install))
        uv_row.addWidget(uv_edit)
        uv_row.addWidget(copy_uv_btn)
        layout.addLayout(uv_row)

        layout.addWidget(QLabel(
            f"<b>Step 2 — Add this block to your MCP config file:</b><br>"
            f"<code>{cfg_path}</code>"
        ))
        code_edit = QTextEdit()
        code_edit.setReadOnly(True)
        code_edit.setFont(QFont("Courier New", 9))
        code_edit.setFixedHeight(120)
        code_edit.setPlainText(_build_config_json())
        layout.addWidget(code_edit)

        btn_row = QHBoxLayout()
        copy_btn = QPushButton("Copy configuration")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(_build_config_json()))
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(copy_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        try:
            dlg.exec()
        except AttributeError:
            dlg.exec_()

    def start_server(self):
        """Start the server, copying files on first run if needed."""
        if not os.path.isfile(os.path.join(SERVER_DIR, "qgis_mcp_server.py")):
            ok, err = _setup_server_files()
            if not ok:
                self.status_label.setText(f"Setup failed: {err}")
                QgsMessageLog.logMessage(f"MCP setup error: {err}", "QGIS MCP", MSG_CRITICAL)
                return
            self._refresh_setup_label()

        if not self.server:
            port = self.port_spin.value()
            self.server = QgisMCPServer(port=port, iface=self.iface)

        if self.server.start():
            self.status_label.setText(f"Server: Running on port {self.server.port}")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.port_spin.setEnabled(False)
    
    def stop_server(self):
        """Stop the server"""
        if self.server:
            self.server.stop()
            self.server = None
            
        self.status_label.setText("Server: Stopped")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.port_spin.setEnabled(True)
        
    def closeEvent(self, event):
        """Handle dialog close"""
        self.closed.emit()
        super().closeEvent(event)


class QgisMCPPlugin:
    """Main plugin class for QGIS MCP"""
    
    def __init__(self, iface):
        self.iface = iface
        self.dock_widget = None
        self.action = None
    
    def initGui(self):
        """Initialize GUI"""
        # Create action
        self.action = QAction(
            "QGIS MCP",
            self.iface.mainWindow()
        )
        self.action.setCheckable(True)
        self.action.triggered.connect(self.toggle_dock)
        
        # Add to plugins menu and toolbar
        self.iface.addPluginToMenu("QGIS MCP", self.action)
        if hasattr(self.iface, 'addToolBarIcon'):
            self.iface.addToolBarIcon(self.action)
        else:
            self.iface.pluginToolBar().addAction(self.action)
    
    def toggle_dock(self, checked):
        """Toggle the dock widget"""
        if checked:
            # Create dock widget if it doesn't exist
            if not self.dock_widget:
                self.dock_widget = QgisMCPDockWidget(self.iface)
                # Compatibilidad Qt6 / Qt5 para enums
                if hasattr(Qt, "DockWidgetArea") and hasattr(Qt.DockWidgetArea, "RightDockWidgetArea"):
                    dock_area = Qt.DockWidgetArea.RightDockWidgetArea
                else:
                    dock_area = Qt.RightDockWidgetArea
                    
                self.iface.addDockWidget(dock_area, self.dock_widget)
                # Connect close event
                self.dock_widget.closed.connect(self.dock_closed)
            else:
                # Show existing dock widget
                self.dock_widget.show()
        else:
            # Hide dock widget
            if self.dock_widget:
                self.dock_widget.hide()
    
    def dock_closed(self):
        """Handle dock widget closed"""
        self.action.setChecked(False)
    
    def unload(self):
        """Unload plugin"""
        # Stop server if running
        if self.dock_widget:
            self.dock_widget.stop_server()
            self.iface.removeDockWidget(self.dock_widget)
            self.dock_widget = None
            
        # Remove plugin menu item and toolbar icon
        self.iface.removePluginMenu("QGIS MCP", self.action)
        if hasattr(self.iface, 'removeToolBarIcon'):
            self.iface.removeToolBarIcon(self.action)
        else:
            self.iface.pluginToolBar().removeAction(self.action)


# Plugin entry point
def classFactory(iface):
    return QgisMCPPlugin(iface)
