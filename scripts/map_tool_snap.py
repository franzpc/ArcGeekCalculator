from qgis.core import (QgsProcessingParameterPoint, QgsPointXY, 
                       QgsSnappingConfig, QgsProject, QgsSnappingUtils,
                       QgsMessageLog, Qgis)
from qgis.gui import QgsMapToolEmitPoint, QgsVertexMarker
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QMessageBox

class MapToolSnapToPour(QgsMapToolEmitPoint):
    def __init__(self, canvas, stream_layer):
        super().__init__(canvas)
        self.canvas = canvas
        self.stream_layer = stream_layer
        self.vertex_marker = None
        
        # Configure snapping globally
        self.config = QgsProject.instance().snappingConfig()
        self.config.setEnabled(True)
        self.config.setMode(QgsSnappingConfig.AllLayers)
        self.config.setTypeFlag(QgsSnappingConfig.VertexAndSegment)
        self.config.setTolerance(20)
        self.config.setUnits(QgsSnappingConfig.Pixels)
        QgsProject.instance().setSnappingConfig(self.config)
        
        self.snapper = QgsSnappingUtils()
        self.snapper.setConfig(self.config)
        self.snapper.setMapSettings(canvas.mapSettings())

    def canvasPressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        # Get clicked point
        point = event.snapPoint()
        
        # Try snapping to nearest vertex/segment
        match = self.snapper.snapToMap(point)
        
        if match.isValid() and match.layer() == self.stream_layer:
            snapped_point = match.point()
            
            # Update marker position
            self.showSnapMarker(snapped_point)
            
            # Emit the point
            self.canvasClicked.emit(snapped_point, event.button())
            QgsMessageLog.logMessage(f"Snapped to: {snapped_point.x()}, {snapped_point.y()}", "WatershedTool")
        else:
            QMessageBox.warning(None, "Warning", 
                "Please click closer to a stream. The point must be on the stream network.")
            return

    def showSnapMarker(self, point):
        if not self.vertex_marker:
            self.vertex_marker = QgsVertexMarker(self.canvas)
            self.vertex_marker.setIconType(QgsVertexMarker.ICON_CROSS)
            self.vertex_marker.setColor(Qt.red)
            self.vertex_marker.setIconSize(10)
        self.vertex_marker.setCenter(point)

    def deactivate(self):
        if self.vertex_marker:
            self.canvas.scene().removeItem(self.vertex_marker)
            self.vertex_marker = None
        super().deactivate()

class SnappingPointParameter(QgsProcessingParameterPoint):
    def __init__(self, name='', description='', defaultValue=None, optional=False):
        super().__init__(name, description, defaultValue, optional)
        self._stream_layer = None

    def setStreamLayer(self, layer):
        self._stream_layer = layer

    def createCustomWidget(self, parent, context):
        widget = super().createCustomWidget(parent, context)
        
        if self._stream_layer and context.mapCanvas():
            tool = MapToolSnapToPour(context.mapCanvas(), self._stream_layer)
            widget.setMapTool(tool)
            
        return widget