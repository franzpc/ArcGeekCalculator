from qgis.PyQt.QtWidgets import QDialog, QVBoxLayout, QLineEdit, QLabel, QPushButton, QComboBox, QCheckBox, QHBoxLayout, QTableWidget, QTableWidgetItem, QAbstractItemView, QApplication
from qgis.PyQt.QtCore import Qt, QSignalBlocker
from qgis.PyQt.QtGui import QIcon, QFont
from qgis.core import (QgsPointXY, QgsGeometry, QgsFeature, QgsVectorLayer, QgsProject, 
                       QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsCoordinateTransformContext, Qgis)
from qgis.gui import QgsProjectionSelectionWidget, QgsMapToolEmitPoint
from qgis.utils import iface

class GoToXYDialog(QDialog):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle("Go to XY")
        
        self.layout = QVBoxLayout()
        
        self.coord_type = QComboBox()
        self.coord_type.addItems(["Projected Coordinates", "Decimal Degrees"])
        self.coord_type.currentIndexChanged.connect(self.update_input_labels)
        self.layout.addWidget(QLabel("Coordinate Type:"))
        self.layout.addWidget(self.coord_type)
        
        self.input_layout = QHBoxLayout()
        self.label_x = QLabel("X:")
        self.input_x = QLineEdit()
        self.input_x.setPlaceholderText("699550")
        self.label_y = QLabel("Y:")
        self.input_y = QLineEdit()
        self.input_y.setPlaceholderText("9557824")
        self.input_layout.addWidget(self.label_x)
        self.input_layout.addWidget(self.input_x)
        self.input_layout.addWidget(self.label_y)
        self.input_layout.addWidget(self.input_y)
        self.layout.addLayout(self.input_layout)
        
        self.crs_selector = QgsProjectionSelectionWidget()
        project_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        self.crs_selector.setCrs(project_crs)
        self.crs_selector.crsChanged.connect(self.crs_changed)
        self.layout.addWidget(QLabel("Input CRS:"))
        self.layout.addWidget(self.crs_selector)
        
        self.create_point = QCheckBox("Create point marker")
        self.create_point.setChecked(True)
        self.layout.addWidget(self.create_point)
        
        self.button = QPushButton("Go")
        self.button.clicked.connect(self.go_to_coordinates)
        self.button.setStyleSheet("QPushButton { font-weight: bold; font-size: 14px; padding: 10px; }")
        self.layout.addWidget(self.button)
        
        self.history_table = QTableWidget(0, 3)
        self.history_table.setHorizontalHeaderLabels(["X/Longitude", "Y/Latitude", "CRS"])
        
        try:
            self.history_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.history_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        except AttributeError:
            self.history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.history_table.setSelectionMode(QAbstractItemView.SingleSelection)
        
        self.history_table.itemSelectionChanged.connect(self.load_history_item)
        self.layout.addWidget(QLabel("History:"))
        self.layout.addWidget(self.history_table)
        
        self.capture_button = QPushButton("Capture Coordinate")
        self.capture_button.clicked.connect(self.capture_coordinate)
        self.layout.addWidget(self.capture_button)
        
        self.button_layout = QHBoxLayout()
        
        self.clear_history_button = QPushButton("Clear History")
        self.clear_history_button.clicked.connect(self.clear_history)
        self.button_layout.addWidget(self.clear_history_button)
        
        self.copy_coordinates_button = QPushButton("Copy Coordinates")
        self.copy_coordinates_button.clicked.connect(self.copy_coordinates)
        self.button_layout.addWidget(self.copy_coordinates_button)
        
        self.layout.addLayout(self.button_layout)
        
        self.setLayout(self.layout)
        
        self.map_tool = QgsMapToolEmitPoint(self.iface.mapCanvas())
        self.map_tool.canvasClicked.connect(self.map_clicked)
        
        self.crs_changed(project_crs)
        
    def update_input_labels(self):
        coord_type = self.coord_type.currentText()
        if coord_type == "Decimal Degrees":
            self.label_x.setText("Longitude:")
            self.label_y.setText("Latitude:")
            self.input_x.setPlaceholderText("-79.1994")
            self.input_y.setPlaceholderText("-3.9961")
            with QSignalBlocker(self.crs_selector):
                self.crs_selector.setCrs(QgsCoordinateReferenceSystem("EPSG:4326"))
            self.crs_selector.setEnabled(False)
        else:
            self.label_x.setText("X:")
            self.label_y.setText("Y:")
            self.input_x.setPlaceholderText("699550")
            self.input_y.setPlaceholderText("9557824")
            self.crs_selector.setEnabled(True)
    
    def crs_changed(self, crs):
        if crs.authid() == "EPSG:4326":
            self.coord_type.setCurrentText("Decimal Degrees")
        else:
            self.coord_type.setCurrentText("Projected Coordinates")
        self.update_input_labels()
    
    def get_coordinates(self):
        try:
            x = float(self.input_x.text().replace(',', '.'))
            y = float(self.input_y.text().replace(',', '.'))
        except ValueError:
            raise ValueError("Please enter valid numeric coordinates.")
        
        coord_type = self.coord_type.currentText()
        crs = self.crs_selector.crs()
        create_point = self.create_point.isChecked()
        return x, y, coord_type, crs, create_point
    
    def go_to_coordinates(self):
        try:
            x, y, coord_type, crs, create_point = self.get_coordinates()
            self.go_to_xy(x, y, coord_type, crs, create_point)
            self.add_to_history(x, y, crs)
        except ValueError as e:
            self.iface.messageBar().pushMessage("Error", str(e), level=Qgis.Warning)
        except Exception as e:
            self.iface.messageBar().pushMessage("Error", f"An unexpected error occurred: {str(e)}", level=Qgis.Critical)
    
    def add_to_history(self, x, y, crs):
        row_position = self.history_table.rowCount()
        self.history_table.insertRow(row_position)
        self.history_table.setItem(row_position, 0, QTableWidgetItem(str(x)))
        self.history_table.setItem(row_position, 1, QTableWidgetItem(str(y)))
        self.history_table.setItem(row_position, 2, QTableWidgetItem(crs.authid()))
    
    def load_history_item(self):
        selected_rows = self.history_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        
        row = selected_rows[0].row()
        x = self.history_table.item(row, 0).text()
        y = self.history_table.item(row, 1).text()
        crs_authid = self.history_table.item(row, 2).text()
        
        self.input_x.setText(x)
        self.input_y.setText(y)
        
        crs = QgsCoordinateReferenceSystem(crs_authid)
        self.crs_selector.setCrs(crs)
        
        self.crs_changed(crs)
    
    def clear_history(self):
        self.history_table.setRowCount(0)
    
    def capture_coordinate(self):
        try:
            self.iface.mapCanvas().setMapTool(self.map_tool)
            self.iface.messageBar().pushMessage("Info", "Click on the map to capture a coordinate", level=Qgis.Info)
        except Exception as e:
            self.iface.messageBar().pushMessage("Error", f"Failed to activate map tool: {str(e)}", level=Qgis.Critical)
    
    def map_clicked(self, point, button):
        try:
            project_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            
            self.crs_selector.setCrs(project_crs)
            
            self.input_x.setText(f"{point.x():.6f}")
            self.input_y.setText(f"{point.y():.6f}")
            
            self.add_to_history(point.x(), point.y(), project_crs)
            
            self.iface.mapCanvas().unsetMapTool(self.map_tool)
            self.iface.messageBar().clearWidgets()
            
            self.crs_changed(project_crs)
            
        except Exception as e:
            self.iface.messageBar().pushMessage("Error", f"Failed to capture coordinate: {str(e)}", level=Qgis.Critical)

    def go_to_xy(self, x, y, coord_type, source_crs, create_point):
        canvas = self.iface.mapCanvas()
        dest_crs = canvas.mapSettings().destinationCrs()
        
        point = QgsPointXY(x, y)
        
        try:
            if source_crs != dest_crs:
                transform = QgsCoordinateTransform(source_crs, dest_crs, QgsProject.instance())
                point = transform.transform(point)
            
            canvas.setCenter(point)
            canvas.zoomScale(5000)
            canvas.refresh()
            
            if create_point:
                self.create_point_marker(point, dest_crs)
        except Exception as e:
            self.iface.messageBar().pushMessage("Error", f"Failed to transform coordinates: {str(e)}", level=Qgis.Warning, duration=5)

    def create_point_marker(self, point, crs):
        vl = QgsVectorLayer("Point?crs={}".format(crs.authid()), "Go to XY Point", "memory")
        pr = vl.dataProvider()
        
        fet = QgsFeature()
        fet.setGeometry(QgsGeometry.fromPointXY(point))
        pr.addFeature(fet)
        
        vl.updateExtents()
        QgsProject.instance().addMapLayer(vl)

    def copy_coordinates(self):
        if self.history_table.rowCount() == 0:
            self.iface.messageBar().pushMessage("Info", "No coordinates to copy.", level=Qgis.Info)
            return
        
        coordinates = ["X/Lon\tY/Lat\tCRS"]
        for row in range(self.history_table.rowCount()):
            x = self.history_table.item(row, 0).text()
            y = self.history_table.item(row, 1).text()
            crs = self.history_table.item(row, 2).text()
            coordinates.append(f"{x}\t{y}\t{crs}")
        
        coordinate_text = "\n".join(coordinates)
        clipboard = QApplication.clipboard()
        clipboard.setText(coordinate_text)
        
        self.iface.messageBar().pushMessage("Success", "Coordinates copied to clipboard.", level=Qgis.Success)