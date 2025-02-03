import os
from qgis.PyQt.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QFileDialog, QSpinBox
from qgis.PyQt.QtCore import Qt
from qgis.core import QgsProject, QgsMapSettings, QgsMapRendererSequentialJob, QgsRasterLayer, QgsCoordinateReferenceSystem
from qgis.gui import QgsMapCanvas
from osgeo import gdal, osr

class ScreenCaptureDialog(QDialog):
    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self.setWindowTitle("Screen Capture")
        self.layout = QVBoxLayout()

        self.output_layout = QHBoxLayout()
        self.output_layout.addWidget(QLabel("Output File:"))
        self.output_file = QLineEdit()
        self.output_layout.addWidget(self.output_file)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_output_file)
        self.output_layout.addWidget(self.browse_button)
        self.layout.addLayout(self.output_layout)

        self.zoom_layout = QHBoxLayout()
        self.zoom_layout.addWidget(QLabel("Zoom Factor:"))
        self.zoom_factor = QSpinBox()
        self.zoom_factor.setRange(1, 4) #zoom
        self.zoom_factor.setValue(2)
        self.zoom_layout.addWidget(self.zoom_factor)
        self.layout.addLayout(self.zoom_layout)

        self.capture_button = QPushButton("Capture and Save")
        self.capture_button.clicked.connect(self.capture_and_save)
        self.layout.addWidget(self.capture_button)

        self.setLayout(self.layout)

    def browse_output_file(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save File", "", "GeoTIFF Files (*.tif)")
        if file_name:
            self.output_file.setText(file_name)

    def capture_and_save(self):
        output_file = self.output_file.text()
        if not output_file:
            QMessageBox.warning(self, "Error", "Please specify an output file")
            return

        if not output_file.lower().endswith('.tif'):
            output_file += '.tif'

        if os.path.exists(output_file):
            QMessageBox.warning(self, "Error", "File already exists. Please choose a different name.")
            return

        zoom = self.zoom_factor.value()

        # Capture the current view with increased resolution
        canvas = self.iface.mapCanvas()
        settings = canvas.mapSettings()
        settings.setOutputSize(canvas.size() * zoom)
        job = QgsMapRendererSequentialJob(settings)
        job.start()
        job.waitForFinished()
        image = job.renderedImage()

        # Save the image as GeoTIFF
        extent = canvas.extent()
        xres = extent.width() / (canvas.width() * zoom)
        yres = extent.height() / (canvas.height() * zoom)

        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(output_file, image.width(), image.height(), 3, gdal.GDT_Byte)

        # Get the current CRS
        crs = canvas.mapSettings().destinationCrs()
        wkt = crs.toWkt()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(wkt)
        ds.SetProjection(srs.ExportToWkt())

        ds.SetGeoTransform([extent.xMinimum(), xres, 0, extent.yMaximum(), 0, -yres])

        # Convert QImage to byte array and write to dataset
        byte_array = image.bits().asstring(image.byteCount())
        for i in range(3):
            band = ds.GetRasterBand(3-i)  # Reverse order of bands
            band_array = bytearray(image.width() * image.height())
            for j in range(0, len(byte_array), 4):
                band_array[j//4] = byte_array[j+i]
            band.WriteRaster(0, 0, image.width(), image.height(), bytes(band_array))

        ds = None  # Close the dataset

        # Add the captured image to the current view
        layer = QgsRasterLayer(output_file, "Screen Capture")
        if layer.isValid():
            QgsProject.instance().addMapLayer(layer)
        else:
            QMessageBox.warning(self, "Error", "Failed to add captured image to the view")

        QMessageBox.information(self, "Success", f"Screen capture saved as {output_file} and added to the view")
        self.close()

def run_screen_capture(iface):
    dialog = ScreenCaptureDialog(iface)
    dialog.exec_()