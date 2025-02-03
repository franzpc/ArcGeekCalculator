from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing, QgsProcessingAlgorithm, QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterField, QgsProcessingParameterRasterDestination,
                       QgsProcessingParameterEnum, QgsProcessingParameterNumber, 
                       QgsProcessingException, QgsRasterFileWriter, QgsWkbTypes,
                       QgsRasterBlock, QgsPointXY, QgsRectangle, QgsProcessingParameterExtent,
                       Qgis, QgsMessageLog, QgsRasterLayer)
import sys
import os
import traceback

# Check for required libraries
MISSING_DEPENDENCIES = []
try:
    import numpy as np
except ImportError:
    MISSING_DEPENDENCIES.append("numpy")

try:
    from pykrige.ok import OrdinaryKriging
    from pykrige.uk import UniversalKriging
except ImportError:
    MISSING_DEPENDENCIES.append("pykrige")

try:
    from scipy.spatial.distance import pdist
except ImportError:
    MISSING_DEPENDENCIES.append("scipy")

class KrigingAnalysisAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    Z_FIELD = 'Z_FIELD'
    OUTPUT_KRIGING = 'OUTPUT_KRIGING'
    VARIOGRAM_MODEL = 'VARIOGRAM_MODEL'
    VARIOGRAM_PRESET = 'VARIOGRAM_PRESET'
    KRIGING_METHOD = 'KRIGING_METHOD'
    CELL_SIZE = 'CELL_SIZE'
    EXTENT = 'EXTENT'
    MIN_VALUE = 'MIN_VALUE'
    MAX_VALUE = 'MAX_VALUE'

    def initAlgorithm(self, config=None):
        # Check for missing dependencies before initializing
        if MISSING_DEPENDENCIES:
            return

        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                self.tr('Input point layer'),
                [QgsProcessing.TypeVectorPoint]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.Z_FIELD,
                self.tr('Z value field'),
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.VARIOGRAM_MODEL,
                self.tr('Variogram model'),
                options=['linear', 'power', 'gaussian', 'spherical', 'exponential'],
                defaultValue=2
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.VARIOGRAM_PRESET,
                self.tr('Variogram parameters preset'),
                options=['Default', 'Low range', 'High range', 'Low nugget', 'High nugget'],
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.KRIGING_METHOD,
                self.tr('Kriging method'),
                options=['ordinary', 'universal'],
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CELL_SIZE,
                self.tr('Cell size'),
                type=QgsProcessingParameterNumber.Double,
                minValue=0.0,
                defaultValue=100.0
            )
        )
        self.addParameter(
            QgsProcessingParameterExtent(
                self.EXTENT,
                self.tr('Interpolation extent')
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_VALUE,
                self.tr('Minimum value'),
                type=QgsProcessingParameterNumber.Double,
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_VALUE,
                self.tr('Maximum value'),
                type=QgsProcessingParameterNumber.Double,
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_KRIGING,
                self.tr('Output Kriging Interpolation')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        # Check for missing dependencies before processing
        if MISSING_DEPENDENCIES:
            missing_libs = ", ".join(MISSING_DEPENDENCIES)
            install_instructions = "\n".join([
                f"pip install {lib}" for lib in MISSING_DEPENDENCIES
            ])
            raise QgsProcessingException(self.tr(
                f"The following libraries are required but not installed: {missing_libs}\n"
                f"Please install them using pip. You can use the following commands:\n"
                f"{install_instructions}\n"
                f"After installing, restart QGIS and try running the tool again."
            ))

        QgsMessageLog.logMessage("Starting Kriging Analysis", 'KrigingAnalysis', Qgis.Info)

        source = self.parameterAsSource(parameters, self.INPUT, context)
        z_field = self.parameterAsString(parameters, self.Z_FIELD, context)
        variogram_model = self.parameterAsEnum(parameters, self.VARIOGRAM_MODEL, context)
        variogram_preset = self.parameterAsEnum(parameters, self.VARIOGRAM_PRESET, context)
        kriging_method = self.parameterAsEnum(parameters, self.KRIGING_METHOD, context)
        cell_size = self.parameterAsDouble(parameters, self.CELL_SIZE, context)
        extent = self.parameterAsExtent(parameters, self.EXTENT, context)
        min_value = parameters[self.MIN_VALUE]
        max_value = parameters[self.MAX_VALUE]
        output_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_KRIGING, context)

        QgsMessageLog.logMessage(f"Parameters: z_field={z_field}, variogram_model={variogram_model}, variogram_preset={variogram_preset}, kriging_method={kriging_method}, cell_size={cell_size}, extent={extent}, min_value={min_value}, max_value={max_value}, output_raster={output_raster}", 'KrigingAnalysis', Qgis.Info)

        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        try:
            from osgeo import gdal
        except ImportError:
            raise QgsProcessingException(self.tr("GDAL library is not available. Please install GDAL."))

        # Prepare data
        x, y, z = [], [], []
        for feature in source.getFeatures():
            geom = feature.geometry()
            if geom:
                point = geom.asPoint()
                x.append(point.x())
                y.append(point.y())
                z.append(feature[z_field])

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        if min_value is None or parameters[self.MIN_VALUE] is None:
            min_value = np.min(z)
        if max_value is None or parameters[self.MAX_VALUE] is None:
            max_value = np.max(z)

        QgsMessageLog.logMessage(f"Data prepared: x={x.shape}, y={y.shape}, z={z.shape}", 'KrigingAnalysis', Qgis.Info)
        QgsMessageLog.logMessage(f"Data sample: x[0]={x[0]}, y[0]={y[0]}, z[0]={z[0]}", 'KrigingAnalysis', Qgis.Info)
        QgsMessageLog.logMessage(f"Value range: min={min_value}, max={max_value}", 'KrigingAnalysis', Qgis.Info)

        # Prepare kriging parameters
        variogram_models = ['linear', 'power', 'gaussian', 'spherical', 'exponential']
        selected_model = variogram_models[variogram_model]

        # Estimate variogram parameters
        distances = pdist(np.column_stack((x, y)))
        max_distance = np.max(distances)
        mean_distance = np.mean(distances)
        variance = np.var(z)

        # Define variogram parameter presets
        variogram_presets = {
            0: None,  # Default (auto-estimate)
            1: {'range': mean_distance / 2, 'sill': variance * 0.75, 'nugget': variance * 0.25},  # Low range
            2: {'range': max_distance / 2, 'sill': variance * 0.75, 'nugget': variance * 0.25},  # High range
            3: {'range': mean_distance, 'sill': variance * 0.9, 'nugget': variance * 0.1},  # Low nugget
            4: {'range': mean_distance, 'sill': variance * 0.6, 'nugget': variance * 0.4}   # High nugget (adjusted)
        }
        variogram_params = variogram_presets[variogram_preset]

        # Adjust parameters for linear and power models
        if selected_model == 'linear':
            if variogram_params is None:
                variogram_params = {'slope': variance / mean_distance, 'nugget': variance * 0.1}
            else:
                variogram_params = {'slope': variogram_params.get('sill', variance) / variogram_params.get('range', mean_distance),
                                    'nugget': variogram_params.get('nugget', variance * 0.1)}
        elif selected_model == 'power':
            if variogram_params is None:
                variogram_params = {'scale': variance / (mean_distance ** 1.5), 'exponent': 1.5, 'nugget': variance * 0.1}
            else:
                variogram_params = {'scale': variogram_params.get('sill', variance) / (variogram_params.get('range', mean_distance) ** 1.5),
                                    'exponent': 1.5,
                                    'nugget': variogram_params.get('nugget', variance * 0.1)}

        QgsMessageLog.logMessage(f"Kriging parameters: model={selected_model}, params={variogram_params}", 'KrigingAnalysis', Qgis.Info)

        # Create grid based on cell size and extent
        cols = int((extent.xMaximum() - extent.xMinimum()) / cell_size)
        rows = int((extent.yMaximum() - extent.yMinimum()) / cell_size)
        grid_x = np.linspace(extent.xMinimum(), extent.xMaximum(), cols)
        grid_y = np.linspace(extent.yMinimum(), extent.yMaximum(), rows)

        QgsMessageLog.logMessage(f"Grid created: cols={cols}, rows={rows}", 'KrigingAnalysis', Qgis.Info)

        # Perform Kriging
        try:
            if kriging_method == 0:  # Ordinary Kriging
                k = OrdinaryKriging(x, y, z, variogram_model=selected_model, variogram_parameters=variogram_params)
                z, ss = k.execute('grid', grid_x, grid_y)
            else:  # Universal Kriging
                # Create drift functions for Universal Kriging
                def linear_drift(x, y):
                    return x + y

                def quadratic_drift(x, y):
                    return x**2 + y**2

                drift_functions = [linear_drift, quadratic_drift]
                k = UniversalKriging(x, y, z, variogram_model=selected_model, variogram_parameters=variogram_params,
                                     drift_terms=drift_functions)
                z, ss = k.execute('grid', grid_x, grid_y)
            
            QgsMessageLog.logMessage(f"Kriging ({['Ordinary', 'Universal'][kriging_method]}) executed successfully", 'KrigingAnalysis', Qgis.Info)
            QgsMessageLog.logMessage(f"Kriging output shape: z={z.shape}, ss={ss.shape}", 'KrigingAnalysis', Qgis.Info)
            QgsMessageLog.logMessage(f"Kriging output sample: z[0,0]={z[0,0]}, ss[0,0]={ss[0,0]}", 'KrigingAnalysis', Qgis.Info)

            # Check for low variation in results
            if np.std(z) < (np.max(z) - np.min(z)) * 0.01:
                feedback.pushWarning("The kriging result shows very low variation. Consider adjusting the variogram parameters or using the 'Default' preset.")

        except Exception as e:
            QgsMessageLog.logMessage(f"Kriging failed: {str(e)}", 'KrigingAnalysis', Qgis.Critical)
            QgsMessageLog.logMessage(f"Traceback: {traceback.format_exc()}", 'KrigingAnalysis', Qgis.Critical)
            raise QgsProcessingException(self.tr(f"Kriging failed: {str(e)}"))

        # Apply min and max constraints
        z = np.clip(z, min_value, max_value)

        # Create and save raster using GDAL directly
        driver = gdal.GetDriverByName('GTiff')
        if driver is None:
            raise QgsProcessingException(self.tr("Could not load GTiff driver"))

        try:
            QgsMessageLog.logMessage(f"Creating raster: {output_raster}", 'KrigingAnalysis', Qgis.Info)
            dataset = driver.Create(output_raster, cols, rows, 1, gdal.GDT_Float32)

            if dataset is None:
                raise QgsProcessingException(self.tr("Could not create raster dataset"))

            dataset.SetGeoTransform((extent.xMinimum(), cell_size, 0, extent.yMaximum(), 0, -cell_size))
            
            band = dataset.GetRasterBand(1)
            band.WriteArray(z)
            band.SetNoDataValue(-9999)
            
            # Set projection if available
            if source.sourceCrs().isValid():
                dataset.SetProjection(source.sourceCrs().toWkt())

            band.FlushCache()
            dataset = None  # Close the dataset

            QgsMessageLog.logMessage("Raster creation completed successfully", 'KrigingAnalysis', Qgis.Info)

        except Exception as e:
            QgsMessageLog.logMessage(f"Error creating raster: {str(e)}", 'KrigingAnalysis', Qgis.Critical)
            QgsMessageLog.logMessage(f"Traceback: {traceback.format_exc()}", 'KrigingAnalysis', Qgis.Critical)
            raise QgsProcessingException(self.tr(f"Error creating raster: {str(e)}"))

        # Verify the raster was created correctly
        if not os.path.exists(output_raster):
            raise QgsProcessingException(self.tr("Output raster file was not created"))

        raster_layer = QgsRasterLayer(output_raster, "Kriging result")
        if not raster_layer.isValid():
            QgsMessageLog.logMessage(f"Created raster is not valid. Error: {raster_layer.error().message()}", 'KrigingAnalysis', Qgis.Critical)
            raise QgsProcessingException(self.tr("Created raster is not valid"))
        else:
            QgsMessageLog.logMessage("Raster verified and is valid", 'KrigingAnalysis', Qgis.Info)

        return {self.OUTPUT_KRIGING: output_raster}

    def name(self):
        return 'advancedkriginganalysis'

    def displayName(self):
        return self.tr('Advanced Kriging Analysis')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'arcgeekcalculator'

    def shortHelpString(self):
        return self.tr("""
        This algorithm performs Kriging interpolation on point data.

        Input parameters:
        - Input point layer: The layer containing the sample points.
        - Z value field: The field containing the values to be interpolated.
        - Variogram model: The theoretical variogram model to use.
        - Variogram parameters preset: Choose a preset for variogram parameters.
        - Kriging method: Choose between Ordinary and Universal Kriging.
        - Cell size: The size of the cells in the output raster.
        - Interpolation extent: The extent of the output raster.
        - Minimum value (optional): The minimum allowed value in the output.
        - Maximum value (optional): The maximum allowed value in the output.

        The algorithm produces a raster output of the interpolated surface.

        Tip: Start with the 'Default' variogram parameters preset. If the results are not satisfactory, try other presets or adjust the cell size.
        """)

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return KrigingAnalysisAlgorithm()