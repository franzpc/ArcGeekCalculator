from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterField, QgsProcessingParameterRasterDestination,
                       QgsProcessingParameterNumber, QgsProcessingParameterEnum,
                       QgsProcessingParameterExtent, QgsProcessingParameterBoolean,
                       QgsProcessingOutputRasterLayer, QgsVectorLayer, QgsRasterLayer,
                       QgsFeature, QgsField, QgsProcessingException, QgsMessageLog,
                       QgsRectangle, QgsPointXY, Qgis, QgsCoordinateReferenceSystem,
                       QgsProcessing)
import os
import warnings
from tempfile import gettempdir
import math
import processing
import tempfile

# Check if numpy is installed
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Check if pykrige is installed
try:
    from pykrige.ok import OrdinaryKriging
    from pykrige.uk import UniversalKriging
    HAS_PYKRIGE = True
except ImportError:
    HAS_PYKRIGE = False

# Check if scipy is installed (required for pykrige)
try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class KrigingAnalysisAlgorithm(QgsProcessingAlgorithm):
    # Define constants for parameter names
    INPUT_LAYER = 'INPUT_LAYER'
    INPUT_FIELD = 'INPUT_FIELD'
    OUTPUT_CELLSIZE = 'OUTPUT_CELLSIZE'
    OUTPUT_EXTENT = 'OUTPUT_EXTENT'
    VARIOGRAM_MODEL = 'VARIOGRAM_MODEL'
    KRIGING_TYPE = 'KRIGING_TYPE'
    DRIFT_TYPE = 'DRIFT_TYPE'
    INCLUDE_ERROR = 'INCLUDE_ERROR'
    OUTPUT_RASTER = 'OUTPUT_RASTER'
    OUTPUT_ERROR = 'OUTPUT_ERROR'
    MIN_VALUE = 'MIN_VALUE'
    MAX_VALUE = 'MAX_VALUE'

    def createInstance(self):
        return KrigingAnalysisAlgorithm()

    def name(self):
        return 'kriginganalysis'

    def displayName(self):
        return self.tr('Kriging Analysis')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'arcgeekcalculator'

    def shortHelpString(self):
        return self.tr("""Performs kriging spatial interpolation on point data.
        
        Kriging is a geostatistical technique that interpolates the value of a field at an unobserved location from observations at nearby locations, weighted according to spatial covariance values.
        
        Parameters:
        - Input point layer: Vector layer containing points with values to interpolate.
        - Value field: Field containing the numeric values to interpolate.
        - Output cell size: Size of the pixels in the output raster (in layer units).
        - Output extent: Extent of the output raster layer.
        - Variogram model: Mathematical function that describes spatial correlation in the data.
        - Kriging type: Ordinary or Universal kriging. Universal kriging includes drift components.
        - Drift type: Drift components to include in Universal kriging.
        - Include error estimation: Generate an additional raster showing kriging variance.
        - Min value (optional): Minimum value allowed in output (leave empty for no limit).
        - Max value (optional): Maximum value allowed in output (leave empty for no limit).
        
        Outputs:
        - Output interpolated raster: Result of kriging interpolation in GeoTIFF format.
        - Output kriging variance (optional): Estimation error/uncertainty in GeoTIFF format.
        
        Requirements:
        - This tool requires the numpy, pykrige and scipy Python packages.
        - If they are not installed, the tool will provide installation instructions.
        """)

    def tr(self, string):
        """Translation function for the algorithm."""
        return QCoreApplication.translate('Processing', string)

    def checkDependencies(self):
        """Check if all required dependencies are installed."""
        missing_libs = []
        
        if not HAS_NUMPY:
            missing_libs.append("numpy")
        if not HAS_SCIPY:
            missing_libs.append("scipy")
        if not HAS_PYKRIGE:
            missing_libs.append("pykrige")
        
        return missing_libs

    def initAlgorithm(self, config=None):
        """Define the inputs and outputs of the algorithm."""
        # Input point layer
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_LAYER,
                self.tr('Input point layer'),
                [QgsProcessing.TypeVectorPoint]
            )
        )
        
        # Value field to interpolate
        self.addParameter(
            QgsProcessingParameterField(
                self.INPUT_FIELD,
                self.tr('Value field to interpolate'),
                parentLayerParameterName=self.INPUT_LAYER,
                type=QgsProcessingParameterField.Numeric
            )
        )
        
        # Cell size
        self.addParameter(
            QgsProcessingParameterNumber(
                self.OUTPUT_CELLSIZE,
                self.tr('Output cell size'),
                QgsProcessingParameterNumber.Double,
                defaultValue=10.0,
                minValue=0.01
            )
        )
        
        # Output extent
        self.addParameter(
            QgsProcessingParameterExtent(
                self.OUTPUT_EXTENT,
                self.tr('Output extent (xmin,xmax,ymin,ymax)'),
                optional=True
            )
        )
        
        # Variogram model
        variogram_models = ['linear', 'power', 'gaussian', 'spherical', 'exponential', 'hole-effect']
        self.addParameter(
            QgsProcessingParameterEnum(
                self.VARIOGRAM_MODEL,
                self.tr('Variogram model'),
                options=variogram_models,
                defaultValue=3  # spherical
            )
        )
        
        # Kriging type
        kriging_types = ['Ordinary Kriging', 'Universal Kriging']
        self.addParameter(
            QgsProcessingParameterEnum(
                self.KRIGING_TYPE,
                self.tr('Kriging type'),
                options=kriging_types,
                defaultValue=0  # Ordinary Kriging
            )
        )
        
        # Drift type (only for Universal Kriging)
        drift_types = ['none', 'regional linear', 'regional quadratic', 'point logarithmic', 'external-z', 'specified']
        self.addParameter(
            QgsProcessingParameterEnum(
                self.DRIFT_TYPE,
                self.tr('Drift type (only for Universal Kriging)'),
                options=drift_types,
                defaultValue=1,  # regional linear
                optional=True
            )
        )
        
        # Include error estimation
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.INCLUDE_ERROR,
                self.tr('Include error estimation'),
                defaultValue=False
            )
        )
        
        # Min value (optional)
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_VALUE,
                self.tr('Minimum value (optional)'),
                QgsProcessingParameterNumber.Double,
                optional=True
            )
        )
        
        # Max value (optional)
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_VALUE,
                self.tr('Maximum value (optional)'),
                QgsProcessingParameterNumber.Double,
                optional=True
            )
        )
        
        # Output raster
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_RASTER,
                self.tr('Output interpolated raster'),
                defaultValue='TEMPORARY_OUTPUT'
            )
        )
        
        # Output error raster (optional)
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_ERROR,
                self.tr('Output kriging variance'),
                optional=True,
                createByDefault=False,
                defaultValue='TEMPORARY_OUTPUT'  # Default to temporary output
            )
        )

    def checkParameterValues(self, parameters, context):
        """This function is called before the algorithm runs to validate parameters."""
        # Check for required libraries before processing
        missing_libs = self.checkDependencies()
        
        if missing_libs:
            error_msg = self.tr(f"This algorithm requires the following Python packages: {', '.join(missing_libs)}.\n\n")
            error_msg += self.tr("Please install them using pip or conda, for example:\n")
            error_msg += self.tr("pip install " + " ".join(missing_libs))
            
            return False, error_msg
        
        # Validate min/max values if both are provided
        min_value_param = parameters.get(self.MIN_VALUE)
        max_value_param = parameters.get(self.MAX_VALUE)
        
        min_value_set = min_value_param is not None and min_value_param != ''
        max_value_set = max_value_param is not None and max_value_param != ''
        
        if min_value_set and max_value_set:
            min_value = self.parameterAsDouble(parameters, self.MIN_VALUE, context)
            max_value = self.parameterAsDouble(parameters, self.MAX_VALUE, context)
            
            if min_value >= max_value:
                return False, self.tr("Maximum value must be greater than minimum value.")
        
        # All checks passed
        return super().checkParameterValues(parameters, context)

    def write_ascii_grid(self, filename, data, xmin, ymax, cell_size, nodata_value=-9999, crs=None):
        """
        Writes an ASCII Grid (.asc) file with proper georeferencing
        
        Parameters:
        - filename: Output file name
        - data: 2D array with data
        - xmin: Minimum X coordinate (left edge)
        - ymax: Maximum Y coordinate (top edge)
        - cell_size: Cell size
        - nodata_value: Value for no data
        - crs: Coordinate reference system
        """
        # Get dimensions
        nrows, ncols = data.shape
        
        with open(filename, 'w') as f:
            # Write header with proper georeferencing information
            f.write(f"ncols {ncols}\n")
            f.write(f"nrows {nrows}\n")
            f.write(f"xllcorner {xmin}\n")
            f.write(f"yllcorner {ymax - nrows * cell_size}\n")
            f.write(f"cellsize {cell_size}\n")
            f.write(f"NODATA_value {nodata_value}\n")
            
            # Write data - ASCII Grid starts from top row, so we need to flip the data
            flipped_data = np.flipud(data)
            for row in range(nrows):
                line = ' '.join([f"{flipped_data[row, col]:.6f}" if not np.isnan(flipped_data[row, col]) 
                               and flipped_data[row, col] != nodata_value else str(nodata_value) 
                               for col in range(ncols)])
                f.write(line + '\n')

        # Create a companion .prj file for CRS information
        if crs and crs.isValid():
            prj_filename = os.path.splitext(filename)[0] + '.prj'
            with open(prj_filename, 'w') as prj_file:
                prj_file.write(crs.toWkt())

    def processAlgorithm(self, parameters, context, feedback):
        """Run the algorithm."""
        # Check for required dependencies
        missing_libs = self.checkDependencies()
        if missing_libs:
            error_msg = self.tr(f"This algorithm requires the following Python packages: {', '.join(missing_libs)}.\n\n")
            error_msg += self.tr("Please install them using pip or conda, for example:\n")
            error_msg += self.tr("pip install " + " ".join(missing_libs))
            
            raise QgsProcessingException(error_msg)

        # Load parameters
        source = self.parameterAsSource(parameters, self.INPUT_LAYER, context)
        field_name = self.parameterAsString(parameters, self.INPUT_FIELD, context)
        cell_size = self.parameterAsDouble(parameters, self.OUTPUT_CELLSIZE, context)
        extent = self.parameterAsExtent(parameters, self.OUTPUT_EXTENT, context, source.sourceCrs())
        variogram_model_idx = self.parameterAsEnum(parameters, self.VARIOGRAM_MODEL, context)
        kriging_type_idx = self.parameterAsEnum(parameters, self.KRIGING_TYPE, context)
        drift_type_idx = self.parameterAsEnum(parameters, self.DRIFT_TYPE, context)
        include_error = self.parameterAsBool(parameters, self.INCLUDE_ERROR, context)
        
        # Check min/max value parameters - directly use them if provided
        min_value_param = parameters.get(self.MIN_VALUE)
        max_value_param = parameters.get(self.MAX_VALUE)
        
        min_value = None
        max_value = None
        
        if min_value_param is not None and min_value_param != '':
            min_value = self.parameterAsDouble(parameters, self.MIN_VALUE, context)
            feedback.pushInfo(self.tr(f'Minimum value limit will be applied: {min_value}'))
        
        if max_value_param is not None and max_value_param != '':
            max_value = self.parameterAsDouble(parameters, self.MAX_VALUE, context)
            feedback.pushInfo(self.tr(f'Maximum value limit will be applied: {max_value}'))
        
        output_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_RASTER, context)
        
        # Handle the output error parameter
        output_error = None
        if include_error:
            output_error = self.parameterAsOutputLayer(parameters, self.OUTPUT_ERROR, context)
            feedback.pushInfo(self.tr(f'Error estimation will be saved to: {output_error}'))

        # Create temporary paths for ASCII Grid output
        temp_dir = tempfile.mkdtemp(prefix='kriging_')
        temp_output_asc = os.path.join(temp_dir, f'kriging_output_{os.getpid()}.asc')
        temp_error_asc = None
        if include_error:
            temp_error_asc = os.path.join(temp_dir, f'kriging_variance_{os.getpid()}.asc')

        # Configure processing based on parameters
        variogram_models = ['linear', 'power', 'gaussian', 'spherical', 'exponential', 'hole-effect']
        variogram_model = variogram_models[variogram_model_idx]
        
        is_universal = (kriging_type_idx == 1)
        
        drift_types = ['none', 'regional_linear', 'regional_quadratic', 'point_logarithmic', 'external_Z', 'specified']
        drift_type = drift_types[drift_type_idx] if is_universal else None

        # Get CRS from source
        crs = source.sourceCrs()
        feedback.pushInfo(self.tr(f'Using CRS: {crs.authid()}'))
        
        # Use input layer extent if not specified
        if extent.isNull():
            extent = source.sourceExtent()
            feedback.pushInfo(self.tr('Using input layer extent'))
            
            # Expand extent slightly to ensure all points are included
            buffer_size = cell_size * 3
            extent.grow(buffer_size)
        
        # Validate extent and create output grid
        if extent.width() <= 0 or extent.height() <= 0:
            raise QgsProcessingException(self.tr('Invalid extent. Width and height must be greater than 0.'))
        
        # Create grid definition
        grid_width = int(math.ceil(extent.width() / cell_size))
        grid_height = int(math.ceil(extent.height() / cell_size))
        
        if grid_width <= 0 or grid_height <= 0:
            raise QgsProcessingException(self.tr('Invalid grid dimensions. Please check cell size and extent.'))
        
        feedback.pushInfo(self.tr(f'Output grid: {grid_width} x {grid_height} cells, cell size: {cell_size}'))
        feedback.pushInfo(self.tr(f'Extent: ({extent.xMinimum()}, {extent.yMinimum()}) - ({extent.xMaximum()}, {extent.yMaximum()})'))
        
        # Extract data from source
        x_coords = []
        y_coords = []
        values = []
        
        total_features = source.featureCount()
        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
                
            # Get point coordinates
            point = feature.geometry().asPoint()
            val = feature[field_name]
            
            # Skip if value is None or not valid
            if val is None or not isinstance(val, (int, float)) or (isinstance(val, float) and math.isnan(val)):
                continue
                
            x_coords.append(point.x())
            y_coords.append(point.y())
            values.append(float(val))
            
            feedback.setProgress(int(current * 20 / total_features))  # First 20% for data loading
        
        # Check if we have enough points
        if len(values) < 3:
            raise QgsProcessingException(self.tr('Not enough valid points for kriging interpolation (minimum 3 required).'))
        
        feedback.pushInfo(self.tr(f'Using {len(values)} points for interpolation'))
        feedback.pushInfo(self.tr(f'Value range: {min(values)} to {max(values)}'))
        
        # Suppress warnings from PyKrige
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            feedback.pushInfo(self.tr(f'Starting kriging with {variogram_model} variogram model...'))
            
            # Create grid using linspace for more precise control
            x_min = extent.xMinimum()
            y_min = extent.yMinimum()
            x_max = extent.xMaximum()
            y_max = extent.yMaximum()
            
            # Using linspace ensures equally spaced grid points
            grid_x = np.linspace(x_min, x_max, grid_width)
            grid_y = np.linspace(y_min, y_max, grid_height)
            
            # Perform kriging based on type
            if is_universal:
                feedback.pushInfo(self.tr(f'Using Universal Kriging with drift type: {drift_type}'))
                
                # Convert drift_type to the format expected by UniversalKriging
                if drift_type == 'none':
                    uk_drift_terms = []
                elif drift_type == 'regional_linear':
                    uk_drift_terms = ['regional_linear']
                elif drift_type == 'regional_quadratic':
                    uk_drift_terms = ['regional_linear', 'regional_quadratic']
                else:
                    # Default to regional_linear if not recognized
                    uk_drift_terms = ['regional_linear']
                    
                # Create Universal Kriging model
                krig = UniversalKriging(
                    np.array(x_coords),
                    np.array(y_coords),
                    np.array(values),
                    variogram_model=variogram_model,
                    drift_terms=uk_drift_terms,
                    exact_values=True  # Force exact interpolation at data points
                )
            else:
                feedback.pushInfo(self.tr('Using Ordinary Kriging'))
                
                # Create Ordinary Kriging model
                krig = OrdinaryKriging(
                    np.array(x_coords),
                    np.array(y_coords),
                    np.array(values),
                    variogram_model=variogram_model,
                    exact_values=True  # Force exact interpolation at data points
                )
            
            # Perform kriging interpolation
            feedback.setProgress(30)  # 30% progress after setup
            feedback.pushInfo(self.tr('Executing kriging interpolation...'))
            z, ss = krig.execute('grid', grid_x, grid_y)
            feedback.setProgress(60)  # 60% progress after calculation
            
            # Apply min/max limits if provided
            apply_limits = min_value is not None or max_value is not None
            
            if apply_limits:
                if min_value is not None and max_value is not None:
                    # Apply both min and max limits
                    feedback.pushInfo(f"Applying limits: min={min_value}, max={max_value}")
                    # Count values to see what's happening
                    below_min = np.sum(z < min_value)
                    above_max = np.sum(z > max_value)
                    feedback.pushInfo(f"Values below min: {below_min}")
                    feedback.pushInfo(f"Values above max: {above_max}")
                    
                    # Now apply the limits
                    z = np.clip(z, min_value, max_value)
                    
                    # Verify the limits were applied
                    new_below_min = np.sum(z < min_value)
                    new_above_max = np.sum(z > max_value)
                    feedback.pushInfo(f"After clipping - Values below min: {new_below_min}")
                    feedback.pushInfo(f"After clipping - Values above max: {new_above_max}")
                    
                elif min_value is not None:
                    # Apply just min limit
                    feedback.pushInfo(f"Applying minimum limit: {min_value}")
                    # Count values to see what's happening
                    below_min = np.sum(z < min_value)
                    feedback.pushInfo(f"Values below min: {below_min}")
                    
                    # Now apply the limit
                    z = np.maximum(z, min_value)
                    
                    # Verify the limit was applied
                    new_below_min = np.sum(z < min_value)
                    feedback.pushInfo(f"After applying minimum - Values below min: {new_below_min}")
                    
                elif max_value is not None:
                    # Apply just max limit
                    feedback.pushInfo(f"Applying maximum limit: {max_value}")
                    # Count values to see what's happening
                    above_max = np.sum(z > max_value)
                    feedback.pushInfo(f"Values above max: {above_max}")
                    
                    # Now apply the limit
                    z = np.minimum(z, max_value)
                    
                    # Verify the limit was applied
                    new_above_max = np.sum(z > max_value)
                    feedback.pushInfo(f"After applying maximum - Values above max: {new_above_max}")
            
            # Handle NaN values
            z = np.nan_to_num(z, nan=-9999)
            ss = np.nan_to_num(ss, nan=-9999)
            
            # Write intermediate ASCII Grid output
            feedback.pushInfo(self.tr('Writing intermediate ASCII Grid...'))
            self.write_ascii_grid(temp_output_asc, z, x_min, y_max, cell_size, nodata_value=-9999, crs=crs)
            
            # Convert ASCII to target format
            feedback.pushInfo(self.tr('Converting to output raster format...'))
            
            # Use gdal:translate to ensure proper output format
            try:
                gdal_params = {
                    'INPUT': temp_output_asc,
                    'TARGET_CRS': crs,
                    'NODATA': -9999,
                    'COPY_SUBDATASETS': False,
                    'OPTIONS': 'COMPRESS=LZW',
                    'DATA_TYPE': 6,  # Float32
                    'OUTPUT': output_raster
                }
                processing.run("gdal:translate", gdal_params, context=context, feedback=feedback)
                feedback.pushInfo(f"Main raster output saved to: {output_raster}")
            except Exception as e:
                feedback.reportError(f"Error converting main raster: {str(e)}")
                # If conversion fails, just copy the ASC file to the output path
                if not output_raster.lower().endswith('.asc'):
                    output_raster = os.path.splitext(output_raster)[0] + '.asc'
                # Copy the ASCII file to the output path
                with open(temp_output_asc, 'r') as src_file, open(output_raster, 'w') as dst_file:
                    dst_file.write(src_file.read())
                # Also copy the PRJ file if it exists
                prj_file = os.path.splitext(temp_output_asc)[0] + '.prj'
                if os.path.exists(prj_file):
                    out_prj = os.path.splitext(output_raster)[0] + '.prj'
                    with open(prj_file, 'r') as src_file, open(out_prj, 'w') as dst_file:
                        dst_file.write(src_file.read())
            
            # Handle error estimation if requested
            if include_error and temp_error_asc and output_error:
                feedback.pushInfo(self.tr('Processing error estimation raster...'))
                self.write_ascii_grid(temp_error_asc, ss, x_min, y_max, cell_size, nodata_value=-9999, crs=crs)
                
                # Convert error ASCII to target format
                try:
                    error_gdal_params = {
                        'INPUT': temp_error_asc,
                        'TARGET_CRS': crs,
                        'NODATA': -9999,
                        'COPY_SUBDATASETS': False,
                        'OPTIONS': 'COMPRESS=LZW',
                        'DATA_TYPE': 6,  # Float32
                        'OUTPUT': output_error
                    }
                    processing.run("gdal:translate", error_gdal_params, context=context, feedback=feedback)
                    feedback.pushInfo(f"Error raster output saved to: {output_error}")
                except Exception as e:
                    feedback.reportError(f"Error converting variance raster: {str(e)}")
                    # If conversion fails, just copy the ASC file to the output path
                    if not output_error.lower().endswith('.asc'):
                        output_error = os.path.splitext(output_error)[0] + '.asc'
                    # Copy the ASCII file to the output path
                    with open(temp_error_asc, 'r') as src_file, open(output_error, 'w') as dst_file:
                        dst_file.write(src_file.read())
            
            feedback.setProgress(90)  # 90% progress after conversion
        
        # Clean up temporary files
        try:
            # Use a more reliable way to clean up temporary files
            import shutil
            # First, close any file handles
            import gc
            gc.collect()
            # Now try to remove the entire temp directory
            try:
                shutil.rmtree(temp_dir)
                feedback.pushInfo(f"Successfully cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                feedback.pushInfo(f"Warning: Could not clean up temporary directory: {str(e)}")
        except Exception as e:
            feedback.pushInfo(f"Warning: Error during cleanup: {str(e)}")
        
        feedback.setProgress(100)
        
        # Return results
        results = {self.OUTPUT_RASTER: output_raster}
        if include_error and output_error:
            results[self.OUTPUT_ERROR] = output_error
            
        return results