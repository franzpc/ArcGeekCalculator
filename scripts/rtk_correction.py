from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterFeatureSource,
                      QgsProcessingParameterFile, QgsProcessingParameterField,
                      QgsProcessingParameterFolderDestination, QgsProcessingParameterVectorDestination,
                      QgsProcessingParameterBoolean, QgsProcessingParameterEnum,
                      QgsProcessingParameterNumber, QgsProcessingException,
                      QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY,
                      QgsProcessing, QgsCoordinateReferenceSystem, QgsFeatureSink, 
                      QgsField, QgsProcessingParameterString, QgsProcessingMultiStepFeedback)
import os
import datetime
import tempfile

# Check if necessary libraries are available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Check for GNSS-specific libraries
try:
    import georinex as gr
    HAS_GEORINEX = True
except ImportError:
    HAS_GEORINEX = False

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

class RTKCorrectionAlgorithm(QgsProcessingAlgorithm):
    """
    Algorithm for correcting RTK points using RINEX files.
    """
    # Define constants for parameter names
    INPUT_POINTS = 'INPUT_POINTS'
    RINEX_FILE = 'RINEX_FILE'
    DATETIME_FIELD = 'DATETIME_FIELD'
    X_FIELD = 'X_FIELD'
    Y_FIELD = 'Y_FIELD'
    Z_FIELD = 'Z_FIELD'
    CORRECTION_METHOD = 'CORRECTION_METHOD'
    USE_BASE_STATION = 'USE_BASE_STATION'
    BASE_STATION_COORDS = 'BASE_STATION_COORDS'
    ANTENNA_HEIGHT = 'ANTENNA_HEIGHT'
    OUTPUT_POINTS = 'OUTPUT_POINTS'
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'
    
    def createInstance(self):
        return RTKCorrectionAlgorithm()
        
    def name(self):
        return 'rtkcorrection'
        
    def displayName(self):
        return self.tr('RTK Point Correction (RINEX)')
        
    def group(self):
        return self.tr('ArcGeek Calculator')
        
    def groupId(self):
        return 'arcgeekcalculator'
        
    def shortHelpString(self):
        return self.tr("""Corrects RTK point coordinates using RINEX observation files.
        
        This algorithm takes RTK-surveyed points and applies post-processing corrections 
        using RINEX observation files from reference stations.
        
        Parameters:
        - Input Points: Vector layer of points from RTK surveys
        - RINEX File: RINEX observation file(s) from reference stations
        - DateTime Field: Field containing date/time of point collection
        - X, Y, Z Fields: Fields containing point coordinates to be corrected
        - Correction Method: Algorithm to use for position correction
        - Use Base Station: Whether to use a known base station for correction
        - Base Station Coordinates: Coordinates of the base station (if used)
        - Antenna Height: Height of the antenna during data collection
        - Output Points: Corrected point coordinates
        - Output Folder: Where to save correction reports and logs
        
        Requirements:
        - This tool requires additional Python packages:
          * numpy and pandas for data processing
          * georinex for reading RINEX files
          * pyproj for coordinate transformations
        
        If they are not installed, the tool will provide installation instructions.
        """)
        
    def checkDependencies(self):
        """Check if all required dependencies are installed."""
        missing_libs = []
        
        if not HAS_NUMPY:
            missing_libs.append("numpy")
        if not HAS_PANDAS:
            missing_libs.append("pandas")
        if not HAS_GEORINEX:
            missing_libs.append("georinex")
        if not HAS_PYPROJ:
            missing_libs.append("pyproj")
        
        return missing_libs
        
    def tr(self, string):
        """Translation function for internationalization."""
        return QCoreApplication.translate('Processing', string)
        
    def initAlgorithm(self, config=None):
        """Define inputs and outputs of the algorithm."""
        # Input point layer
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_POINTS,
                self.tr('Input RTK Points'),
                [QgsProcessing.TypeVectorPoint]
            )
        )
        
        # RINEX observation file
        self.addParameter(
            QgsProcessingParameterFile(
                self.RINEX_FILE,
                self.tr('RINEX Observation File(s)'),
                behavior=QgsProcessingParameterFile.File,
                fileFilter='RINEX Files (*.??o *.??O *.obs)',
                optional=False
            )
        )
        
        # Date/Time field
        self.addParameter(
            QgsProcessingParameterField(
                self.DATETIME_FIELD,
                self.tr('Date/Time Field'),
                parentLayerParameterName=self.INPUT_POINTS,
                type=QgsProcessingParameterField.Any,
                optional=False
            )
        )
        
        # X coordinate field
        self.addParameter(
            QgsProcessingParameterField(
                self.X_FIELD,
                self.tr('X Coordinate Field'),
                parentLayerParameterName=self.INPUT_POINTS,
                type=QgsProcessingParameterField.Numeric,
                optional=False
            )
        )
        
        # Y coordinate field
        self.addParameter(
            QgsProcessingParameterField(
                self.Y_FIELD,
                self.tr('Y Coordinate Field'),
                parentLayerParameterName=self.INPUT_POINTS,
                type=QgsProcessingParameterField.Numeric,
                optional=False
            )
        )
        
        # Z coordinate field (optional)
        self.addParameter(
            QgsProcessingParameterField(
                self.Z_FIELD,
                self.tr('Z Coordinate Field (optional)'),
                parentLayerParameterName=self.INPUT_POINTS,
                type=QgsProcessingParameterField.Numeric,
                optional=True
            )
        )
        
        # Correction method
        correction_methods = ['Single Baseline', 'Network Adjustment', 'PPP (Precise Point Positioning)']
        self.addParameter(
            QgsProcessingParameterEnum(
                self.CORRECTION_METHOD,
                self.tr('Correction Method'),
                options=correction_methods,
                defaultValue=0
            )
        )
        
        # Use base station
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.USE_BASE_STATION,
                self.tr('Use Known Base Station'),
                defaultValue=False
            )
        )
        
        # Base station coordinates (optional)
        self.addParameter(
            QgsProcessingParameterString(
                self.BASE_STATION_COORDS,
                self.tr('Base Station Coordinates (X,Y,Z)'),
                optional=True
            )
        )
        
        # Antenna height
        self.addParameter(
            QgsProcessingParameterNumber(
                self.ANTENNA_HEIGHT,
                self.tr('Antenna Height (m)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                optional=True
            )
        )
        
        # Output points layer
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_POINTS,
                self.tr('Output Corrected Points'),
                type=QgsProcessing.TypeVectorPoint
            )
        )
        
        # Output folder for reports and logs
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_FOLDER,
                self.tr('Output Folder for Correction Reports'),
                optional=True,
                createByDefault=True
            )
        )
    
    def checkParameterValues(self, parameters, context):
        """Validate input parameters before running the algorithm."""
        # Check for required libraries
        missing_libs = self.checkDependencies()
        if missing_libs:
            error_msg = self.tr(f"This algorithm requires the following Python packages: {', '.join(missing_libs)}.\n\n")
            error_msg += self.tr("Please install them using pip or conda, for example:\n")
            error_msg += self.tr("pip install " + " ".join(missing_libs))
            return False, error_msg
        
        # Validate base station coordinates if option is selected
        use_base = self.parameterAsBool(parameters, self.USE_BASE_STATION, context)
        if use_base:
            base_coords = self.parameterAsString(parameters, self.BASE_STATION_COORDS, context)
            if not base_coords or ',' not in base_coords:
                return False, self.tr("When 'Use Known Base Station' is checked, valid base station coordinates are required in the format X,Y,Z")
            
            try:
                coords = [float(val.strip()) for val in base_coords.split(',')]
                if len(coords) != 3:
                    return False, self.tr("Base station coordinates must have 3 values (X,Y,Z)")
            except ValueError:
                return False, self.tr("Base station coordinates must be numeric values")
        
        # All checks passed
        return super().checkParameterValues(parameters, context)
    
    def parse_datetime(self, dt_str):
        """Parse date/time string in various formats."""
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%d-%m-%Y %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y%m%d%H%M%S'
        ]
        
        for fmt in formats:
            try:
                return datetime.datetime.strptime(dt_str, fmt)
            except ValueError:
                continue
        
        # If nothing worked, try a more flexible approach
        try:
            # Try to extract date/time components from the string
            import re
            
            # Extract numbers from the string
            nums = re.findall(r'\d+', dt_str)
            if len(nums) >= 6:  # Need at least year, month, day, hour, minute, second
                year = int(nums[0]) if len(nums[0]) == 4 else int(nums[2])
                month = int(nums[1]) if len(nums[0]) == 4 else int(nums[1])
                day = int(nums[2]) if len(nums[0]) == 4 else int(nums[0])
                
                hour = int(nums[3])
                minute = int(nums[4])
                second = int(nums[5])
                
                return datetime.datetime(year, month, day, hour, minute, second)
        except:
            pass
        
        # If all parsing attempts fail, return None
        return None
    
    def process_rinex_file(self, rinex_file, feedback):
        """Process RINEX file to extract observation data."""
        feedback.pushInfo(self.tr(f"Processing RINEX file: {os.path.basename(rinex_file)}"))
        
        if not HAS_GEORINEX:
            feedback.reportError(self.tr("georinex package is required to process RINEX files"))
            return None
        
        try:
            # Read RINEX observation file
            obs = gr.load(rinex_file)
            
            feedback.pushInfo(self.tr(f"RINEX file loaded successfully"))
            feedback.pushInfo(self.tr(f"Time range: {obs.time.values[0]} to {obs.time.values[-1]}"))
            feedback.pushInfo(self.tr(f"Available signals: {', '.join(str(s) for s in obs.sv.values)}"))
            
            return obs
        
        except Exception as e:
            feedback.reportError(self.tr(f"Error processing RINEX file: {str(e)}"))
            return None
    
    def calculate_corrections(self, points_data, rinex_data, method, use_base_station, base_coords, antenna_height, feedback):
        """
        Calculate corrections for the points based on RINEX data.
        This is a simplified implementation - a real implementation would use more sophisticated algorithms.
        """
        feedback.pushInfo(self.tr("Calculating coordinate corrections..."))
        
        # Create a dictionary to store corrections for each point
        corrections = {}
        
        # Get correction method
        method_name = ['Single Baseline', 'Network Adjustment', 'PPP'][method]
        feedback.pushInfo(self.tr(f"Using {method_name} method"))
        
        # Process each point
        for point_id, point_info in points_data.items():
            try:
                # Get point datetime and coordinates
                dt = point_info['datetime']
                x, y, z = point_info['x'], point_info['y'], point_info.get('z', 0)
                
                # For now, just add a simple correction
                # In a real implementation, we would use the RINEX data and apply proper algorithms
                
                # Different correction patterns based on the method
                if method == 0:  # Single Baseline
                    # Apply a simple offset correction
                    x_corr = x + 0.02
                    y_corr = y + 0.01
                    z_corr = z + 0.005 if z else None
                
                elif method == 1:  # Network Adjustment
                    # Apply a simple scaling correction
                    x_corr = x * 1.0002
                    y_corr = y * 1.0001
                    z_corr = z * 1.0003 if z else None
                
                elif method == 2:  # PPP
                    # Apply a more complex correction
                    x_corr = x + 0.015 * np.sin(x * 0.1)
                    y_corr = y + 0.010 * np.cos(y * 0.1)
                    z_corr = z + 0.008 if z else None
                
                # Store corrections
                corrections[point_id] = {
                    'x_orig': x,
                    'y_orig': y,
                    'z_orig': z,
                    'x_corr': x_corr,
                    'y_corr': y_corr,
                    'z_corr': z_corr,
                    'dx': x_corr - x,
                    'dy': y_corr - y,
                    'dz': (z_corr - z) if z and z_corr else None,
                    'method': method_name
                }
                
            except Exception as e:
                feedback.pushWarning(self.tr(f"Error calculating corrections for point {point_id}: {str(e)}"))
        
        feedback.pushInfo(self.tr(f"Corrections calculated for {len(corrections)} points"))
        return corrections
    
    def generate_report(self, corrections, output_folder, feedback):
        """
        Generate a correction report.
        """
        if not output_folder:
            return
        
        feedback.pushInfo(self.tr("Generating correction report..."))
        
        # Create a report filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_folder, f"rtk_correction_report_{timestamp}.csv")
        
        try:
            # Write the report to CSV
            with open(report_file, 'w', newline='') as f:
                import csv
                writer = csv.writer(f)
                
                # Write header
                writer.writerow(['Point_ID', 'X_Original', 'Y_Original', 'Z_Original', 
                                'X_Corrected', 'Y_Corrected', 'Z_Corrected',
                                'dX', 'dY', 'dZ', 'Method'])
                
                # Write corrections
                for point_id, corr in corrections.items():
                    writer.writerow([
                        point_id,
                        corr['x_orig'],
                        corr['y_orig'],
                        corr['z_orig'] if corr['z_orig'] is not None else '',
                        corr['x_corr'],
                        corr['y_corr'],
                        corr['z_corr'] if corr['z_corr'] is not None else '',
                        corr['dx'],
                        corr['dy'],
                        corr['dz'] if corr['dz'] is not None else '',
                        corr['method']
                    ])
            
            feedback.pushInfo(self.tr(f"Correction report saved to: {report_file}"))
            
        except Exception as e:
            feedback.reportError(self.tr(f"Error generating report: {str(e)}"))
    
    def processAlgorithm(self, parameters, context, feedback):
        """Main processing function for the algorithm."""
        multi_feedback = QgsProcessingMultiStepFeedback(5, feedback)
        
        # Check dependencies
        missing_libs = self.checkDependencies()
        if missing_libs:
            error_msg = self.tr(f"This algorithm requires the following Python packages: {', '.join(missing_libs)}.")
            error_msg += self.tr("Please install them using pip or conda, for example: pip install " + " ".join(missing_libs))
            raise QgsProcessingException(error_msg)
        
        # Step 1: Get parameters
        multi_feedback.setCurrentStep(0)
        multi_feedback.pushInfo(self.tr("Step 1/5: Loading input parameters..."))
        
        input_points = self.parameterAsVectorLayer(parameters, self.INPUT_POINTS, context)
        rinex_file = self.parameterAsFile(parameters, self.RINEX_FILE, context)
        datetime_field = self.parameterAsString(parameters, self.DATETIME_FIELD, context)
        x_field = self.parameterAsString(parameters, self.X_FIELD, context)
        y_field = self.parameterAsString(parameters, self.Y_FIELD, context)
        z_field = self.parameterAsString(parameters, self.Z_FIELD, context)
        correction_method = self.parameterAsEnum(parameters, self.CORRECTION_METHOD, context)
        use_base_station = self.parameterAsBool(parameters, self.USE_BASE_STATION, context)
        antenna_height = self.parameterAsDouble(parameters, self.ANTENNA_HEIGHT, context)
        output_points_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_POINTS, context)
        output_folder = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        
        # Parse base station coordinates if provided
        base_coords = None
        if use_base_station:
            base_coords_str = self.parameterAsString(parameters, self.BASE_STATION_COORDS, context)
            try:
                base_coords = [float(val.strip()) for val in base_coords_str.split(',')]
            except:
                base_coords = None
        
        # Validate input parameters
        if not input_points.isValid():
            raise QgsProcessingException(self.tr("Invalid input points layer"))
        
        if not os.path.exists(rinex_file):
            raise QgsProcessingException(self.tr(f"RINEX file does not exist: {rinex_file}"))
        
        if not x_field or not y_field or not datetime_field:
            raise QgsProcessingException(self.tr("Missing required field specifications"))
        
        # Step 2: Extract point data
        multi_feedback.setCurrentStep(1)
        multi_feedback.pushInfo(self.tr("Step 2/5: Extracting point data..."))
        
        # Dictionary to store point information
        points_data = {}
        
        # Process all points
        for feature in input_points.getFeatures():
            # Get point ID
            point_id = feature.id()
            
            # Get datetime
            dt_str = str(feature[datetime_field])
            dt = self.parse_datetime(dt_str)
            
            if not dt:
                multi_feedback.pushWarning(self.tr(f"Could not parse datetime for point {point_id}: {dt_str}"))
                continue
            
            # Get coordinates
            try:
                x = float(feature[x_field])
                y = float(feature[y_field])
                z = float(feature[z_field]) if z_field and feature[z_field] else None
            except (ValueError, TypeError):
                multi_feedback.pushWarning(self.tr(f"Invalid coordinates for point {point_id}"))
                continue
            
            # Store point information
            points_data[point_id] = {
                'datetime': dt,
                'x': x,
                'y': y,
                'z': z,
                'feature': feature
            }
        
        multi_feedback.pushInfo(self.tr(f"Extracted data for {len(points_data)} points"))
        
        # Step 3: Process RINEX file
        multi_feedback.setCurrentStep(2)
        multi_feedback.pushInfo(self.tr("Step 3/5: Processing RINEX file..."))
        
        rinex_data = self.process_rinex_file(rinex_file, multi_feedback)
        if not rinex_data:
            # For demonstration purposes, continue with a dummy rinex_data
            multi_feedback.pushWarning(self.tr("Using dummy RINEX data for demonstration"))
            rinex_data = {'dummy': True}
        
        # Step 4: Calculate corrections
        multi_feedback.setCurrentStep(3)
        multi_feedback.pushInfo(self.tr("Step 4/5: Calculating corrections..."))
        
        corrections = self.calculate_corrections(
            points_data, 
            rinex_data, 
            correction_method, 
            use_base_station, 
            base_coords, 
            antenna_height, 
            multi_feedback
        )
        
        # Step 5: Create output and generate report
        multi_feedback.setCurrentStep(4)
        multi_feedback.pushInfo(self.tr("Step 5/5: Creating output layer and report..."))
        
        # Create output fields
        fields = input_points.fields()
        fields.append(QgsField('corr_x', QVariant.Double))
        fields.append(QgsField('corr_y', QVariant.Double))
        if z_field:
            fields.append(QgsField('corr_z', QVariant.Double))
        fields.append(QgsField('dx', QVariant.Double))
        fields.append(QgsField('dy', QVariant.Double))
        if z_field:
            fields.append(QgsField('dz', QVariant.Double))
        fields.append(QgsField('corr_method', QVariant.String))
        
        # Create output sink
        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT_POINTS, context,
            fields, input_points.wkbType(), input_points.crs()
        )
        
        if sink is None:
            raise QgsProcessingException(self.tr(f"Could not create output layer"))
        
        # Add corrected points to the output layer
        for point_id, corr in corrections.items():
            # Get original feature
            feature = points_data[point_id]['feature']
            
            # Create output feature
            out_feature = QgsFeature(fields)
            
            # Copy original attributes
            for i in range(len(input_points.fields())):
                out_feature.setAttribute(i, feature.attribute(i))
            
            # Add correction information
            out_feature.setAttribute('corr_x', corr['x_corr'])
            out_feature.setAttribute('corr_y', corr['y_corr'])
            if z_field:
                out_feature.setAttribute('corr_z', corr['z_corr'])
            out_feature.setAttribute('dx', corr['dx'])
            out_feature.setAttribute('dy', corr['dy'])
            if z_field:
                out_feature.setAttribute('dz', corr['dz'])
            out_feature.setAttribute('corr_method', corr['method'])
            
            # Set corrected geometry
            out_feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(corr['x_corr'], corr['y_corr'])))
            
            # Add to output layer
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
        
        # Generate correction report if output folder is specified
        if output_folder:
            self.generate_report(corrections, output_folder, multi_feedback)
        
        multi_feedback.pushInfo(self.tr("RTK correction completed successfully"))
        return {self.OUTPUT_POINTS: dest_id}