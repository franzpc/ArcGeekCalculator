from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterNumber, QgsProcessingParameterEnum,
                       QgsProcessingParameterString, QgsVectorLayer, QgsFeature,
                       QgsProcessingParameterFeatureSink, QgsGeometry, QgsPointXY,
                       QgsProcessingOutputNumber, QgsWkbTypes, QgsFeatureSink,
                       QgsField, QgsFields, QgsProcessingUtils, QgsProcessingException,
                       QgsProcessing)
import math
import re

class TreePlantingPatternAlgorithm(QgsProcessingAlgorithm):
    """
    This algorithm generates tree planting points with different patterns inside polygon features.
    """
    
    # Define constants for parameter names
    INPUT = 'INPUT'
    PATTERN_TYPE = 'PATTERN_TYPE'
    BORDER_MARGIN = 'BORDER_MARGIN'
    SPACING = 'SPACING'
    OUTPUT = 'OUTPUT'
    OUTPUT_TRIANGULAR_COUNT = 'OUTPUT_TRIANGULAR_COUNT'
    OUTPUT_RECTANGULAR_COUNT = 'OUTPUT_RECTANGULAR_COUNT'
    OUTPUT_CINCOOROS_COUNT = 'OUTPUT_CINCOOROS_COUNT'

    def initAlgorithm(self, config=None):
        """
        Define the inputs and outputs of the algorithm.
        """
        # Input polygon layer
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT,
                self.tr('Input polygon layer'),
                [QgsProcessing.TypeVectorPolygon]
            )
        )
        
        # Define pattern types as enum parameter
        self.addParameter(
            QgsProcessingParameterEnum(
                self.PATTERN_TYPE,
                self.tr('Planting pattern'),
                options=['Triangular/Quincunx (Tresbolillo)', 'Rectangular/Square', 'Five of Diamonds (Cinco de Oros)'],
                defaultValue=0
            )
        )
        
        # Border margin
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BORDER_MARGIN,
                self.tr('Border margin (distance from boundary)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.5,
                minValue=0.0
            )
        )
        
        # Spacing parameter as string (format: 3x3, 4.5x3, etc.)
        self.addParameter(
            QgsProcessingParameterString(
                self.SPACING,
                self.tr('Spacing (format: XxY - e.g. 3x3, 4.5x3)'),
                defaultValue='3x3'
            )
        )
        
        # Output points layer
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr('Output planting points'),
                QgsProcessing.TypeVectorPoint
            )
        )
        
        # Add outputs for counts
        self.addOutput(
            QgsProcessingOutputNumber(
                self.OUTPUT_TRIANGULAR_COUNT,
                self.tr('Triangular/Quincunx planting count')
            )
        )
        
        self.addOutput(
            QgsProcessingOutputNumber(
                self.OUTPUT_RECTANGULAR_COUNT,
                self.tr('Rectangular/Square planting count')
            )
        )
        
        self.addOutput(
            QgsProcessingOutputNumber(
                self.OUTPUT_CINCOOROS_COUNT,
                self.tr('Five of Diamonds planting count')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        """
        Process the algorithm.
        """
        # Get the input parameters
        source = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        pattern_type = self.parameterAsEnum(parameters, self.PATTERN_TYPE, context)
        border_margin = self.parameterAsDouble(parameters, self.BORDER_MARGIN, context)
        spacing_str = self.parameterAsString(parameters, self.SPACING, context)
        
        # Parse spacing string (format: XxY)
        try:
            # First, try to match the format XxY (e.g., 3x3, 4.5x3)
            match = re.match(r'(\d+(\.\d+)?)x(\d+(\.\d+)?)', spacing_str)
            if match:
                spacing_x = float(match.group(1))
                spacing_y = float(match.group(3))
            else:
                # If not in the format XxY, try to parse as a single number
                spacing_x = float(spacing_str)
                spacing_y = spacing_x
                
            # Validate spacing values
            if spacing_x <= 0 or spacing_y <= 0:
                raise ValueError("Spacing values must be positive")
                
        except (ValueError, TypeError) as e:
            raise QgsProcessingException(
                self.tr(f"Invalid spacing format. Please use format like '3x3' or '4.5x3'. Error: {str(e)}")
            )
        
        # If spacing_x equals spacing_y for rectangular/square pattern, it's a square pattern
        is_square = (pattern_type == 1 and spacing_x == spacing_y)
        
        # Provide feedback on the spacing being used
        if pattern_type == 0:  # Triangular/Quincunx
            feedback.pushInfo(self.tr(f'Using spacing {spacing_x}x{spacing_x} for Triangular/Quincunx pattern (only X value is used)'))
            # Force equal spacing for triangular
            spacing_y = spacing_x
        elif pattern_type == 1:  # Rectangular/Square
            if is_square:
                feedback.pushInfo(self.tr(f'Using equal spacing {spacing_x}x{spacing_y} for Square pattern'))
            else:
                feedback.pushInfo(self.tr(f'Using spacing {spacing_x}x{spacing_y} for Rectangular pattern'))
        elif pattern_type == 2:  # Five of Diamonds
            if spacing_x != spacing_y:
                feedback.pushInfo(self.tr(f'Using spacing {spacing_x}x{spacing_y} for Five of Diamonds pattern'))
            else:
                feedback.pushInfo(self.tr(f'Using spacing {spacing_x}x{spacing_x} for Five of Diamonds pattern'))
        
        # Prepare fields for output
        fields = QgsFields()
        fields.append(QgsField('pattern', QVariant.String))
        fields.append(QgsField('spacing_x', QVariant.Double))
        fields.append(QgsField('spacing_y', QVariant.Double))
        fields.append(QgsField('point_type', QVariant.String))
        
        # Create the feature sink
        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT, context,
            fields, QgsWkbTypes.Point, source.crs()
        )
        
        if sink is None:
            raise QgsProcessingException(self.tr('Failed to create output layer'))
        
        # Initialize counters for all pattern types
        triangular_count = 0
        rectangular_count = 0
        cincooros_count = 0
        
        # Process each polygon in the source layer
        total = 100.0 / source.featureCount() if source.featureCount() else 0
        
        # We'll store points for all patterns to compare counts later
        all_triangular_points = []
        all_rectangular_points = []
        all_cincooros_points = []
        
        # Process each polygon
        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
                
            # Get the polygon geometry
            geom = feature.geometry()
            
            # Create an internal polygon with the margin of respect
            buffered_geom = geom.buffer(-border_margin, 5)
            
            if buffered_geom.isEmpty():
                feedback.pushWarning(self.tr(f"Warning: A polygon is too small for the specified margin"))
                continue
                
            bbox = buffered_geom.boundingBox()
            
            # Generate points for each pattern type
            if pattern_type == 0 or feedback.isCanceled() == False:
                # Generate triangular/quincunx pattern points
                triangular_points = self.generate_triangular_points(buffered_geom, bbox, spacing_x)
                all_triangular_points.extend(triangular_points)
                
                # If this is the selected pattern, add to sink
                if pattern_type == 0:
                    for point in triangular_points:
                        f = QgsFeature(fields)
                        f.setGeometry(QgsGeometry.fromPointXY(point))
                        f.setAttributes(['Triangular/Quincunx', spacing_x, spacing_x, 'regular'])
                        sink.addFeature(f, QgsFeatureSink.FastInsert)
            
            if pattern_type == 1 or feedback.isCanceled() == False:
                # Generate rectangular/square pattern points
                rectangular_points = self.generate_rectangular_points(buffered_geom, bbox, spacing_x, spacing_y)
                all_rectangular_points.extend(rectangular_points)
                
                # If this is the selected pattern, add to sink
                if pattern_type == 1:
                    pattern_name = 'Square' if is_square else 'Rectangular'
                    for point in rectangular_points:
                        f = QgsFeature(fields)
                        f.setGeometry(QgsGeometry.fromPointXY(point))
                        f.setAttributes([pattern_name, spacing_x, spacing_y, 'regular'])
                        sink.addFeature(f, QgsFeatureSink.FastInsert)
            
            if pattern_type == 2 or feedback.isCanceled() == False:
                # Generate Five of Diamonds pattern points (with centers)
                cincooros_points = self.generate_cincooros_points(buffered_geom, bbox, spacing_x, spacing_y)
                all_cincooros_points.extend(cincooros_points)
                
                # If this is the selected pattern, add to sink
                if pattern_type == 2:
                    for point, point_type in cincooros_points:
                        f = QgsFeature(fields)
                        f.setGeometry(QgsGeometry.fromPointXY(point))
                        f.setAttributes(['Five of Diamonds', spacing_x, spacing_y, point_type])
                        sink.addFeature(f, QgsFeatureSink.FastInsert)
            
            # Update progress
            feedback.setProgress(int(current * total))
        
        # Update final counts
        triangular_count = len(all_triangular_points)
        rectangular_count = len(all_rectangular_points)
        cincooros_count = len([p for p, _ in all_cincooros_points])
        
        # Display results in feedback
        pattern_names = ['Triangular/Quincunx (Tresbolillo)', 'Rectangular/Square', 'Five of Diamonds (Cinco de Oros)']
        selected_pattern = pattern_names[pattern_type]
        
        feedback.pushInfo(self.tr(f"TREE PLANTING PATTERN SUMMARY:"))
        feedback.pushInfo(self.tr(f"Current selected pattern: {selected_pattern}"))
        feedback.pushInfo(self.tr(f"Border margin: {border_margin} units"))
        
        if pattern_type == 0:
            feedback.pushInfo(self.tr(f"- Triangular/Quincunx (Tresbolillo) ({spacing_x}x{spacing_x}): {triangular_count} trees [SELECTED]"))
        else:
            feedback.pushInfo(self.tr(f"- Triangular/Quincunx (Tresbolillo) ({spacing_x}x{spacing_x}): {triangular_count} trees"))
        
        if pattern_type == 1:
            if is_square:
                feedback.pushInfo(self.tr(f"- Square ({spacing_x}x{spacing_x}): {rectangular_count} trees [SELECTED]"))
            else:
                feedback.pushInfo(self.tr(f"- Rectangular ({spacing_x}x{spacing_y}): {rectangular_count} trees [SELECTED]"))
        else:
            if spacing_x == spacing_y:
                feedback.pushInfo(self.tr(f"- Square ({spacing_x}x{spacing_x}): {rectangular_count} trees"))
            else:
                feedback.pushInfo(self.tr(f"- Rectangular ({spacing_x}x{spacing_y}): {rectangular_count} trees"))
        
        if pattern_type == 2:
            feedback.pushInfo(self.tr(f"- Five of Diamonds (Cinco de Oros) ({spacing_x}x{spacing_y}): {cincooros_count} trees [SELECTED]"))
            regular_count = len([p for p, t in all_cincooros_points if t == 'regular'])
            center_count = len([p for p, t in all_cincooros_points if t == 'center'])
            feedback.pushInfo(self.tr(f"  * Grid points: {regular_count} trees"))
            feedback.pushInfo(self.tr(f"  * Center points: {center_count} trees"))
        else:
            feedback.pushInfo(self.tr(f"- Five of Diamonds (Cinco de Oros) ({spacing_x}x{spacing_y}): {cincooros_count} trees"))
        
        # Return the results
        return {
            self.OUTPUT: dest_id,
            self.OUTPUT_TRIANGULAR_COUNT: triangular_count,
            self.OUTPUT_RECTANGULAR_COUNT: rectangular_count,
            self.OUTPUT_CINCOOROS_COUNT: cincooros_count
        }

    def generate_triangular_points(self, geom, bbox, spacing):
        """Generate points in a triangular/quincunx pattern (staggered rows)"""
        points = []
        
        x_min = bbox.xMinimum() - spacing
        x_max = bbox.xMaximum() + spacing
        y_min = bbox.yMinimum() - spacing
        y_max = bbox.yMaximum() + spacing
        
        # Calculate the height of the equilateral triangle for triangular pattern
        triangle_height = spacing * math.sqrt(3) / 2
        
        # Number of columns and rows possible
        width = x_max - x_min
        height = y_max - y_min
        num_cols = math.ceil(width / spacing)
        num_rows = math.ceil(height / triangle_height)
        
        # Adjust the starts to center the grid
        offset_x = (width - (num_cols - 1) * spacing) / 2
        offset_y = (height - (num_rows - 1) * triangle_height) / 2
        
        # Generate points in triangular pattern
        for row in range(num_rows):
            # Offset for alternating rows (triangular arrangement)
            row_offset = spacing / 2 if row % 2 else 0
            
            y = y_min + offset_y + (row * triangle_height)
            
            for col in range(num_cols):
                x = x_min + offset_x + row_offset + (col * spacing)
                point = QgsPointXY(x, y)
                
                if geom.contains(point):
                    points.append(point)
        
        return points

    def generate_rectangular_points(self, geom, bbox, spacing_x, spacing_y):
        """Generate points in a rectangular/square pattern"""
        points = []
        
        x_min = bbox.xMinimum() - spacing_x
        x_max = bbox.xMaximum() + spacing_x
        y_min = bbox.yMinimum() - spacing_y
        y_max = bbox.yMaximum() + spacing_y
        
        # Number of columns and rows possible
        width = x_max - x_min
        height = y_max - y_min
        num_cols = math.ceil(width / spacing_x)
        num_rows = math.ceil(height / spacing_y)
        
        # Adjust the starts to center the grid
        offset_x = (width - (num_cols - 1) * spacing_x) / 2
        offset_y = (height - (num_rows - 1) * spacing_y) / 2
        
        # Generate points in rectangular pattern
        for row in range(num_rows):
            y = y_min + offset_y + (row * spacing_y)
            
            for col in range(num_cols):
                x = x_min + offset_x + (col * spacing_x)
                point = QgsPointXY(x, y)
                
                if geom.contains(point):
                    points.append(point)
        
        return points

    def generate_cincooros_points(self, geom, bbox, spacing_x, spacing_y):
        """
        Generate points in a Five of Diamonds pattern 
        (rectangular/square grid plus center points)
        """
        points = []
        
        x_min = bbox.xMinimum() - spacing_x
        x_max = bbox.xMaximum() + spacing_x
        y_min = bbox.yMinimum() - spacing_y
        y_max = bbox.yMaximum() + spacing_y
        
        # Step 1: Create the regular grid points
        width = x_max - x_min
        height = y_max - y_min
        num_cols = math.ceil(width / spacing_x)
        num_rows = math.ceil(height / spacing_y)
        
        # Adjust the starts to center the grid
        offset_x = (width - (num_cols - 1) * spacing_x) / 2
        offset_y = (height - (num_rows - 1) * spacing_y) / 2
        
        # Regular grid points (corners of cells)
        grid_points = []
        for row in range(num_rows):
            y = y_min + offset_y + (row * spacing_y)
            
            for col in range(num_cols):
                x = x_min + offset_x + (col * spacing_x)
                point = QgsPointXY(x, y)
                
                if geom.contains(point):
                    grid_points.append((point, 'regular'))
        
        # Step 2: Create center points for each cell
        center_points = []
        for row in range(num_rows - 1):
            for col in range(num_cols - 1):
                # Calculate center point of the current cell
                center_x = x_min + offset_x + (col * spacing_x) + (spacing_x / 2)
                center_y = y_min + offset_y + (row * spacing_y) + (spacing_y / 2)
                
                center_point = QgsPointXY(center_x, center_y)
                
                if geom.contains(center_point):
                    center_points.append((center_point, 'center'))
        
        # Combine all points
        points = grid_points + center_points
        
        return points

    def name(self):
        return 'treeplantingpattern'

    def displayName(self):
        return self.tr('Tree Planting Pattern Generator')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'arcgeekcalculator'

    def shortHelpString(self):
        return self.tr("""
        This algorithm generates tree planting points with different patterns inside polygon boundaries.
        
        It supports three planting patterns:
        - Triangular/Quincunx (Tresbolillo): Trees arranged in staggered rows (ideal for sloped terrain)
        - Rectangular/Square: Trees arranged in a grid pattern (square when X=Y, rectangular when X≠Y)
        - Five of Diamonds (Cinco de Oros): Trees arranged as a grid with additional points in the center of each cell
        
        Parameters:
        - Input polygon layer: The area where trees will be planted
        - Planting pattern: The pattern to generate
        - Border margin: Distance to keep from the polygon boundary
        - Spacing: Format as "XxY" (e.g., "3x3" or "4.5x3")
          * For Triangular: only the X value is used
          * For Rectangular/Square: both X and Y values determine the pattern shape
          * For Five of Diamonds: both X and Y values define the grid spacing
        
        Outputs:
        - Tree planting points layer
        - Count statistics for all pattern types
        
        Notes:
        - The spacing determines whether a Rectangular pattern becomes Square (when X=Y)
        - The Five of Diamonds pattern creates a regular grid with additional points in the center of each cell
        - The algorithm calculates the optimal placement of trees and provides a comparison 
          of how many trees could be planted using each pattern
        """)

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return TreePlantingPatternAlgorithm()