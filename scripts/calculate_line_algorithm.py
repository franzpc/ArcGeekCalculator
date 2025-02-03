from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterNumber,
                       QgsProcessingParameterFeatureSource, QgsProcessingParameterField,
                       QgsProcessingParameterVectorDestination, QgsVectorLayer,
                       QgsFeature, QgsGeometry, QgsPointXY, QgsField, QgsProject,
                       QgsWkbTypes, QgsFeatureSink, QgsProcessingParameterCrs,
                       QgsFields, QgsProcessingMultiStepFeedback, QgsCoordinateReferenceSystem,
                       QgsProcessing, QgsProcessingException, QgsProcessingParameterEnum)
from qgis.PyQt.QtCore import QVariant
import math

class CalculateLineAlgorithm(QgsProcessingAlgorithm):
    INPUT_X = 'INPUT_X'
    INPUT_Y = 'INPUT_Y'
    INPUT_TABLE = 'INPUT_TABLE'
    FIELD_DISTANCE = 'FIELD_DISTANCE'
    FIELD_ANGLE = 'FIELD_ANGLE'
    ANGLE_TYPE = 'ANGLE_TYPE'
    FIELD_OBSERVATIONS = 'FIELD_OBSERVATIONS'
    OUTPUT_CRS = 'OUTPUT_CRS'
    OUTPUT_LINE = 'OUTPUT_LINE'
    OUTPUT_POINTS = 'OUTPUT_POINTS'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterNumber(self.INPUT_X, 'Starting X coordinate', QgsProcessingParameterNumber.Double))
        self.addParameter(QgsProcessingParameterNumber(self.INPUT_Y, 'Starting Y coordinate', QgsProcessingParameterNumber.Double))
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT_TABLE, 'Input table', [QgsProcessing.TypeVector, QgsProcessing.TypeFile], optional=False))
        self.addParameter(QgsProcessingParameterField(self.FIELD_DISTANCE, 'Distance field', parentLayerParameterName=self.INPUT_TABLE, type=QgsProcessingParameterField.Any, optional=False))
        self.addParameter(QgsProcessingParameterField(self.FIELD_ANGLE, 'Angle field', parentLayerParameterName=self.INPUT_TABLE, type=QgsProcessingParameterField.Any, optional=False))
        self.addParameter(QgsProcessingParameterEnum(self.ANGLE_TYPE, 'Angle type in input data', options=['Azimuth', 'Polar'], defaultValue=0))
        self.addParameter(QgsProcessingParameterField(self.FIELD_OBSERVATIONS, 'Observations field', parentLayerParameterName=self.INPUT_TABLE, type=QgsProcessingParameterField.Any, optional=True))
        self.addParameter(QgsProcessingParameterCrs(self.OUTPUT_CRS, 'Output CRS', optional=False))
        self.addParameter(QgsProcessingParameterVectorDestination(self.OUTPUT_LINE, 'Output line layer'))
        self.addParameter(QgsProcessingParameterVectorDestination(self.OUTPUT_POINTS, 'Output points layer'))

    def processAlgorithm(self, parameters, context, feedback):
        multi_feedback = QgsProcessingMultiStepFeedback(2, feedback)
        
        x_start = self.parameterAsDouble(parameters, self.INPUT_X, context)
        y_start = self.parameterAsDouble(parameters, self.INPUT_Y, context)
        source = self.parameterAsSource(parameters, self.INPUT_TABLE, context)
        field_distance = self.parameterAsString(parameters, self.FIELD_DISTANCE, context)
        field_angle = self.parameterAsString(parameters, self.FIELD_ANGLE, context)
        angle_type = self.parameterAsEnum(parameters, self.ANGLE_TYPE, context)
        field_observations = self.parameterAsString(parameters, self.FIELD_OBSERVATIONS, context)
        output_crs = self.parameterAsCrs(parameters, self.OUTPUT_CRS, context)

        # Prepare fields for the line layer
        line_fields = QgsFields()
        line_fields.append(QgsField('length', QVariant.Double))

        # Prepare fields for the point layer
        point_fields = QgsFields()
        point_fields.append(QgsField('ID', QVariant.Int))
        point_fields.append(QgsField('Distance', QVariant.Double))
        point_fields.append(QgsField('Angle', QVariant.Double))
        point_fields.append(QgsField('X', QVariant.Double))
        point_fields.append(QgsField('Y', QVariant.Double))
        if field_observations:
            point_fields.append(QgsField('Observations', QVariant.String))
        
        # Create sinks for output layers
        (line_sink, line_dest_id) = self.parameterAsSink(parameters, self.OUTPUT_LINE, context,
                                                         line_fields, QgsWkbTypes.LineString, output_crs)
        (point_sink, point_dest_id) = self.parameterAsSink(parameters, self.OUTPUT_POINTS, context,
                                                           point_fields, QgsWkbTypes.Point, output_crs)

        if line_sink is None or point_sink is None:
            raise QgsProcessingException(self.tr('Could not create output layers'))

        points = [QgsPointXY(x_start, y_start)]
        x_previous, y_previous = x_start, y_start
        
        # Create initial point
        initial_point = QgsFeature(point_fields)
        initial_point.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x_start, y_start)))
        initial_attributes = [0, 0.0, 0.0, float(x_start), float(y_start)]
        if field_observations:
            initial_attributes.append('')
        initial_point.setAttributes(initial_attributes)
        point_sink.addFeature(initial_point, QgsFeatureSink.FastInsert)
        
        features = source.getFeatures()
        total = 100.0 / source.featureCount() if source.featureCount() else 0

        for current, feature in enumerate(features):
            if feedback.isCanceled():
                break

            try:
                distance = float(feature[field_distance] or 0)
                angle = float(feature[field_angle] or 0)
                
                # Calculate the next point based on the angle type
                if angle_type == 0:  # Azimuth
                    angle_radians = math.radians(angle)
                    dx = distance * math.sin(angle_radians)
                    dy = distance * math.cos(angle_radians)
                else:  # Polar
                    angle_radians = math.radians(angle)
                    dx = distance * math.cos(angle_radians)
                    dy = distance * math.sin(angle_radians)
                
                x_current = x_previous + dx
                y_current = y_previous + dy
                new_point = QgsPointXY(x_current, y_current)
                points.append(new_point)
                
                # Create point feature
                point_feature = QgsFeature(point_fields)
                point_feature.setGeometry(QgsGeometry.fromPointXY(new_point))
                point_attributes = [
                    current + 1,
                    float(distance),
                    float(angle),  # Store the original angle value
                    float(x_current),
                    float(y_current)
                ]
                if field_observations:
                    obs_value = str(feature[field_observations] or '')
                    point_attributes.append(obs_value)
                point_feature.setAttributes(point_attributes)
                point_sink.addFeature(point_feature, QgsFeatureSink.FastInsert)
                
                # Create line segment
                line_feature = QgsFeature(line_fields)
                line_geom = QgsGeometry.fromPolylineXY([QgsPointXY(x_previous, y_previous), new_point])
                length = line_geom.length()
                line_feature.setGeometry(line_geom)
                line_feature.setAttributes([float(length)])
                line_sink.addFeature(line_feature, QgsFeatureSink.FastInsert)
                
                x_previous, y_previous = x_current, y_current
                
            except (ValueError, TypeError) as e:
                feedback.reportError(f"Error in row {current + 1}: {str(e)}")
                continue

            feedback.setProgress(int(current * total))

        return {self.OUTPUT_LINE: line_dest_id, self.OUTPUT_POINTS: point_dest_id}

    def name(self):
        return 'calculatelinefromcoordinatesandtable'

    def displayName(self):
        return 'Azimuth and distance from Coordinates and Table'

    def group(self):
        return 'ArcGeek Calculator'

    def groupId(self):
        return 'arcgeek_calculator'

    def shortHelpString(self):
        return """
        This algorithm calculates a line and points from starting coordinates and a table of distances and angles.

        Parameters:
        - Starting X and Y coordinates: The initial point of the line.
        - Input table: A vector layer or table (csv, txt, etc) containing distance and angle data.
        - Distance field: The field in the input table containing distance values.
        - Angle field: The field in the input table containing angle values.
        - Angle type in input data: Specify whether the input angles are Azimuth or Polar.
        - Observations field (optional): A field for additional information about each point.
        - Output CRS: The coordinate reference system for the output layers.

        Outputs:
        - A line layer with segments connecting the calculated points.
        - A point layer with all calculated points.

        The algorithm reads the input table row by row, calculating new points based on the distance and angle from the previous point. It starts from the given X and Y coordinates and creates a new point for each row in the table.

        Angle types:
        - Azimuth: Measured clockwise from north (0-360 degrees)
        - Polar: Measured counterclockwise from east (0-360 degrees)

        Use this tool when you need to:
        - Convert tabular distance and angle data into spatial features.
        - Generate a line path from a series of movements.
        - Create point locations from relative positioning data.

        Note: Ensure that you select the correct angle type that matches your input data, and that you select the appropriate CRS for your data to avoid potential inaccuracies in the results.
        """

    def createInstance(self):
        return CalculateLineAlgorithm()