from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessing, QgsProcessingAlgorithm,
                      QgsProcessingParameterFeatureSource,
                      QgsProcessingParameterVectorDestination,
                      QgsFeature, QgsGeometry, QgsPointXY,
                      QgsField, QgsFields, QgsWkbTypes,
                      QgsProcessingException, QgsVectorLayer)
import math

class CalculateAnglesAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                self.tr('Input layer (line or polygon)'),
                [QgsProcessing.TypeVectorLine, QgsProcessing.TypeVectorPolygon]
            )
        )
        
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT,
                self.tr('Output vertices with angles'),
                type=QgsProcessing.TypeVectorPoint
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        
        fields = QgsFields()
        fields.append(QgsField('internal_angle', QVariant.Double))
        fields.append(QgsField('external_angle', QVariant.Double))
        fields.append(QgsField('feature_id', QVariant.Int))

        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            fields,
            QgsWkbTypes.Point,
            source.sourceCrs()
        )

        feature_count = 0
        total_points = 0

        # Process features based on geometry type
        for feature in source.getFeatures():
            if feedback.isCanceled():
                break

            geom = feature.geometry()
            feature_count += 1

            if geom.type() == QgsWkbTypes.LineGeometry:
                # Process lines individually
                points = self.process_line_feature(geom, feature_count)
                for point in points:
                    sink.addFeature(point)
                    total_points += 1
            else:
                # Process polygon directly
                points = self.process_polygon_feature(geom, feature_count)
                for point in points:
                    sink.addFeature(point)
                    total_points += 1

            feedback.setProgress(feature_count * 100 / source.featureCount())

        feedback.pushInfo(f'Processed {feature_count} features and created {total_points} angle points')
        
        return {self.OUTPUT: dest_id}

    def process_line_feature(self, geometry, feature_id):
        points = []
        
        # Get all lines (handle both single and multi)
        lines = []
        if geometry.isMultipart():
            lines.extend(geometry.asMultiPolyline())
        else:
            lines.append(geometry.asPolyline())

        # Process each line independently
        for line_idx, line in enumerate(lines):
            if len(line) < 3:
                continue

            # Process each vertex except first and last
            for i in range(1, len(line) - 1):
                # Get three consecutive points
                p1 = line[i - 1]
                p2 = line[i]
                p3 = line[i + 1]

                # Calculate angle
                angle = self.calculate_angle(p1, p2, p3)
                if angle is not None:
                    # Create feature with angles
                    feat = QgsFeature()
                    feat.setGeometry(QgsGeometry.fromPointXY(p2))
                    feat.setAttributes([
                        angle,
                        360 - angle,
                        feature_id
                    ])
                    points.append(feat)

        return points

    def process_polygon_feature(self, geometry, feature_id):
        points = []
        
        # Get all polygons (handle both single and multi)
        polygons = []
        if geometry.isMultipart():
            polygons.extend(geometry.asMultiPolygon())
        else:
            polygons.append(geometry.asPolygon())

        # Process each polygon
        for polygon in polygons:
            vertices = polygon[0]  # Get outer ring
            if len(vertices) < 4:  # Need at least 4 points (first = last)
                continue

            # Remove last point if it's the same as first
            if vertices[0] == vertices[-1]:
                vertices = vertices[:-1]

            # Process each vertex
            for i in range(len(vertices)):
                p1 = vertices[i-1]
                p2 = vertices[i]
                p3 = vertices[(i+1) % len(vertices)]

                # Calculate angle
                angle = self.calculate_angle(p1, p2, p3)
                if angle is not None:
                    # Create feature with angles
                    feat = QgsFeature()
                    feat.setGeometry(QgsGeometry.fromPointXY(p2))
                    feat.setAttributes([
                        angle,
                        360 - angle,
                        feature_id
                    ])
                    points.append(feat)

        return points

    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points."""
        # Calculate vectors
        vector1 = QgsPointXY(p1.x() - p2.x(), p1.y() - p2.y())
        vector2 = QgsPointXY(p3.x() - p2.x(), p3.y() - p2.y())
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(vector1.x()**2 + vector1.y()**2)
        magnitude2 = math.sqrt(vector2.x()**2 + vector2.y()**2)
        
        if magnitude1 * magnitude2 == 0:
            return None
        
        # Calculate angle
        dot_product = vector1.x() * vector2.x() + vector1.y() * vector2.y()
        cos_angle = dot_product / (magnitude1 * magnitude2)
        cos_angle = max(-1, min(1, cos_angle))
        
        return math.degrees(math.acos(cos_angle))

    def name(self):
        return 'calculateangles'

    def displayName(self):
        return self.tr('Calculate Angles')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'arcgeekcalculator'

    def shortHelpString(self):
        return self.tr("""
        This algorithm calculates internal and external angles at vertices of lines or polygons.
        
        Parameters:
        - Input layer: Line or polygon layer to process
        - Output: Point layer with calculated angles
        
        Requirements:
        - Line features must have at least 3 vertices
        - Polygon features must have at least 4 vertices (closed)
        
        Output fields:
        - internal_angle: Interior angle at the vertex
        - external_angle: Exterior angle at the vertex
        - feature_id: ID of the original feature
        """)

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return CalculateAnglesAlgorithm()