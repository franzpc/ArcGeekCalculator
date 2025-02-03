from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessing, QgsFeatureSink, QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource, QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterField, QgsFeature, QgsGeometry, QgsPointXY,
                       QgsFields, QgsField, QgsWkbTypes)

class PolygonToPointsAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'
    POLYGON_ID_FIELD = 'POLYGON_ID_FIELD'

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                self.tr('Input polygon layer'),
                [QgsProcessing.TypeVectorPolygon]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.POLYGON_ID_FIELD,
                self.tr('Polygon ID Field'),
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Any
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr('Output points')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        polygon_id_field = self.parameterAsString(parameters, self.POLYGON_ID_FIELD, context)

        fields = QgsFields()
        fields.append(QgsField('Point_ID_CW', QVariant.Int))
        fields.append(QgsField('Point_ID_CCW', QVariant.Int))
        fields.append(QgsField('Polygon_ID', QVariant.String))

        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                               fields, QgsWkbTypes.Point,
                                               source.sourceCrs())

        total = 100.0 / source.featureCount() if source.featureCount() else 0
        features = source.getFeatures()

        for current, feature in enumerate(features):
            if feedback.isCanceled():
                break

            polygon_id = feature[polygon_id_field]
            polygon_geom = feature.geometry()

            if polygon_geom.isMultipart():
                polygons = polygon_geom.asMultiPolygon()
            else:
                polygons = [polygon_geom.asPolygon()]

            for polygon in polygons:
                exterior_ring = polygon[0]
                # Remove the last point if it's the same as the first point
                if exterior_ring[0] == exterior_ring[-1]:
                    exterior_ring = exterior_ring[:-1]
                
                # Find the northernmost point
                max_y = max(pt.y() for pt in exterior_ring)
                start_index = next(i for i, pt in enumerate(exterior_ring) if pt.y() == max_y)
                num_points = len(exterior_ring)

                # Use a set to keep track of unique points
                unique_points = set()

                for i in range(num_points):
                    index = (start_index + i) % num_points
                    point = exterior_ring[index]
                    
                    # Check if the point is already processed
                    point_tuple = (point.x(), point.y())
                    if point_tuple in unique_points:
                        continue
                    unique_points.add(point_tuple)

                    f = QgsFeature()
                    f.setGeometry(QgsGeometry.fromPointXY(point))
                    f.setAttributes([
                        i + 1,  # Point_ID_CW
                        (num_points - i) % num_points + 1,  # Point_ID_CCW
                        str(polygon_id)
                    ])
                    sink.addFeature(f, QgsFeatureSink.FastInsert)

            feedback.setProgress(int(current * total))

        return {self.OUTPUT: dest_id}

    def name(self):
        return 'extractorderedpointswithbidirectionalnumbering'

    def displayName(self):
        return self.tr('Extract Ordered Points with Bi-directional Numbering')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'arcgeekcalculator'

    def shortHelpString(self):
        return self.tr("""
        This algorithm extracts ordered points from the vertices of input polygons and provides bi-directional numbering.

        Polygon ID Field: A polygon with an identifying field is required.
        
        It's particularly useful when working with multiple polygons, as it:
        1. Extracts unique points from each polygon's vertices.
        2. Orders the points starting from the northernmost point (highest Y coordinate).
        3. Assigns each point both clockwise (CW) and counter-clockwise (CCW) IDs, both starting from 1 at the northernmost point.
        4. Provides the Polygon_ID (from the selected field).

        The Polygon ID Field is crucial when processing multiple polygons, as it allows you to identify which points in the output belong to which input polygon.
        
        Use this tool when you need to:
        - Convert polygon boundaries to point features with bi-directional numbering
        - Analyze or process polygon vertices as individual points with flexible ordering
        - Create input for other point-based algorithms requiring ordinal information
        """)

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return PolygonToPointsAlgorithm()

# This algorithm is based on the work of Mohamed Mokashifi:
# https://github.com/Eng-Moka/Get-Polygon-Points