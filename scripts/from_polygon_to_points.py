from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessing, QgsFeatureSink, QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource, QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterField, QgsFeature, QgsGeometry, QgsPointXY,
                       QgsFields, QgsField, QgsWkbTypes, QgsProcessingParameterPoint,
                       QgsCoordinateTransform, QgsProject)
import math

class PolygonToPointsAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'
    POLYGON_ID_FIELD = 'POLYGON_ID_FIELD'
    START_POINT = 'START_POINT'

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
            QgsProcessingParameterPoint(
                self.START_POINT,
                self.tr('Starting coordinate'),
                optional=True
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
        
        start_point_geom = self.parameterAsPoint(parameters, self.START_POINT, context)
        start_point_crs = self.parameterAsPointCrs(parameters, self.START_POINT, context)
        
        start_point = None
        if start_point_geom and not start_point_geom.isEmpty():
            if start_point_crs.isValid() and start_point_crs != source.sourceCrs():
                transform = QgsCoordinateTransform(start_point_crs, source.sourceCrs(), QgsProject.instance())
                transformed_point = transform.transform(start_point_geom)
                start_point = QgsPointXY(transformed_point.x(), transformed_point.y())
            else:
                start_point = QgsPointXY(start_point_geom.x(), start_point_geom.y())

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
                if exterior_ring[0] == exterior_ring[-1]:
                    exterior_ring = exterior_ring[:-1]
                
                if start_point is not None:
                    min_distance = float('inf')
                    start_index = 0
                    
                    for i, pt in enumerate(exterior_ring):
                        dx = pt.x() - start_point.x()
                        dy = pt.y() - start_point.y()
                        distance = math.sqrt(dx * dx + dy * dy)
                        if distance < min_distance:
                            min_distance = distance
                            start_index = i
                else:
                    max_y = max(pt.y() for pt in exterior_ring)
                    start_index = next(i for i, pt in enumerate(exterior_ring) if pt.y() == max_y)
                
                num_points = len(exterior_ring)
                unique_points = set()
                point_counter = 0

                for i in range(num_points):
                    index = (start_index + i) % num_points
                    point = exterior_ring[index]
                    
                    point_tuple = (point.x(), point.y())
                    if point_tuple in unique_points:
                        continue
                    unique_points.add(point_tuple)
                    
                    point_counter += 1
                    
                    # CCW: punto 1 = 1, luego va en reversa desde num_points
                    if point_counter == 1:
                        ccw_id = 1
                    else:
                        ccw_id = num_points - point_counter + 2
                    
                    f = QgsFeature()
                    f.setGeometry(QgsGeometry.fromPointXY(point))
                    f.setAttributes([
                        point_counter,
                        ccw_id,
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
        
        Starting Coordinate (optional): Click on the map to select a coordinate. The algorithm will use the vertex closest to this point as the starting point for numbering. If not provided, the northernmost point is used.
        
        Features:
        1. Extracts unique points from each polygon's vertices.
        2. Orders the points starting from a specified coordinate or the northernmost point.
        3. Assigns each point both clockwise (CW) and counter-clockwise (CCW) IDs, both starting from 1.
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