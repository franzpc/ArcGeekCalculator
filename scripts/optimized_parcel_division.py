from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessing, QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource, QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterNumber, QgsFeature, QgsGeometry, QgsWkbTypes,
                       QgsProcessingException, QgsField, QgsFields, QgsProcessingUtils,
                       QgsVectorLayer, QgsFeatureSink, QgsPointXY, QgsLineString,
                       QgsProcessingMultiStepFeedback, QgsSpatialIndex, QgsProcessingParameterBoolean)
import processing
import math

class OptimizedParcelDivisionAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'
    LOT_WIDTH = 'LOT_WIDTH'
    MERGE_THRESHOLD = 'MERGE_THRESHOLD'
    UNIFORM_CORNERS = 'UNIFORM_CORNERS'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT, self.tr('Input polygon layer'), [QgsProcessing.TypeVectorPolygon]))
        self.addParameter(QgsProcessingParameterNumber(self.LOT_WIDTH, self.tr('Desired lot width'), QgsProcessingParameterNumber.Double, 10.0))
        self.addParameter(QgsProcessingParameterNumber(self.MERGE_THRESHOLD, self.tr('Merge threshold (% of average area)'), QgsProcessingParameterNumber.Double, 30.0, False, 0.0, 100.0))
        self.addParameter(QgsProcessingParameterBoolean(self.UNIFORM_CORNERS, self.tr('Distribute corners uniformly'), defaultValue=True))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, self.tr('Output divided parcels')))

    def processAlgorithm(self, parameters, context, feedback):
        feedback = QgsProcessingMultiStepFeedback(30, feedback)
        
        results = {}
        outputs = {}

        try:
            source = self.parameterAsSource(parameters, self.INPUT, context)
            if source is None:
                raise QgsProcessingException(self.tr("Failed to load the input layer. Please check your input data."))

            feedback.pushInfo(f"Input layer loaded successfully. Feature count: {source.featureCount()}")

            lot_width = self.parameterAsDouble(parameters, self.LOT_WIDTH, context)
            merge_threshold = self.parameterAsDouble(parameters, self.MERGE_THRESHOLD, context) / 100.0
            uniform_corners = self.parameterAsBool(parameters, self.UNIFORM_CORNERS, context)
            feedback.pushInfo(f"Lot width parameter: {lot_width}")
            feedback.pushInfo(f"Merge threshold: {merge_threshold * 100}% of average area")
            feedback.pushInfo(f"Uniform corners: {uniform_corners}")
            
            # Create output fields
            fields = source.fields()
            
            (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                                   fields, QgsWkbTypes.MultiPolygon, source.sourceCrs())
            
            if sink is None:
                raise QgsProcessingException(self.tr("Failed to create the output layer."))

            feedback.pushInfo("Output sink created successfully.")

            # Create temporary layers for regular polygons and OMB of irregular polygons
            regular_layer = QgsVectorLayer("Polygon?crs=" + source.sourceCrs().authid(), "regular_polygons", "memory")
            irregular_omb_layer = QgsVectorLayer("Polygon?crs=" + source.sourceCrs().authid(), "irregular_omb_polygons", "memory")
            
            regular_provider = regular_layer.dataProvider()
            irregular_omb_provider = irregular_omb_layer.dataProvider()
            
            regular_provider.addAttributes(source.fields())
            irregular_omb_provider.addAttributes(source.fields())
            
            regular_layer.updateFields()
            irregular_omb_layer.updateFields()

            feedback.pushInfo("Temporary layers created. Processing input features.")

            features = source.getFeatures()
            total = 100.0 / source.featureCount() if source.featureCount() else 0
            
            has_irregular = False

            for current, feature in enumerate(features):
                if feedback.isCanceled():
                    break

                feedback.pushInfo(f"Processing feature {current + 1}/{source.featureCount()}")

                geom = feature.geometry()
                if geom is None:
                    feedback.pushInfo(f"Feature {current + 1} has no geometry. Skipping.")
                    continue

                if geom.isMultipart():
                    polygons = geom.asMultiPolygon()
                else:
                    polygons = [geom.asPolygon()]

                for polygon in polygons:
                    new_feature = QgsFeature(source.fields())
                    new_feature.setAttributes(feature.attributes())
                    
                    if len(polygon[0]) == 5:  # Regular polygon (rectangular)
                        new_feature.setGeometry(QgsGeometry.fromPolygonXY(polygon))
                        regular_provider.addFeature(new_feature)
                    else:  # Irregular polygon
                        has_irregular = True
                        new_feature.setGeometry(QgsGeometry.fromPolygonXY(polygon))
                        irregular_omb_provider.addFeature(new_feature)

                feedback.setProgress(int(current * total))

            if has_irregular:
                feedback.pushInfo("Processing irregular polygons with Oriented Minimum Bounding Box.")
                omb_result = processing.run("native:orientedminimumboundingbox",
                                            {'INPUT': irregular_omb_layer,
                                             'OUTPUT': 'memory:'}, 
                                            context=context, feedback=feedback)
                
                # Merge regular and OMB of irregular polygons for division lines calculation
                merged_layer = processing.run("native:mergevectorlayers",
                                              {'LAYERS': [regular_layer, omb_result['OUTPUT']],
                                               'OUTPUT': 'memory:'}, 
                                              context=context, feedback=feedback)['OUTPUT']
            else:
                merged_layer = regular_layer

            feedback.pushInfo("Creating division lines.")
            
            horizontal_lines_layer = QgsVectorLayer("LineString?crs=" + source.sourceCrs().authid(), "horizontal_lines", "memory")
            perpendicular_lines_layer = QgsVectorLayer("LineString?crs=" + source.sourceCrs().authid(), "perpendicular_lines", "memory")
            
            horizontal_provider = horizontal_lines_layer.dataProvider()
            perpendicular_provider = perpendicular_lines_layer.dataProvider()
            
            horizontal_provider.addAttributes([QgsField("polygon_id", QVariant.Int)])
            perpendicular_provider.addAttributes([QgsField("polygon_id", QVariant.Int)])
            
            horizontal_lines_layer.updateFields()
            perpendicular_lines_layer.updateFields()

            for feature in merged_layer.getFeatures():
                geom = feature.geometry()
                if geom.isMultipart():
                    polygons = geom.asMultiPolygon()
                else:
                    polygons = [geom.asPolygon()]

                for polygon in polygons:
                    sides = []
                    for i in range(len(polygon[0]) - 1):
                        side = QgsLineString([polygon[0][i], polygon[0][i+1]])
                        sides.append((side, side.length()))
                    
                    sides.sort(key=lambda x: x[1])
                    shortest_sides = sides[:2]

                    midpoints = [QgsPointXY((side[0].startPoint().x() + side[0].endPoint().x()) / 2,
                                            (side[0].startPoint().y() + side[0].endPoint().y()) / 2)
                                 for side in shortest_sides]

                    division_line = QgsGeometry.fromPolylineXY(midpoints)

                    # Add horizontal line
                    horizontal_feature = QgsFeature()
                    horizontal_feature.setGeometry(division_line)
                    horizontal_feature.setAttributes([feature.id()])
                    horizontal_provider.addFeature(horizontal_feature)

                    dx = midpoints[1].x() - midpoints[0].x()
                    dy = midpoints[1].y() - midpoints[0].y()
                    angle = math.atan2(dy, dx)

                    max_width = max(sides[0][1], sides[1][1])

                    line_length = division_line.length()
                    num_segments = math.floor(line_length / lot_width)
                    remainder = line_length - (num_segments * lot_width)
                    start_offset = remainder / 2 if uniform_corners else 0

                    for i in range(1, num_segments):
                        current_distance = start_offset + i * lot_width
                        point = division_line.interpolate(current_distance).asPoint()
                        
                        perp_angle = angle + math.pi/2
                        extended_width = max_width * 1.05
                        x1 = point.x() + extended_width/2 * math.cos(perp_angle)
                        y1 = point.y() + extended_width/2 * math.sin(perp_angle)
                        x2 = point.x() - extended_width/2 * math.cos(perp_angle)
                        y2 = point.y() - extended_width/2 * math.sin(perp_angle)
                        perp_line = QgsGeometry.fromPolylineXY([QgsPointXY(x1, y1), QgsPointXY(x2, y2)])

                        # Add perpendicular line
                        perp_feature = QgsFeature()
                        perp_feature.setGeometry(perp_line)
                        perp_feature.setAttributes([feature.id()])
                        perpendicular_provider.addFeature(perp_feature)

            feedback.pushInfo("Extending lines.")

            # Extend lines by 5 cm
            extended_horizontal = processing.run("native:extendlines",
                                                 {'INPUT': horizontal_lines_layer,
                                                  'START_DISTANCE': 0.05,
                                                  'END_DISTANCE': 0.05,
                                                  'OUTPUT': 'memory:'},
                                                 context=context, feedback=feedback)['OUTPUT']

            extended_perpendicular = processing.run("native:extendlines",
                                                    {'INPUT': perpendicular_lines_layer,
                                                     'START_DISTANCE': 0.05,
                                                     'END_DISTANCE': 0.05,
                                                     'OUTPUT': 'memory:'},
                                                    context=context, feedback=feedback)['OUTPUT']

            feedback.pushInfo("Merging extended lines.")

            # Merge extended lines
            merged_lines = processing.run("native:mergevectorlayers",
                                          {'LAYERS': [extended_horizontal, extended_perpendicular],
                                           'OUTPUT': 'memory:'},
                                          context=context, feedback=feedback)['OUTPUT']

            feedback.pushInfo("Clipping lines with input layer.")

            # Clip the lines layer with the input layer
            clipped_lines = processing.run("native:clip",
                                           {'INPUT': merged_lines,
                                            'OVERLAY': parameters[self.INPUT],
                                            'OUTPUT': 'memory:'},
                                           context=context, feedback=feedback)['OUTPUT']

            feedback.pushInfo("Splitting polygons with lines.")

            # Split polygons with lines
            split_polygons = processing.run("native:splitwithlines",
                                            {'INPUT': parameters[self.INPUT],
                                             'LINES': clipped_lines,
                                             'OUTPUT': 'memory:'},
                                            context=context, feedback=feedback)['OUTPUT']

            feedback.pushInfo("Calculating areas and merging small polygons.")

            # Function to merge small polygons
            def merge_small_polygons(polygons_layer, min_area_threshold):
                # Calculate areas and average area
                total_area = 0
                feature_count = 0
                areas = {}
                for feature in polygons_layer.getFeatures():
                    area = feature.geometry().area()
                    total_area += area
                    areas[feature.id()] = area
                    feature_count += 1

                avg_area = total_area / feature_count if feature_count > 0 else 0

                # Create a spatial index
                spatial_index = QgsSpatialIndex()
                for feature in polygons_layer.getFeatures():
                    spatial_index.addFeature(feature)

                # Merge small polygons
                merged_features = []
                processed_ids = set()

                for feature in polygons_layer.getFeatures():
                    if feature.id() in processed_ids:
                        continue

                    if areas[feature.id()] < min_area_threshold:
                        # Find neighboring features
                        neighbors = spatial_index.intersects(feature.geometry().boundingBox())
                        best_neighbor = None
                        max_shared_boundary = 0

                        for neighbor_id in neighbors:
                            if neighbor_id == feature.id() or neighbor_id in processed_ids:
                                continue

                            neighbor_feature = polygons_layer.getFeature(neighbor_id)
                            shared_boundary = feature.geometry().intersection(neighbor_feature.geometry())
                            
                            if shared_boundary.type() == QgsWkbTypes.LineGeometry:
                                shared_length = shared_boundary.length()
                                if shared_length > max_shared_boundary:
                                    max_shared_boundary = shared_length
                                    best_neighbor = neighbor_feature

                        if best_neighbor:
                            merged_geom = feature.geometry().combine(best_neighbor.geometry())
                            merged_feature = QgsFeature(feature.fields())
                            merged_feature.setGeometry(merged_geom)
                            merged_feature.setAttributes(feature.attributes())
                            merged_features.append(merged_feature)
                            processed_ids.add(feature.id())
                            processed_ids.add(best_neighbor.id())
                        else:
                            merged_features.append(feature)
                    else:
                        merged_features.append(feature)

                return merged_features, len(processed_ids) // 2

            # First pass of merging small polygons
            min_area_threshold = merge_threshold * (sum(f.geometry().area() for f in split_polygons.getFeatures()) / split_polygons.featureCount())
            merged_features, merged_count = merge_small_polygons(split_polygons, min_area_threshold)
            feedback.pushInfo(f"First pass: Merged {merged_count} small polygons.")

            # Create a temporary layer with the merged features
            temp_layer = QgsVectorLayer("Polygon?crs=" + source.sourceCrs().authid(), "temp_layer", "memory")
            temp_provider = temp_layer.dataProvider()
            temp_provider.addAttributes(split_polygons.fields())
            temp_layer.updateFields()
            temp_provider.addFeatures(merged_features)

            # Second pass of merging small polygons
            min_area_threshold = merge_threshold * (sum(f.geometry().area() for f in temp_layer.getFeatures()) / temp_layer.featureCount())
            final_features, merged_count = merge_small_polygons(temp_layer, min_area_threshold)
            feedback.pushInfo(f"Second pass: Merged {merged_count} small polygons.")

            # Add final features to the output sink
            for feature in final_features:
                sink.addFeature(feature, QgsFeatureSink.FastInsert)

            feedback.pushInfo(f"Processing completed. Final parcels created.")

            return {self.OUTPUT: dest_id}

        except Exception as e:
            feedback.reportError(f"Error occurred: {str(e)}")
            raise QgsProcessingException(str(e))

    def name(self):
        return 'optimizedparceldivision'

    def displayName(self):
        return self.tr('Optimized Parcel Division')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'arcgeekcalculator'

    def shortHelpString(self):
        return self.tr("""
        This algorithm performs an optimized division of parcels based on a specified lot width.

        Parameters:
        - Input polygon layer: The layer containing the parcels to be divided.
        - Desired lot width: The width you want each resulting lot to have.
        - Merge threshold: Percentage of average area below which small polygons will be merged.
        - Distribute corners uniformly: If checked, the algorithm will distribute corner lots uniformly.

        The algorithm works as follows:
        1. Divides parcels based on the specified lot width.
        2. Performs two passes of merging small polygons to avoid leaving very small parcels.
        3. Creates a new layer with the divided parcels.

        Note: This tool works best with rectangular polygons, but for irregular plots it is recommended to split them when the angle of the polygon is very steep.
        """)

    def createInstance(self):
        return OptimizedParcelDivisionAlgorithm()

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)