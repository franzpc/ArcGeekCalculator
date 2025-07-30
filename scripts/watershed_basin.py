from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer, 
                       QgsProcessingParameterFeatureSink, QgsProcessingParameterPoint, 
                       QgsWkbTypes, QgsField, QgsVectorLayer, QgsFeatureSink, 
                       QgsProcessing, QgsProcessingParameterVectorLayer,
                       QgsProcessingException, QgsMessageLog, Qgis,
                       QgsProcessingParameterNumber, QgsRasterLayer, QgsSnappingConfig)
from qgis.PyQt.QtCore import QVariant, QCoreApplication
from qgis.utils import iface
import processing
from collections import deque

class WatershedBasinDelineationAlgorithm(QgsProcessingAlgorithm):
    INPUT_DEM = 'INPUT_DEM'
    POUR_POINT = 'POUR_POINT'
    INPUT_STREAM = 'INPUT_STREAM'
    OUTPUT_BASIN = 'OUTPUT_BASIN'
    OUTPUT_STREAM = 'OUTPUT_STREAM'
    SMOOTH_ITERATIONS = 'SMOOTH_ITERATIONS'
    SMOOTH_OFFSET = 'SMOOTH_OFFSET'
    MAX_RASTER_SIZE = 100000000

    def __init__(self):
        super().__init__()
        self.activate_snapping()

    def activate_snapping(self):
        snapping_config = iface.mapCanvas().snappingUtils().config()
        snapping_config.setEnabled(True)
        try:
            snapping_config.setTypeFlag(QgsSnappingConfig.VertexFlag | QgsSnappingConfig.SegmentFlag)
        except TypeError:
            snapping_config.setType(QgsSnappingConfig.Vertex | QgsSnappingConfig.Segment)
        
        iface.mapCanvas().snappingUtils().setConfig(snapping_config)

    def deactivate_snapping(self):
        snapping_config = iface.mapCanvas().snappingUtils().config()
        snapping_config.setEnabled(False)
        try:
            snapping_config.setTypeFlag(QgsSnappingConfig.VertexFlag | QgsSnappingConfig.SegmentFlag)
        except TypeError:
            snapping_config.setType(QgsSnappingConfig.Vertex | QgsSnappingConfig.Segment)
        
        iface.mapCanvas().snappingUtils().setConfig(snapping_config)

    def get_points(self, geometry):
        if geometry.type() == QgsWkbTypes.LineGeometry:
            if geometry.isMultipart():
                multilines = geometry.asMultiPolyline()
                if multilines:
                    start_point = multilines[0][0]
                    end_point = multilines[-1][-1]
                    return start_point, end_point
            else:
                polyline = geometry.asPolyline()
                if polyline:
                    return polyline[0], polyline[-1]
        return None, None

    def find_upstream_features(self, feature, layer, tolerance=0.0001):
        if not feature.geometry() or feature.geometry().isEmpty():
            return []
            
        start_point, _ = self.get_points(feature.geometry())
        if start_point is None:
            return []
        
        upstream_features = []
        for f in layer.getFeatures():
            if f.id() == feature.id():
                continue
                
            if not f.geometry() or f.geometry().isEmpty():
                continue
                
            _, end_point = self.get_points(f.geometry())
            if end_point is None:
                continue
                
            if abs(start_point.x() - end_point.x()) < tolerance and abs(start_point.y() - end_point.y()) < tolerance:
                upstream_features.append(f)
                
        return upstream_features

    def find_downstream_feature(self, feature, layer, tolerance=0.0001):
        if not feature.geometry() or feature.geometry().isEmpty():
            return None
            
        _, end_point = self.get_points(feature.geometry())
        if end_point is None:
            return None
        
        for f in layer.getFeatures():
            if f.id() == feature.id():
                continue
                
            if not f.geometry() or f.geometry().isEmpty():
                continue
                
            start_point, _ = self.get_points(f.geometry())
            if start_point is None:
                continue
                
            if abs(end_point.x() - start_point.x()) < tolerance and abs(end_point.y() - start_point.y()) < tolerance:
                return f
                
        return None

    def calculate_strahler(self, stream_layer, feedback):
        feedback.pushInfo("Starting Strahler order calculation...")
        
        if not stream_layer.startEditing():
            feedback.reportError("Could not start editing stream layer")
            return
        
        field_names = [field.name() for field in stream_layer.fields()]
        if 'sthr_original' not in field_names:
            if not stream_layer.dataProvider().addAttributes([QgsField('sthr_original', QVariant.Int)]):
                feedback.reportError("Could not add sthr_original field")
                return
            stream_layer.updateFields()
        
        for feature in stream_layer.getFeatures():
            feature['sthr_original'] = 1
            if not stream_layer.updateFeature(feature):
                feedback.reportError(f"Could not update feature {feature.id()}")
        
        processed_features = {}
        
        def get_strahler_order(feature_id):
            if feature_id in processed_features:
                return processed_features[feature_id]
            
            feature = stream_layer.getFeature(feature_id)
            if not feature.isValid():
                return 1
                
            upstream_features = self.find_upstream_features(feature, stream_layer)
            
            if not upstream_features:
                order = 1
            else:
                upstream_orders = [get_strahler_order(f.id()) for f in upstream_features if f.isValid()]
                if upstream_orders:
                    max_order = max(upstream_orders)
                    count_max = upstream_orders.count(max_order)
                    order = max_order + 1 if count_max > 1 else max_order
                else:
                    order = 1
            
            processed_features[feature_id] = order
            feature['sthr_original'] = order
            stream_layer.updateFeature(feature)
            return order
        
        outlet_features = []
        for feature in stream_layer.getFeatures():
            if not self.find_downstream_feature(feature, stream_layer):
                outlet_features.append(feature)
        
        if outlet_features:
            for outlet in outlet_features:
                get_strahler_order(outlet.id())
        
        if not stream_layer.commitChanges():
            feedback.reportError("Could not commit changes to stream layer")
        else:
            feedback.pushInfo("Strahler order calculation completed")

    def extend_main_channel(self, stream_layer, feedback):
        feedback.pushInfo("Starting main channel extension...")
        
        if not stream_layer.startEditing():
            feedback.reportError("Could not start editing stream layer for extension")
            return
        
        field_names = [field.name() for field in stream_layer.fields()]
        if 'sthr_extend' not in field_names:
            if not stream_layer.dataProvider().addAttributes([QgsField('sthr_extend', QVariant.Int)]):
                feedback.reportError("Could not add sthr_extend field")
                return
            stream_layer.updateFields()
        
        for feature in stream_layer.getFeatures():
            original_value = feature['sthr_original'] if feature['sthr_original'] is not None else 1
            feature['sthr_extend'] = original_value
            stream_layer.updateFeature(feature)
        
        outlets = []
        for feature in stream_layer.getFeatures():
            if not self.find_downstream_feature(feature, stream_layer):
                outlets.append(feature)
        
        if not outlets:
            if not stream_layer.commitChanges():
                feedback.reportError("Could not commit changes")
            return
            
        main_outlet = max(outlets, key=lambda feat: feat['sthr_extend'] if feat['sthr_extend'] is not None else 0)
        max_order = main_outlet['sthr_extend']
        
        for current_order in range(max_order, 1, -1):
            last_segments = []
            for feature in stream_layer.getFeatures():
                if feature['sthr_extend'] == current_order:
                    upstream_features = self.find_upstream_features(feature, stream_layer)
                    same_or_higher_order = [f for f in upstream_features 
                                           if f['sthr_extend'] >= current_order]
                    if not same_or_higher_order:
                        last_segments.append(feature)
            
            for last_segment in last_segments:
                upstream_features = self.find_upstream_features(last_segment, stream_layer)
                next_order_features = [f for f in upstream_features 
                                      if f['sthr_extend'] == current_order - 1]
                
                if next_order_features:
                    branches = []
                    for next_feature in next_order_features:
                        branch = self.find_complete_branch(next_feature, current_order - 1, stream_layer)
                        if branch:
                            total_length = sum(f.geometry().length() for f in branch if f.geometry() is not None)
                            branches.append((branch, total_length))
                    
                    if branches:
                        longest_branch = max(branches, key=lambda x: x[1])[0]
                        for feature in longest_branch:
                            feature['sthr_extend'] = current_order
                            stream_layer.updateFeature(feature)
        
        if not stream_layer.commitChanges():
            feedback.reportError("Could not commit changes for extend")
        else:
            feedback.pushInfo("Main channel extension completed")

    def find_complete_branch(self, start_feature, target_order, stream_layer, visited=None):
        if visited is None:
            visited = set()
        
        if start_feature.id() in visited:
            return []
        
        visited.add(start_feature.id())
        
        if start_feature['sthr_extend'] != target_order:
            return []
        
        result = [start_feature]
        
        upstream_features = self.find_upstream_features(start_feature, stream_layer)
        target_order_upstream = [f for f in upstream_features if f['sthr_extend'] == target_order]
        
        for upstream in target_order_upstream:
            branch = self.find_complete_branch(upstream, target_order, stream_layer, visited)
            result.extend(branch)
        
        return result

    def extend_to_headwaters(self, stream_layer, feedback):
        feedback.pushInfo("Starting final channel extension to headwaters...")
        
        if not stream_layer.startEditing():
            feedback.reportError("Could not start editing stream layer for final extension")
            return
        
        field_names = [field.name() for field in stream_layer.fields()]
        if 'sthr_final' not in field_names:
            if not stream_layer.dataProvider().addAttributes([QgsField('sthr_final', QVariant.Int)]):
                feedback.reportError("Could not add sthr_final field")
                return
            stream_layer.updateFields()
        
        for feature in stream_layer.getFeatures():
            feature['sthr_final'] = 1
            stream_layer.updateFeature(feature)
        
        outlets = []
        for feature in stream_layer.getFeatures():
            if not self.find_downstream_feature(feature, stream_layer):
                outlets.append(feature)
        
        if not outlets:
            if not stream_layer.commitChanges():
                feedback.reportError("Could not commit changes")
            return
            
        main_outlet = max(outlets, key=lambda feat: feat['sthr_extend'] if feat['sthr_extend'] is not None else 0)
        max_order = main_outlet['sthr_extend'] if main_outlet['sthr_extend'] is not None else 1
        
        try:
            main_channel_path = self.find_main_channel_to_headwater(main_outlet, stream_layer)
            feedback.pushInfo(f"Found main channel path with {len(main_channel_path)} segments")
            
            for feature in main_channel_path:
                feature['sthr_final'] = max_order
                if not stream_layer.updateFeature(feature):
                    feedback.reportError(f"Could not update feature {feature.id()}")
        except Exception as e:
            feedback.reportError(f"Error tracing main channel: {str(e)}")
        
        if not stream_layer.commitChanges():
            feedback.reportError("Could not commit final changes")
        else:
            feedback.pushInfo("Final channel extension completed")

    def find_main_channel_to_headwater(self, start_feature, stream_layer, visited=None):
        if visited is None:
            visited = set()
        
        if not start_feature or not start_feature.isValid() or start_feature.id() in visited:
            return []
        
        visited.add(start_feature.id())
        result = [start_feature]
        
        try:
            upstream_features = self.find_upstream_features(start_feature, stream_layer)
            upstream_features = [f for f in upstream_features if f.isValid()]
            
            if not upstream_features:
                return result
            
            if len(upstream_features) == 1:
                upstream_path = self.find_main_channel_to_headwater(upstream_features[0], stream_layer, visited)
                result.extend(upstream_path)
            else:
                branches = []
                for upstream_feature in upstream_features:
                    if upstream_feature.id() not in visited:
                        branch_path = self.trace_complete_path(upstream_feature, stream_layer, set())
                        if branch_path:
                            total_length = sum(f.geometry().length() for f in branch_path if f.geometry() is not None)
                            branches.append((total_length, upstream_feature))
                
                if branches:
                    longest_branch = max(branches, key=lambda x: x[0])
                    upstream_path = self.find_main_channel_to_headwater(longest_branch[1], stream_layer, visited)
                    result.extend(upstream_path)
        except Exception as e:
            QgsMessageLog.logMessage(f"Error in find_main_channel_to_headwater: {str(e)}", level=Qgis.Critical)
        
        return result

    def trace_complete_path(self, start_feature, stream_layer, visited=None):
        if visited is None:
            visited = set()
        
        if not start_feature or not start_feature.isValid() or start_feature.id() in visited:
            return []
        
        visited.add(start_feature.id())
        result = [start_feature]
        
        try:
            upstream_features = self.find_upstream_features(start_feature, stream_layer)
            upstream_features = [f for f in upstream_features if f.isValid() and f.id() not in visited]
            
            for upstream in upstream_features:
                branch = self.trace_complete_path(upstream, stream_layer, visited)
                result.extend(branch)
        except Exception as e:
            QgsMessageLog.logMessage(f"Error in trace_complete_path: {str(e)}", level=Qgis.Critical)
        
        return result

    def canCancel(self):
        return True
        
    def onClose(self):
        self.deactivate_snapping()
        super().onClose()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_DEM, 'Input DEM'))
        self.addParameter(QgsProcessingParameterPoint(self.POUR_POINT, 'Pour Point (click on the river)'))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_STREAM, 'Input Stream Network', 
                                                            types=[QgsProcessing.TypeVectorLine], optional=False))
        self.addParameter(QgsProcessingParameterNumber(self.SMOOTH_ITERATIONS, 'Smoothing Iterations', 
                                                       type=QgsProcessingParameterNumber.Integer, 
                                                       minValue=0, maxValue=10, defaultValue=1))
        self.addParameter(QgsProcessingParameterNumber(self.SMOOTH_OFFSET, 'Smoothing Offset', 
                                                       type=QgsProcessingParameterNumber.Double, 
                                                       minValue=0.0, maxValue=0.5, defaultValue=0.25))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT_BASIN, 'Watershed Basin', QgsProcessing.TypeVectorPolygon))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT_STREAM, 'Basin Stream Network', QgsProcessing.TypeVectorLine))

    def processAlgorithm(self, parameters, context, feedback):
        dem = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
        pour_point = self.parameterAsPoint(parameters, self.POUR_POINT, context)
        input_stream = self.parameterAsVectorLayer(parameters, self.INPUT_STREAM, context)
        smooth_iterations = self.parameterAsInt(parameters, self.SMOOTH_ITERATIONS, context)
        smooth_offset = self.parameterAsDouble(parameters, self.SMOOTH_OFFSET, context)

        if not dem.isValid():
            raise QgsProcessingException(self.tr('Invalid input DEM'))

        if not input_stream or input_stream.geometryType() != QgsWkbTypes.LineGeometry:
            raise QgsProcessingException(self.tr('Input Stream Network must be a line layer'))

        original_cell_size = dem.rasterUnitsPerPixelX()
        cell_size_multiplier = 1
        max_attempts = 3
        
        for attempt in range(max_attempts):
            current_cell_size = original_cell_size * cell_size_multiplier
            resampled_dem = self.resample_dem(dem, current_cell_size, context, feedback)
            
            raster_size = resampled_dem.width() * resampled_dem.height()
            if raster_size <= self.MAX_RASTER_SIZE:
                if attempt > 0:
                    feedback.pushInfo(self.tr(f'DEM resampled to {current_cell_size:.2f} units per pixel for processing.'))
                break
            
            cell_size_multiplier *= 3
        
        if raster_size > self.MAX_RASTER_SIZE:
            raise QgsProcessingException(self.tr('Input DEM is too large to process efficiently even after resampling.'))

        filled_dem = processing.run('grass7:r.fill.dir', {
            'input': resampled_dem,
            'format': 0,
            'output': 'TEMPORARY_OUTPUT',
            'direction': 'TEMPORARY_OUTPUT',
            'areas': 'TEMPORARY_OUTPUT'
        }, context=context, feedback=feedback)['output']

        watershed_result = processing.run('grass7:r.watershed', {
            'elevation': filled_dem,
            'convergence': 5,
            'memory': 300,
            '-s': True,
            'accumulation': 'TEMPORARY_OUTPUT',
            'drainage': 'TEMPORARY_OUTPUT'
        }, context=context, feedback=feedback)
        
        drainage = watershed_result['drainage']

        pour_point_str = f'{pour_point.x()},{pour_point.y()}'
        basin_raster = processing.run('grass7:r.water.outlet', {
            'input': drainage,
            'coordinates': pour_point_str,
            'output': 'TEMPORARY_OUTPUT'
        }, context=context, feedback=feedback)['output']

        basin_vector_result = processing.run('grass7:r.to.vect', {
            'input': basin_raster,
            'type': 2,
            'column': 'value',
            '-s': True,
            'output': 'TEMPORARY_OUTPUT',
            'GRASS_OUTPUT_TYPE_PARAMETER': 3
        }, context=context, feedback=feedback)
        
        basin_vector = basin_vector_result['output']
        basin_layer = QgsVectorLayer(basin_vector, 'basin', 'ogr')

        if basin_layer.featureCount() > 1:
            max_area = 0
            largest_feature = None
    
            for feature in basin_layer.getFeatures():
                area = feature.geometry().area()
                if area > max_area:
                    max_area = area
                    largest_feature = feature
    
            if largest_feature:
                largest_polygon = QgsVectorLayer("Polygon?crs=" + basin_layer.crs().authid(), "largest_polygon", "memory")
                provider = largest_polygon.dataProvider()
                provider.addFeatures([largest_feature])
                largest_polygon.updateExtents()
            else:
                largest_polygon = basin_vector
        else:
            largest_polygon = basin_vector

        smoothed_basin = processing.run('native:smoothgeometry', {
            'INPUT': largest_polygon,
            'ITERATIONS': smooth_iterations,
            'OFFSET': smooth_offset,
            'MAX_ANGLE': 180,
            'OUTPUT': 'memory:'
        }, context=context, feedback=feedback)['OUTPUT']

        basin_layer = smoothed_basin

        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT_BASIN, context,
                                               basin_layer.fields(), QgsWkbTypes.Polygon, basin_layer.crs())
        
        if sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT_BASIN))

        features = basin_layer.getFeatures()
        for feature in features:
            sink.addFeature(feature, QgsFeatureSink.FastInsert)

        results = {self.OUTPUT_BASIN: dest_id}

        try:
            clipped_stream = processing.run('native:clip', {
                'INPUT': input_stream,
                'OVERLAY': basin_layer,
                'OUTPUT': 'memory:'
            }, context=context, feedback=feedback)['OUTPUT']

            self.calculate_strahler(clipped_stream, feedback)
            self.extend_main_channel(clipped_stream, feedback)
            self.extend_to_headwaters(clipped_stream, feedback)

            (stream_sink, stream_dest_id) = self.parameterAsSink(parameters, self.OUTPUT_STREAM, context,
                                                                 clipped_stream.fields(), QgsWkbTypes.LineString, clipped_stream.crs())
            
            if stream_sink is None:
                raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT_STREAM))

            stream_features = clipped_stream.getFeatures()
            for feature in stream_features:
                stream_sink.addFeature(feature, QgsFeatureSink.FastInsert)

            results[self.OUTPUT_STREAM] = stream_dest_id
        except Exception as e:
            feedback.reportError(f"Error processing stream network: {str(e)}")
            raise QgsProcessingException(f"Error processing stream network: {str(e)}")

        return results

    def resample_dem(self, dem, new_cell_size, context, feedback):
        extent = dem.extent()
        width = int(extent.width() / new_cell_size)
        height = int(extent.height() / new_cell_size)
        
        resampled = processing.run("gdal:warpreproject", {
            'INPUT': dem,
            'SOURCE_CRS': dem.crs(),
            'TARGET_CRS': dem.crs(),
            'RESAMPLING': 0,
            'TARGET_RESOLUTION': new_cell_size,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }, context=context, feedback=feedback)['OUTPUT']
        
        return QgsRasterLayer(resampled, 'resampled_dem', 'gdal')

    def name(self):
        return 'watershedbasindelineation'

    def displayName(self):
        return self.tr('Watershed Basin Delineation')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'arcgeekcalculator'

    def createInstance(self):
        return WatershedBasinDelineationAlgorithm()

    def shortHelpString(self):
        return self.tr("""
        This algorithm delineates a watershed basin based on a Digital Elevation Model (DEM) and a pour point.
        It uses GRASS GIS algorithms for hydrological analysis and watershed delineation.
        
        The stream network is required and will be clipped to the basin boundary with three Strahler stream order calculations:
        - sthr_original: Standard Strahler order calculation
        - sthr_extend: Extended main channel along longest paths
        - sthr_final: Main channel extended to all headwaters
        
        Parameters:
            Input DEM: A raster layer representing the terrain elevation
            Pour Point: The outlet point of the watershed. Snapping to streams is enabled
            Input Stream Network: A line vector layer representing the stream network (required)
            Smoothing Iterations: Number of iterations for smoothing the basin boundary (0-10)
            Smoothing Offset: Offset value for smoothing (0.0-0.5)
            
        Outputs:
            Output Basin: A polygon layer representing the delineated watershed basin
            Output Basin Stream Network: A line layer with three different Strahler order calculations
        """)

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)