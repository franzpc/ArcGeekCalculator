from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer, 
                       QgsProcessingParameterFeatureSink, QgsProcessingParameterPoint, 
                       QgsWkbTypes, QgsField, QgsVectorLayer, QgsFeatureSink, 
                       QgsProcessing, QgsProcessingParameterVectorLayer,
                       QgsProcessingException, QgsMessageLog, Qgis,
                       QgsProcessingParameterNumber, QgsRasterLayer, QgsSnappingConfig,
                       QgsProcessingParameterBoolean)
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
    EXTEND_MAIN_CHANNEL = 'EXTEND_MAIN_CHANNEL'
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
        stream_layer.startEditing()
        
        if 'strah_rec' not in [field.name() for field in stream_layer.fields()]:
            stream_layer.dataProvider().addAttributes([QgsField('strah_rec', QVariant.Int)])
            stream_layer.updateFields()
        
        processed_features = {}
        
        def get_strahler_order(feature_id):
            if feature_id in processed_features:
                return processed_features[feature_id]
            
            feature = stream_layer.getFeature(feature_id)
            upstream_features = self.find_upstream_features(feature, stream_layer)
            
            if not upstream_features:
                order = 1
            else:
                upstream_orders = [get_strahler_order(f.id()) for f in upstream_features]
                max_order = max(upstream_orders)
                count_max = upstream_orders.count(max_order)
                order = max_order + 1 if count_max > 1 else max_order
            
            processed_features[feature_id] = order
            feature['strah_rec'] = order
            stream_layer.updateFeature(feature)
            return order
        
        outlet_features = []
        for feature in stream_layer.getFeatures():
            if not self.find_downstream_feature(feature, stream_layer):
                outlet_features.append(feature)
        
        if not outlet_features:
            for feature in stream_layer.getFeatures():
                feature['strah_rec'] = 1
                stream_layer.updateFeature(feature)
        else:
            for outlet in outlet_features:
                get_strahler_order(outlet.id())
        
        stream_layer.commitChanges()

    def extend_main_channel(self, stream_layer, feedback):
        stream_layer.startEditing()
        
        if 'strah_ext' not in [field.name() for field in stream_layer.fields()]:
            stream_layer.dataProvider().addAttributes([QgsField('strah_ext', QVariant.Int)])
            stream_layer.updateFields()
        
        # Copy values from strah_rec to strah_ext
        for feature in stream_layer.getFeatures():
            if feature['strah_rec'] is not None:
                feature['strah_ext'] = feature['strah_rec']
                stream_layer.updateFeature(feature)
        
        # Find the main outlet
        outlets = []
        for feature in stream_layer.getFeatures():
            if not self.find_downstream_feature(feature, stream_layer):
                outlets.append(feature)
        
        if not outlets:
            stream_layer.commitChanges()
            return
            
        main_outlet = max(outlets, key=lambda feat: feat['strah_ext'] if feat['strah_ext'] is not None else 0)
        max_order = main_outlet['strah_ext']
        
        # For each order, starting from the highest
        for current_order in range(max_order, 1, -1):
            # Find terminal segments of this order
            last_segments = []
            for feature in stream_layer.getFeatures():
                if feature['strah_ext'] == current_order:
                    upstream_features = self.find_upstream_features(feature, stream_layer)
                    same_or_higher_order = [f for f in upstream_features 
                                           if f['strah_ext'] >= current_order]
                    if not same_or_higher_order:
                        last_segments.append(feature)
            
            if not last_segments:
                continue
            
            # For each terminal segment, find all branches of the next lower order
            for last_segment in last_segments:
                upstream_features = self.find_upstream_features(last_segment, stream_layer)
                next_order_features = [f for f in upstream_features 
                                      if f['strah_ext'] == current_order - 1]
                
                if not next_order_features:
                    continue
                
                # Find all complete branches of the lower order
                branches = []
                for next_feature in next_order_features:
                    branch = self.find_complete_branch(next_feature, current_order - 1, stream_layer)
                    if branch:
                        total_length = sum(f.geometry().length() for f in branch)
                        branches.append((branch, total_length))
                
                if not branches:
                    continue
                
                # Select the longest branch
                longest_branch = max(branches, key=lambda x: x[1])[0]
                
                # Extend the current order to this branch
                for feature in longest_branch:
                    feature['strah_ext'] = current_order
                    stream_layer.updateFeature(feature)
        
        stream_layer.commitChanges()
        
        # Run the fix hierarchy function automatically
        self.fix_tributary_hierarchy(stream_layer, feedback)

    def find_complete_branch(self, start_feature, target_order, stream_layer, visited=None):
        if visited is None:
            visited = set()
        
        if start_feature.id() in visited:
            return []
        
        visited.add(start_feature.id())
        
        if start_feature['strah_ext'] != target_order:
            return []
        
        result = [start_feature]
        
        upstream_features = self.find_upstream_features(start_feature, stream_layer)
        target_order_upstream = [f for f in upstream_features if f['strah_ext'] == target_order]
        
        for upstream in target_order_upstream:
            branch = self.find_complete_branch(upstream, target_order, stream_layer, visited)
            result.extend(branch)
        
        return result

    def fix_tributary_hierarchy(self, stream_layer, feedback):
        stream_layer.startEditing()
        
        # Create a field for the final result if it doesn't exist
        if 'strah_final' not in [field.name() for field in stream_layer.fields()]:
            stream_layer.dataProvider().addAttributes([QgsField('strah_final', QVariant.Int)])
            stream_layer.updateFields()
        
        # Initialize all values to 1
        for feature in stream_layer.getFeatures():
            feature['strah_final'] = 1
            stream_layer.updateFeature(feature)
        
        # Identify all segments of the main channel
        max_order = max(feature['strah_ext'] for feature in stream_layer.getFeatures() 
                       if feature['strah_ext'] is not None)
        main_channel_features = {f.id(): f for f in stream_layer.getFeatures() if f['strah_ext'] == max_order}
        
        # Mark main channel with the maximum order
        for feature_id, feature in main_channel_features.items():
            feature['strah_final'] = max_order
            stream_layer.updateFeature(feature)
        
        # Build a graph of connections
        graph = {}
        for feature in stream_layer.getFeatures():
            fid = feature.id()
            upstream_ids = [f.id() for f in self.find_upstream_features(feature, stream_layer)]
            downstream = self.find_downstream_feature(feature, stream_layer)
            downstream_id = downstream.id() if downstream else None
            
            graph[fid] = {'upstream': upstream_ids, 'downstream': downstream_id}
        
        # Starting from each main channel segment, assign orders to tributaries
        for start_id in main_channel_features.keys():
            # All segments directly connected upstream to the main channel
            for upstream_id in graph[start_id]['upstream']:
                if upstream_id in main_channel_features:
                    continue  # Ignore if part of main channel
                
                # Process this tributary and all its upstream segments
                queue = deque([(upstream_id, max_order - 1)])  # (id, expected_order)
                
                while queue:
                    current_id, expected_order = queue.popleft()
                    current_feature = stream_layer.getFeature(current_id)
                    
                    if expected_order < 1:
                        expected_order = 1
                    
                    if current_feature['strah_final'] < expected_order:
                        current_feature['strah_final'] = expected_order
                        stream_layer.updateFeature(current_feature)
                    
                    next_order = expected_order - 1
                    if next_order < 1:
                        next_order = 1
                        
                    for next_id in graph[current_id]['upstream']:
                        queue.append((next_id, next_order))
        
        # Second pass to verify no jumps of more than one level
        all_changed = True
        max_iterations = 10
        iterations = 0
        
        while all_changed and iterations < max_iterations:
            all_changed = False
            iterations += 1
            
            for feature in stream_layer.getFeatures():
                downstream = self.find_downstream_feature(feature, stream_layer)
                if not downstream:
                    continue
                
                downstream_order = downstream['strah_final']
                current_order = feature['strah_final']
                
                if current_order < downstream_order - 1 and current_order > 1:
                    feature['strah_final'] = downstream_order - 1
                    stream_layer.updateFeature(feature)
                    all_changed = True
                elif current_order == 1 and downstream_order > 2:
                    feature['strah_final'] = 2
                    stream_layer.updateFeature(feature)
                    all_changed = True
        
        stream_layer.commitChanges()

    def canCancel(self):
        return True
        
    def onClose(self):
        self.deactivate_snapping()
        super().onClose()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_DEM, 'Input DEM'))
        self.addParameter(QgsProcessingParameterPoint(self.POUR_POINT, 'Pour Point (click on the river)'))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_STREAM, 'Input Stream Network', 
                                                            types=[QgsProcessing.TypeVectorLine], optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.SMOOTH_ITERATIONS, 'Smoothing Iterations', 
                                                       type=QgsProcessingParameterNumber.Integer, 
                                                       minValue=0, maxValue=10, defaultValue=1))
        self.addParameter(QgsProcessingParameterNumber(self.SMOOTH_OFFSET, 'Smoothing Offset', 
                                                       type=QgsProcessingParameterNumber.Double, 
                                                       minValue=0.0, maxValue=0.5, defaultValue=0.25))
        self.addParameter(QgsProcessingParameterBoolean(self.EXTEND_MAIN_CHANNEL, 'Extend Main Channel', 
                                                       defaultValue=False))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT_BASIN, 'Output Basin', QgsProcessing.TypeVectorPolygon))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT_STREAM, 'Output Basin Stream Network', QgsProcessing.TypeVectorLine, optional=True))

    def processAlgorithm(self, parameters, context, feedback):
        dem = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
        pour_point = self.parameterAsPoint(parameters, self.POUR_POINT, context)
        input_stream = self.parameterAsVectorLayer(parameters, self.INPUT_STREAM, context)
        smooth_iterations = self.parameterAsInt(parameters, self.SMOOTH_ITERATIONS, context)
        smooth_offset = self.parameterAsDouble(parameters, self.SMOOTH_OFFSET, context)
        extend_main = self.parameterAsBoolean(parameters, self.EXTEND_MAIN_CHANNEL, context)

        if not dem.isValid():
            raise QgsProcessingException(self.tr('Invalid input DEM'))

        if input_stream and input_stream.geometryType() != QgsWkbTypes.LineGeometry:
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

        if input_stream:
            try:
                clipped_stream = processing.run('native:clip', {
                    'INPUT': input_stream,
                    'OVERLAY': basin_layer,
                    'OUTPUT': 'memory:'
                }, context=context, feedback=feedback)['OUTPUT']

                self.calculate_strahler(clipped_stream, feedback)
                
                if extend_main:
                    self.extend_main_channel(clipped_stream, feedback)

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
                feedback.pushInfo("Continuing with basin output only")

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
        
        If a stream network is provided, it will be clipped to the basin boundary and Strahler stream order will be calculated.
        
        Parameters:
            Input DEM: A raster layer representing the terrain elevation
            Pour Point: The outlet point of the watershed. Snapping to streams is enabled
            Input Stream Network: Optional. A line vector layer representing the stream network
            Smoothing Iterations: Number of iterations for smoothing the basin boundary (0-10)
            Smoothing Offset: Offset value for smoothing (0.0-0.5)
            Extend Main Channel: When checked, extends the main channel upstream along the longest path and fixes tributary hierarchy
            
        Outputs:
            Output Basin: A polygon layer representing the delineated watershed basin
            Output Basin Stream Network: Optional. A line layer with Strahler order values
            
        Note: When using the Extend Main Channel option, the calculated Strahler orders are modified to create
        a more consistent stream network classification for cartographic purposes.
        """)

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)