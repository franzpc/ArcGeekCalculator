import os
import processing
from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessing, QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterNumber, QgsProcessingParameterFeatureSink,
                       QgsVectorLayer, QgsRasterLayer, QgsField, QgsWkbTypes,
                       QgsProcessingException, QgsFeatureSink, QgsFeature, 
                       QgsProcessingParameterVectorLayer, QgsProcessingParameterBoolean,
                       QgsUnitTypes, QgsSpatialIndex, QgsMessageLog, Qgis)

class MultipleBasinDelineationAlgorithm(QgsProcessingAlgorithm):
    """
    Multiple Basin Delineation by Points
    Combines stream network generation, point snapping to streams, 
    and watershed delineation in one tool.
    """
    
    # Input parameters
    INPUT_DEM = 'INPUT_DEM'
    INPUT_POINTS = 'INPUT_POINTS'
    FLOW_THRESHOLD = 'FLOW_THRESHOLD'
    SMOOTH_ITERATIONS = 'SMOOTH_ITERATIONS'
    SMOOTH_OFFSET = 'SMOOTH_OFFSET'
    PROCESS_ALL_POINTS = 'PROCESS_ALL_POINTS'
    MAX_POINTS = 'MAX_POINTS'
    
    # Output parameters
    OUTPUT_STREAMS = 'OUTPUT_STREAMS'
    OUTPUT_INTERSECTION_POINTS = 'OUTPUT_INTERSECTION_POINTS'
    OUTPUT_BASINS = 'OUTPUT_BASINS'
    OUTPUT_BASIN_STREAMS = 'OUTPUT_BASIN_STREAMS'
    
    MAX_RASTER_SIZE = 100000000  # 100 million cells

    def initAlgorithm(self, config=None):
        """Initialize algorithm parameters"""
        
        # Input DEM
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT_DEM, 
            'Input DEM (30m recommended)',
            optional=False
        ))
        
        # Input point shapefile
        self.addParameter(QgsProcessingParameterVectorLayer(
            self.INPUT_POINTS,
            'Input Point Shapefile',
            types=[QgsProcessing.TypeVectorPoint],
            optional=False
        ))
        
        # Flow accumulation threshold
        self.addParameter(QgsProcessingParameterNumber(
            self.FLOW_THRESHOLD,
            'Flow Accumulation Threshold',
            type=QgsProcessingParameterNumber.Integer,
            minValue=1,
            maxValue=100000,
            defaultValue=5000
        ))
        
        
        # Smoothing parameters
        self.addParameter(QgsProcessingParameterNumber(
            self.SMOOTH_ITERATIONS,
            'Smooth Iterations',
            type=QgsProcessingParameterNumber.Integer,
            minValue=1,
            maxValue=10,
            defaultValue=1
        ))
        
        self.addParameter(QgsProcessingParameterNumber(
            self.SMOOTH_OFFSET,
            'Smooth Offset',
            type=QgsProcessingParameterNumber.Double,
            minValue=0.0,
            maxValue=0.5,
            defaultValue=0.25
        ))
        
        # Point processing options
        self.addParameter(QgsProcessingParameterBoolean(
            self.PROCESS_ALL_POINTS,
            'Process All Points',
            defaultValue=True
        ))
        
        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_POINTS,
            'Maximum Points to Process (if not processing all)',
            type=QgsProcessingParameterNumber.Integer,
            minValue=1,
            maxValue=1000,
            defaultValue=5
        ))
        
        # Output parameters
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_STREAMS,
            'Global Stream Network',
            QgsProcessing.TypeVectorLine
        ))
        
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_INTERSECTION_POINTS,
            'Snapped Pour Points',
            QgsProcessing.TypeVectorPoint
        ))
        
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_BASINS,
            'Watershed Basins (with Drainage Areas)',
            QgsProcessing.TypeVectorPolygon
        ))
        
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_BASIN_STREAMS,
            'Local Stream Network',
            QgsProcessing.TypeVectorLine
        ))

    def processAlgorithm(self, parameters, context, feedback):
        """Execute the complete watershed analysis workflow"""
        try:
            # Get input parameters
            dem_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
            points_layer = self.parameterAsVectorLayer(parameters, self.INPUT_POINTS, context)
            flow_threshold = self.parameterAsInt(parameters, self.FLOW_THRESHOLD, context)
            smooth_iterations = self.parameterAsInt(parameters, self.SMOOTH_ITERATIONS, context)
            smooth_offset = self.parameterAsDouble(parameters, self.SMOOTH_OFFSET, context)
            process_all_points = self.parameterAsBoolean(parameters, self.PROCESS_ALL_POINTS, context)
            max_points = self.parameterAsInt(parameters, self.MAX_POINTS, context)
            
            # Validate inputs
            if not dem_layer.isValid():
                raise QgsProcessingException(self.tr('Invalid input DEM'))
            
            if not points_layer.isValid():
                raise QgsProcessingException(self.tr('Invalid points layer'))
            
            feedback.pushInfo("Starting complete watershed analysis workflow...")
            feedback.pushInfo(f"Using Flow Accumulation Threshold: {flow_threshold}")
            
            # Step 1: Generate stream network (30% progress)
            feedback.setProgress(5)
            feedback.pushInfo("Step 1: Generating stream network...")
            stream_network = self.generate_stream_network(
                dem_layer, flow_threshold, smooth_iterations, smooth_offset, context, feedback
            )
            
            # Step 2: Snap input points to stream network (60% progress)
            feedback.setProgress(30)
            feedback.pushInfo("Step 2: Snapping input points to stream network...")
            snapped_points = self.snap_points_to_stream(points_layer, stream_network, context, feedback)
            
            if not snapped_points or snapped_points.featureCount() == 0:
                raise QgsProcessingException(self.tr('No points found or snapping failed'))
            
            # Limit points if requested (70% progress)
            feedback.setProgress(60)
            pour_points = snapped_points  # Use snapped points directly
            if not process_all_points:
                feedback.pushInfo(f"Limiting processing to {max_points} points...")
                pour_points = self.limit_points(snapped_points, max_points, context, feedback)
            
            feedback.pushInfo(f"Processing {pour_points.featureCount()} pour points...")
            
            # Step 3: Run watershed delineation (85% progress)
            feedback.setProgress(70)
            feedback.pushInfo("Step 3: Running watershed basin delineation...")
            basin_polygons, basin_streams = self.run_watershed_delineation(
                dem_layer, pour_points, stream_network, smooth_iterations, smooth_offset, context, feedback
            )
            
            # Step 4: Calculate drainage areas (95% progress)
            feedback.setProgress(90)
            feedback.pushInfo("Step 4: Calculating drainage areas...")
            final_basins = self.calculate_drainage_areas(basin_polygons, context, feedback)
            
            # Prepare output sinks (100% progress)
            feedback.setProgress(95)
            feedback.pushInfo("Preparing outputs...")
            
            # Stream network output
            stream_sink, stream_dest_id = self.parameterAsSink(
                parameters, self.OUTPUT_STREAMS, context,
                stream_network.fields(), QgsWkbTypes.LineString, dem_layer.crs()
            )
            
            for feature in stream_network.getFeatures():
                stream_sink.addFeature(feature, QgsFeatureSink.FastInsert)
            
            # Snapped points output
            points_sink, points_dest_id = self.parameterAsSink(
                parameters, self.OUTPUT_INTERSECTION_POINTS, context,
                pour_points.fields(), QgsWkbTypes.Point, dem_layer.crs()
            )
            
            for feature in pour_points.getFeatures():
                points_sink.addFeature(feature, QgsFeatureSink.FastInsert)
            
            # Basin polygons output
            basins_sink, basins_dest_id = self.parameterAsSink(
                parameters, self.OUTPUT_BASINS, context,
                final_basins.fields(), QgsWkbTypes.Polygon, dem_layer.crs()
            )
            
            for feature in final_basins.getFeatures():
                basins_sink.addFeature(feature, QgsFeatureSink.FastInsert)
            
            # Basin streams output
            basin_streams_sink, basin_streams_dest_id = self.parameterAsSink(
                parameters, self.OUTPUT_BASIN_STREAMS, context,
                basin_streams.fields(), QgsWkbTypes.LineString, dem_layer.crs()
            )
            
            for feature in basin_streams.getFeatures():
                basin_streams_sink.addFeature(feature, QgsFeatureSink.FastInsert)
            
            feedback.setProgress(100)
            feedback.pushInfo("Watershed analysis completed successfully!")
            
            return {
                self.OUTPUT_STREAMS: stream_dest_id,
                self.OUTPUT_INTERSECTION_POINTS: points_dest_id,
                self.OUTPUT_BASINS: basins_dest_id,
                self.OUTPUT_BASIN_STREAMS: basin_streams_dest_id
            }
            
        except Exception as e:
            feedback.reportError(f"Error in watershed analysis: {str(e)}")
            raise QgsProcessingException(str(e))

    def generate_stream_network(self, dem_layer, threshold, smooth_iterations, smooth_offset, context, feedback):
        """Generate stream network using GRASS algorithms"""
        try:
            # Check DEM size and resample if needed
            original_cell_size = dem_layer.rasterUnitsPerPixelX()
            cell_size_multiplier = 1
            max_attempts = 3
            
            for attempt in range(max_attempts):
                current_cell_size = original_cell_size * cell_size_multiplier
                resampled_dem = self.resample_dem(dem_layer, current_cell_size, context, feedback)
                
                raster_size = resampled_dem.width() * resampled_dem.height()
                if raster_size <= self.MAX_RASTER_SIZE:
                    if attempt > 0:
                        feedback.pushInfo(f'DEM resampled to {current_cell_size:.2f} units per pixel for processing.')
                    break
                
                cell_size_multiplier *= 3
            
            if raster_size > self.MAX_RASTER_SIZE:
                raise QgsProcessingException('Input DEM is too large to process efficiently even after resampling.')
            
            # Fill DEM
            filled_dem = processing.run("grass7:r.fill.dir", {
                'input': resampled_dem,
                'output': QgsProcessing.TEMPORARY_OUTPUT,
                'direction': QgsProcessing.TEMPORARY_OUTPUT,
                'areas': QgsProcessing.TEMPORARY_OUTPUT,
                'format': 0
            }, context=context, feedback=feedback)['output']
            
            # Calculate flow accumulation
            flow_accumulation = processing.run("grass7:r.watershed", {
                'elevation': filled_dem,
                'accumulation': QgsProcessing.TEMPORARY_OUTPUT,
                'drainage': QgsProcessing.TEMPORARY_OUTPUT,
                'threshold': 10000,
                '-s': True,
                '-m': True
            }, context=context, feedback=feedback)['accumulation']
            
            # Extract streams
            streams = processing.run("grass7:r.stream.extract", {
                'elevation': filled_dem,
                'accumulation': flow_accumulation,
                'threshold': threshold,
                'stream_vector': QgsProcessing.TEMPORARY_OUTPUT,
                'stream_raster': QgsProcessing.TEMPORARY_OUTPUT,
                'direction': QgsProcessing.TEMPORARY_OUTPUT,
                'GRASS_OUTPUT_TYPE_PARAMETER': 2
            }, context=context, feedback=feedback)['stream_vector']
            
            # Apply smoothing
            smoothed_streams = processing.run("native:smoothgeometry", {
                'INPUT': streams,
                'ITERATIONS': smooth_iterations,
                'OFFSET': smooth_offset,
                'MAX_ANGLE': 180,
                'OUTPUT': 'memory:'
            }, context=context, feedback=feedback)['OUTPUT']
            
            # Calculate stream orders
            ordered_streams = self.calculate_stream_orders(smoothed_streams, context, feedback)
            
            return ordered_streams
            
        except Exception as e:
            feedback.reportError(f"Error generating stream network: {str(e)}")
            raise

    def calculate_stream_orders(self, stream_layer, context, feedback):
        """Calculate Strahler and Shreve stream orders"""
        try:
            if isinstance(stream_layer, str):
                layer = QgsVectorLayer(stream_layer, "Streams", "ogr")
            elif isinstance(stream_layer, QgsVectorLayer):
                layer = stream_layer
            else:
                raise QgsProcessingException(self.tr('Invalid stream layer type'))
            
            if not layer.isValid():
                raise QgsProcessingException(self.tr('Invalid stream layer'))
            
            layer_provider = layer.dataProvider()
            
            # Add Strahler and Shreve order fields if they don't exist
            fields_to_add = []
            if layer.fields().indexFromName("Strahler") == -1:
                fields_to_add.append(QgsField("Strahler", QVariant.Int))
            if layer.fields().indexFromName("Shreve") == -1:
                fields_to_add.append(QgsField("Shreve", QVariant.Int))
            
            if fields_to_add:
                layer_provider.addAttributes(fields_to_add)
                layer.updateFields()
            
            index = QgsSpatialIndex(layer.getFeatures())
            outlets = [f for f in layer.getFeatures() if self.is_valid_feature(f) and not self.find_downstream_features(f, index, layer)]
            
            layer.startEditing()
            total_features = len(outlets)
            for current, outlet in enumerate(outlets):
                if feedback.isCanceled():
                    break
                self.get_stream_orders(outlet, layer, index)
                feedback.setProgress(int((current + 1) / total_features * 100))
            layer.commitChanges()
            
            return layer
            
        except Exception as e:
            feedback.reportError(f"Error calculating stream orders: {str(e)}")
            raise

    def get_stream_orders(self, feature, layer, index):
        try:
            upstream_features = self.find_upstream_features(feature, index, layer)
            if not upstream_features:
                feature['Strahler'] = 1
                feature['Shreve'] = 1
                layer.updateFeature(feature)
                return 1, 1
            else:
                upstream_orders = [self.get_stream_orders(f, layer, index) for f in upstream_features]
                max_strahler = max([order[0] for order in upstream_orders])
                strahler = max_strahler + 1 if [order[0] for order in upstream_orders].count(max_strahler) > 1 else max_strahler
                shreve = sum([order[1] for order in upstream_orders])
                feature['Strahler'] = strahler
                feature['Shreve'] = shreve
                layer.updateFeature(feature)
                return strahler, shreve
        except Exception as e:
            QgsMessageLog.logMessage(f"Error in get_stream_orders: {str(e)}", level=Qgis.Critical)
            raise

    def find_upstream_features(self, feature, index, layer):
        try:
            if not self.is_valid_feature(feature):
                return []
            start_point = self.get_start_point(feature.geometry())
            if start_point is None:
                return []
            return [f for f in self.get_nearby_features(start_point, index, layer)
                    if f.id() != feature.id() and self.get_end_point(f.geometry()) == start_point]
        except Exception as e:
            QgsMessageLog.logMessage(f"Error in find_upstream_features: {str(e)}", level=Qgis.Critical)
            return []

    def find_downstream_features(self, feature, index, layer):
        try:
            if not self.is_valid_feature(feature):
                return []
            end_point = self.get_end_point(feature.geometry())
            if end_point is None:
                return []
            return [f for f in self.get_nearby_features(end_point, index, layer)
                    if f.id() != feature.id() and self.get_start_point(f.geometry()) == end_point]
        except Exception as e:
            QgsMessageLog.logMessage(f"Error in find_downstream_features: {str(e)}", level=Qgis.Critical)
            return []

    def is_valid_feature(self, feature):
        return feature.geometry() is not None and not feature.geometry().isNull() and feature.geometry().isGeosValid()

    def get_start_point(self, geometry):
        if geometry.type() == QgsWkbTypes.LineGeometry:
            return geometry.asPolyline()[0] if geometry.asPolyline() else None
        elif geometry.type() == QgsWkbTypes.MultiLineGeometry:
            lines = geometry.asMultiPolyline()
            return lines[0][0] if lines else None
        return None

    def get_end_point(self, geometry):
        if geometry.type() == QgsWkbTypes.LineGeometry:
            return geometry.asPolyline()[-1] if geometry.asPolyline() else None
        elif geometry.type() == QgsWkbTypes.MultiLineGeometry:
            lines = geometry.asMultiPolyline()
            return lines[-1][-1] if lines else None
        return None

    def get_nearby_features(self, point, index, layer):
        return [layer.getFeature(fid) for fid in index.nearestNeighbor(point, 5)]

    def resample_dem(self, dem, new_cell_size, context, feedback):
        """Resample DEM to new cell size"""
        try:
            resampled = processing.run("gdal:warpreproject", {
                'INPUT': dem,
                'SOURCE_CRS': dem.crs(),
                'TARGET_CRS': dem.crs(),
                'RESAMPLING': 0,  # Nearest neighbor
                'TARGET_RESOLUTION': new_cell_size,
                'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
            }, context=context, feedback=feedback)['OUTPUT']
            
            return QgsRasterLayer(resampled, 'resampled_dem', 'gdal')
        except Exception as e:
            feedback.reportError(f"Error resampling DEM: {str(e)}")
            raise

    def limit_points(self, points_layer, max_points, context, feedback):
        """Limit the number of points for processing"""
        try:
            limited_layer = QgsVectorLayer(f"Point?crs={points_layer.crs().authid()}", 
                                         "limited_points", "memory")
            
            limited_layer.dataProvider().addAttributes(points_layer.fields())
            limited_layer.updateFields()
            
            features = []
            for i, feature in enumerate(points_layer.getFeatures()):
                if i >= max_points:
                    break
                features.append(feature)
            
            limited_layer.dataProvider().addFeatures(features)
            limited_layer.updateExtents()
            
            feedback.pushInfo(f"Limited to {len(features)} points for processing")
            return limited_layer
            
        except Exception as e:
            feedback.reportError(f"Error limiting points: {str(e)}")
            raise

    def snap_points_to_stream(self, points_layer, stream_layer, context, feedback):
        """Snap input points to the nearest stream network"""
        try:
            feedback.pushInfo("Snapping points to stream network...")
            
            # Use QGIS snap geometries to lines processing algorithm
            snapped_points = processing.run("native:snapgeometries", {
                'INPUT': points_layer,
                'REFERENCE_LAYER': stream_layer,
                'TOLERANCE': 1000,  # 1000 units tolerance (adjust based on DEM resolution)
                'BEHAVIOR': 0,  # Snap to closest point
                'OUTPUT': 'memory:snapped_points'
            }, context=context, feedback=feedback)['OUTPUT']
            
            feedback.pushInfo(f"Snapped {snapped_points.featureCount()} points to stream network")
            return snapped_points
            
        except Exception as e:
            feedback.reportError(f"Error snapping points to stream: {str(e)}")
            raise

    def run_watershed_delineation(self, dem_layer, pour_points, stream_layer, smooth_iterations, smooth_offset, context, feedback):
        """Run watershed basin delineation using GRASS algorithms"""
        try:
            # Resample DEM if needed
            original_cell_size = dem_layer.rasterUnitsPerPixelX()
            cell_size_multiplier = 1
            max_attempts = 3
            
            for attempt in range(max_attempts):
                current_cell_size = original_cell_size * cell_size_multiplier
                resampled_dem = self.resample_dem(dem_layer, current_cell_size, context, feedback)
                
                raster_size = resampled_dem.width() * resampled_dem.height()
                if raster_size <= self.MAX_RASTER_SIZE:
                    break
                
                cell_size_multiplier *= 3
            
            # Fill DEM and calculate drainage
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

            # Create output layers
            basin_fields = pour_points.fields()
            basin_fields.append(QgsField("basin_id", QVariant.Int))
            basin_layer = QgsVectorLayer(f"Polygon?crs={dem_layer.crs().authid()}", "basins", "memory")
            basin_layer.dataProvider().addAttributes(basin_fields)
            basin_layer.updateFields()

            stream_fields = stream_layer.fields()
            stream_fields.append(QgsField("basin_id", QVariant.Int))
            stream_output_layer = QgsVectorLayer(f"LineString?crs={stream_layer.crs().authid()}", "basin_streams", "memory")
            stream_output_layer.dataProvider().addAttributes(stream_fields)
            stream_output_layer.updateFields()

            basin_id_counter = 1
            for point_feature in pour_points.getFeatures():
                pour_point = point_feature.geometry().asPoint()
                pour_point_str = f'{pour_point.x()},{pour_point.y()}'
                
                feedback.pushInfo(f"Processing pour point ID: {point_feature.id()}")

                # Delineate basin
                basin_raster = processing.run('grass7:r.water.outlet', {
                    'input': drainage,
                    'coordinates': pour_point_str,
                    'output': 'TEMPORARY_OUTPUT'
                }, context=context, feedback=feedback)['output']

                # Convert to vector
                basin_vector_result = processing.run('grass7:r.to.vect', {
                    'input': basin_raster,
                    'type': 2,
                    'column': 'value',
                    '-s': True,
                    'output': 'TEMPORARY_OUTPUT',
                    'GRASS_OUTPUT_TYPE_PARAMETER': 3
                }, context=context, feedback=feedback)
                
                basin_vector = basin_vector_result['output']
                temp_basin_layer = QgsVectorLayer(basin_vector, f'basin_{basin_id_counter}', 'ogr')

                # Get largest polygon if multiple
                if temp_basin_layer.featureCount() > 1:
                    max_area = 0
                    largest_feature = None
                    for feature in temp_basin_layer.getFeatures():
                        area = feature.geometry().area()
                        if area > max_area:
                            max_area = area
                            largest_feature = feature
                    if largest_feature:
                        temp_basin_layer = QgsVectorLayer("Polygon?crs=" + temp_basin_layer.crs().authid(), f"largest_polygon_{basin_id_counter}", "memory")
                        temp_basin_layer.dataProvider().addFeatures([largest_feature])
                        temp_basin_layer.updateExtents()
                
                # Smooth basin
                smoothed_basin = processing.run('native:smoothgeometry', {
                    'INPUT': temp_basin_layer,
                    'ITERATIONS': smooth_iterations,
                    'OFFSET': smooth_offset,
                    'MAX_ANGLE': 180,
                    'OUTPUT': 'memory:'
                }, context=context, feedback=feedback)['OUTPUT']
                
                # Add to basin layer
                for feature in smoothed_basin.getFeatures():
                    new_basin_feature = QgsFeature()
                    new_basin_feature.setGeometry(feature.geometry())
                    new_basin_feature.setFields(basin_fields)
                    
                    # Copy attributes from pour point
                    for i, field in enumerate(pour_points.fields()):
                        new_basin_feature.setAttribute(i, point_feature.attribute(field.name()))
                    
                    new_basin_feature.setAttribute("basin_id", basin_id_counter)
                    basin_layer.dataProvider().addFeatures([new_basin_feature])

                # Clip streams to basin
                try:
                    clipped_stream = processing.run('native:clip', {
                        'INPUT': stream_layer,
                        'OVERLAY': smoothed_basin,
                        'OUTPUT': f'memory:clipped_stream_{basin_id_counter}'
                    }, context=context, feedback=feedback)['OUTPUT']

                    # Add to stream output layer
                    for feature in clipped_stream.getFeatures():
                        new_stream_feature = QgsFeature()
                        new_stream_feature.setGeometry(feature.geometry())
                        new_stream_feature.setFields(stream_fields)
                        
                        # Copy original attributes
                        for field in stream_layer.fields():
                            new_stream_feature.setAttribute(field.name(), feature.attribute(field.name()))
                            
                        new_stream_feature.setAttribute("basin_id", basin_id_counter)
                        stream_output_layer.dataProvider().addFeatures([new_stream_feature])
                
                except Exception as e:
                    feedback.reportError(f"Error clipping streams for basin {basin_id_counter}: {str(e)}")
                
                basin_id_counter += 1

            basin_layer.updateExtents()
            stream_output_layer.updateExtents()
            
            return basin_layer, stream_output_layer
            
        except Exception as e:
            feedback.reportError(f"Error in watershed delineation: {str(e)}")
            raise

    def calculate_drainage_areas(self, basin_layer, context, feedback):
        """Calculate drainage areas in square miles and square meters"""
        try:
            basin_layer.startEditing()
            
            # Add Area fields if they don't exist
            fields = basin_layer.fields()
            if fields.indexFromName("SqMi") == -1:
                basin_layer.dataProvider().addAttributes([QgsField("SqMi", QVariant.Double)])
            if fields.indexFromName("SqM") == -1:
                basin_layer.dataProvider().addAttributes([QgsField("SqM", QVariant.Double)])
            
            basin_layer.updateFields()
            
            # Calculate area for each feature
            for feature in basin_layer.getFeatures():
                area_layer_units = feature.geometry().area()
                
                # Convert to square meters first (base unit)
                crs = basin_layer.crs()
                if crs.mapUnits() == QgsUnitTypes.DistanceMeters:
                    area_sq_meters = area_layer_units
                elif crs.mapUnits() == QgsUnitTypes.DistanceFeet:
                    area_sq_meters = area_layer_units * 0.092903
                elif crs.mapUnits() == QgsUnitTypes.DistanceDegrees:
                    # Approximate conversion for degrees near equator, but better to warn user
                    # Ideally should reproject, but for now using simple approximation
                    area_sq_meters = area_layer_units * 111319.9 * 111319.9
                else:
                    area_sq_meters = area_layer_units
                
                area_sq_miles = area_sq_meters / 2589988.11
                
                feature.setAttribute("SqMi", round(area_sq_miles, 4))
                feature.setAttribute("SqM", round(area_sq_meters, 2))
                basin_layer.updateFeature(feature)
            
            basin_layer.commitChanges()
            
            feedback.pushInfo(f"Calculated drainage areas for {basin_layer.featureCount()} basins")
            return basin_layer
            
        except Exception as e:
            basin_layer.rollBack()
            feedback.reportError(f"Error calculating drainage areas: {str(e)}")
            raise

    def name(self):
        return 'multiplebasindelineation'

    def displayName(self):
        return self.tr('Multiple Basin Delineation by Points')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'arcgeekcalculator'

    def shortHelpString(self):
        return self.tr("""
        <h3>Multiple Basin Delineation by Points</h3>
        
        <p>This tool performs a complete watershed analysis workflow in one step:</p>
        
        <ol>
        <li><b>Generates stream network</b> from DEM using flow accumulation</li>
        <li><b>Snaps input points</b> to the nearest stream network locations</li>
        <li><b>Delineates watersheds</b> using snapped points as pour points</li>
        <li><b>Calculates drainage areas</b> in square miles for each basin</li>
        </ol>
        
        <h4>Parameters:</h4>
        <ul>
        <li><b>Input DEM:</b> Digital Elevation Model (30m resolution recommended)</li>
        <li><b>Input Point Shapefile:</b> Point features to use as watershed pour points</li>
        <li><b>Flow Accumulation Threshold:</b> Minimum cells to form a stream (default: 50)</li>
        <li><b>Smooth Iterations:</b> Number of smoothing passes (default: 1)</li>
        <li><b>Smooth Offset:</b> Smoothing offset value (default: 0.25)</li>
        <li><b>Process All Points:</b> Whether to process all points or limit the number</li>
        <li><b>Maximum Points:</b> If not processing all, maximum number of points to process (default: 5)</li>
        </ul>
        
        <h4>Outputs:</h4>
        <ul>
        <li><b>Global Stream Network:</b> Generated stream network with Strahler/Shreve orders</li>
        <li><b>Snapped Pour Points:</b> Input points snapped to stream network for watershed delineation</li>
        <li><b>Watershed Basins:</b> Delineated basins with drainage areas in square miles</li>
        <li><b>Local Stream Network:</b> Stream network clipped to each basin</li>
        </ul>
        
        <p><i>Note: This tool requires GRASS GIS to be properly configured in QGIS.</i></p>
        
        <p><b>Developed by the Anthropocene Engineering team (anthroeng.com)</b></p>
        """)

    def createInstance(self):
        return MultipleBasinDelineationAlgorithm()

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)
