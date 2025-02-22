from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer, 
                       QgsProcessingParameterFeatureSink, QgsProcessingParameterPoint, 
                       QgsWkbTypes, QgsField, QgsProcessingUtils, QgsFields, 
                       QgsVectorLayer, QgsProject, QgsFeatureSink, QgsProcessing, QgsFeature,
                       QgsProcessingParameterVectorLayer, QgsProcessingException,
                       QgsProcessingParameterNumber, QgsRasterLayer, QgsSnappingConfig)
from qgis.PyQt.QtCore import QVariant, QCoreApplication
from qgis.utils import iface
import processing

class WatershedBasinDelineationAlgorithm(QgsProcessingAlgorithm):
    INPUT_DEM = 'INPUT_DEM'
    POUR_POINT = 'POUR_POINT'
    INPUT_STREAM = 'INPUT_STREAM'
    OUTPUT_BASIN = 'OUTPUT_BASIN'
    OUTPUT_STREAM = 'OUTPUT_STREAM'
    SMOOTH_ITERATIONS = 'SMOOTH_ITERATIONS'
    SMOOTH_OFFSET = 'SMOOTH_OFFSET'
    MAX_RASTER_SIZE = 100000000  # 100 million cells, adjust as needed

    def __init__(self):
        super().__init__()
        self.activate_snapping()

    def activate_snapping(self):
        """Enable snapping to vertices and segments"""
        snapping_config = iface.mapCanvas().snappingUtils().config()
        snapping_config.setEnabled(True)
        
        # Handle different QGIS versions
        try:
            # For QGIS 3.38 and above
            snapping_config.setTypeFlag(QgsSnappingConfig.VertexFlag | QgsSnappingConfig.SegmentFlag)
        except TypeError:
            # For QGIS 3.34 and below
            snapping_config.setType(QgsSnappingConfig.Vertex | QgsSnappingConfig.Segment)
        
        iface.mapCanvas().snappingUtils().setConfig(snapping_config)

    def deactivate_snapping(self):
        """Disable snapping when tool window closes"""
        snapping_config = iface.mapCanvas().snappingUtils().config()
        snapping_config.setEnabled(False)
        
        # Handle different QGIS versions
        try:
            # For QGIS 3.38 and above
            snapping_config.setTypeFlag(QgsSnappingConfig.VertexFlag | QgsSnappingConfig.SegmentFlag)
        except TypeError:
            # For QGIS 3.34 and below
            snapping_config.setType(QgsSnappingConfig.Vertex | QgsSnappingConfig.Segment)
        
        iface.mapCanvas().snappingUtils().setConfig(snapping_config)

    def canCancel(self):
        """Should be set to True for algorithms that can be cancelled"""
        return True
        
    def onClose(self):
        """Handles cleanup when algorithm window is closed"""
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
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT_BASIN, 'Output Basin', QgsProcessing.TypeVectorPolygon))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT_STREAM, 'Output Basin Stream Network', QgsProcessing.TypeVectorLine, optional=True))

    def processAlgorithm(self, parameters, context, feedback):
        dem = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
        pour_point = self.parameterAsPoint(parameters, self.POUR_POINT, context)
        input_stream = self.parameterAsVectorLayer(parameters, self.INPUT_STREAM, context)
        smooth_iterations = self.parameterAsInt(parameters, self.SMOOTH_ITERATIONS, context)
        smooth_offset = self.parameterAsDouble(parameters, self.SMOOTH_OFFSET, context)

        if not dem.isValid():
            raise QgsProcessingException(self.tr('Invalid input DEM'))

        if input_stream and input_stream.geometryType() != QgsWkbTypes.LineGeometry:
            raise QgsProcessingException(self.tr('Input Stream Network must be a line layer'))

        # Check and resample DEM if necessary
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
            raise QgsProcessingException(self.tr('Input DEM is too large to process efficiently even after resampling. '
                                                 'Please use a smaller area or lower resolution DEM.'))

        # Step 1: Fill sinks
        filled_dem = processing.run('grass7:r.fill.dir', {
            'input': resampled_dem,
            'format': 0,
            'output': 'TEMPORARY_OUTPUT',
            'direction': 'TEMPORARY_OUTPUT',
            'areas': 'TEMPORARY_OUTPUT'
        }, context=context, feedback=feedback)['output']

        # Step 2: Calculate flow direction and accumulation
        watershed_result = processing.run('grass7:r.watershed', {
            'elevation': filled_dem,
            'convergence': 5,
            'memory': 300,
            '-s': True,
            'accumulation': 'TEMPORARY_OUTPUT',
            'drainage': 'TEMPORARY_OUTPUT'
        }, context=context, feedback=feedback)
        
        drainage = watershed_result['drainage']

        # Step 3: Delineate watershed
        pour_point_str = f'{pour_point.x()},{pour_point.y()}'
        basin_raster = processing.run('grass7:r.water.outlet', {
            'input': drainage,
            'coordinates': pour_point_str,
            'output': 'TEMPORARY_OUTPUT'
        }, context=context, feedback=feedback)['output']

        # Step 4: Convert raster basin to vector
        basin_vector_result = processing.run('grass7:r.to.vect', {
            'input': basin_raster,
            'type': 2,  # area
            'column': 'value',
            '-s': True,
            'output': 'TEMPORARY_OUTPUT',
            'GRASS_OUTPUT_TYPE_PARAMETER': 3  # auto
        }, context=context, feedback=feedback)
        
        basin_vector = basin_vector_result['output']  # GRASS uses 'output' instead of 'OUTPUT'
        feedback.pushInfo('Converting raster basin to vector completed')

        # Step 4.1: Keep only the largest polygon if there are multiple polygons
        basin_layer = QgsVectorLayer(basin_vector, 'basin', 'ogr')

        if basin_layer.featureCount() > 1:
            feedback.pushInfo(f'Found {basin_layer.featureCount()} polygons. Keeping only the largest one.')
    
            # Calculate the area for each polygon and find the largest
            max_area = 0
            largest_feature = None
    
            for feature in basin_layer.getFeatures():
                area = feature.geometry().area()
                if area > max_area:
                    max_area = area
                    largest_feature = feature
    
            if largest_feature:
                # Create a new layer with only the largest feature
                largest_polygon = QgsVectorLayer("Polygon?crs=" + basin_layer.crs().authid(), "largest_polygon", "memory")
                provider = largest_polygon.dataProvider()
                provider.addFeatures([largest_feature])
                largest_polygon.updateExtents()
        
                feedback.pushInfo(f'Selected largest polygon with area: {max_area:.2f} square units')
            else:
                feedback.pushWarning('No valid features found - using original basin')
                largest_polygon = basin_vector
        else:
            largest_polygon = basin_vector
            feedback.pushInfo('Single polygon found - no need to filter')

        # Step 5: Apply smoothing to the basin
        smoothed_basin = processing.run('native:smoothgeometry', {
            'INPUT': largest_polygon,
            'ITERATIONS': smooth_iterations,
            'OFFSET': smooth_offset,
            'MAX_ANGLE': 180,
            'OUTPUT': 'memory:'
        }, context=context, feedback=feedback)['OUTPUT']

        # Load the vector layer
        basin_layer = smoothed_basin  # smoothed_basin is already a QgsVectorLayer

        # Save the basin result
        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT_BASIN, context,
                                               basin_layer.fields(), QgsWkbTypes.Polygon, basin_layer.crs())
        
        if sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT_BASIN))

        features = basin_layer.getFeatures()
        for feature in features:
            sink.addFeature(feature, QgsFeatureSink.FastInsert)

        results = {self.OUTPUT_BASIN: dest_id}

        # Process input stream network if provided
        if input_stream:
            clipped_stream = processing.run('native:clip', {
                'INPUT': input_stream,
                'OVERLAY': basin_layer,
                'OUTPUT': 'memory:'
            }, context=context, feedback=feedback)['OUTPUT']

            # Save the clipped stream result
            (stream_sink, stream_dest_id) = self.parameterAsSink(parameters, self.OUTPUT_STREAM, context,
                                                                 clipped_stream.fields(), QgsWkbTypes.LineString, clipped_stream.crs())
            
            if stream_sink is None:
                raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT_STREAM))

            stream_features = clipped_stream.getFeatures()
            for feature in stream_features:
                stream_sink.addFeature(feature, QgsFeatureSink.FastInsert)

            results[self.OUTPUT_STREAM] = stream_dest_id

        return results

    def resample_dem(self, dem, new_cell_size, context, feedback):
        extent = dem.extent()
        width = int(extent.width() / new_cell_size)
        height = int(extent.height() / new_cell_size)
        
        resampled = processing.run("gdal:warpreproject", {
            'INPUT': dem,
            'SOURCE_CRS': dem.crs(),
            'TARGET_CRS': dem.crs(),
            'RESAMPLING': 0,  # Nearest neighbor
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
        
        For large, high-resolution DEMs, the algorithm may automatically resample the input to a lower resolution to enable processing.
        
        Parameters:
            Input DEM: A raster layer representing the terrain elevation
            Pour Point: The outlet point of the watershed. The tool automatically enables snapping to streams
            Input Stream Network: Optional. A line vector layer representing the stream network
            Smoothing Iterations: Number of iterations for smoothing the basin boundary (0-10)
            Smoothing Offset: Offset value for smoothing (0.0-0.5)
        
        Outputs:
            Output Basin: A polygon layer representing the delineated watershed basin
            Output Basin Stream Network: Optional. A line layer representing the clipped stream network within the basin
        
        Note: The accuracy of the watershed delineation depends on the resolution and quality of the input DEM.
        """)

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)