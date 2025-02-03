import os
import tempfile
import processing
from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessing, QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterNumber, QgsProcessingParameterFeatureSink,
                       QgsVectorLayer, QgsRasterLayer, QgsField, QgsWkbTypes,
                       QgsProcessingException, QgsFeatureSink, QgsSpatialIndex, QgsRasterLayer,
                       QgsCoordinateReferenceSystem, QgsRectangle, QgsFeature, QgsGeometry,
                       QgsMessageLog, Qgis)
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class WatershedAnalysisAlgorithm(QgsProcessingAlgorithm):
    INPUT_DEM = 'INPUT_DEM'
    THRESHOLD = 'THRESHOLD'
    OUTPUT_STREAMS = 'OUTPUT_STREAMS'
    SMOOTH_ITERATIONS = 'SMOOTH_ITERATIONS'
    SMOOTH_OFFSET = 'SMOOTH_OFFSET'
    MAX_RASTER_SIZE = 100000000  # 100 million cells, adjust as needed

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_DEM, 'Input DEM'))
        self.addParameter(QgsProcessingParameterNumber(self.THRESHOLD, 'Flow Accumulation Threshold', 
                                                       type=QgsProcessingParameterNumber.Integer, minValue=0, defaultValue=5000))
        self.addParameter(QgsProcessingParameterNumber(self.SMOOTH_ITERATIONS, 'Smooth Iterations',
                                                       type=QgsProcessingParameterNumber.Integer, minValue=1, defaultValue=1))
        self.addParameter(QgsProcessingParameterNumber(self.SMOOTH_OFFSET, 'Smooth Offset',
                                                       type=QgsProcessingParameterNumber.Double, minValue=0, maxValue=0.5, defaultValue=0.25))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT_STREAMS, 'Output Stream Network'))

    def processAlgorithm(self, parameters, context, feedback):
        try:
            dem = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
            if not dem.isValid():
                raise QgsProcessingException(self.tr('Invalid input DEM'))
            
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
            
            threshold = self.parameterAsInt(parameters, self.THRESHOLD, context)
            smooth_iterations = self.parameterAsInt(parameters, self.SMOOTH_ITERATIONS, context)
            smooth_offset = self.parameterAsDouble(parameters, self.SMOOTH_OFFSET, context)
            
            # Use a temporary directory with a simple path
            temp_dir = tempfile.mkdtemp(prefix='qgis_temp_')
            
            filled_dem = processing.run("grass7:r.fill.dir", {
                'input': resampled_dem,
                'output': QgsProcessing.TEMPORARY_OUTPUT,
                'direction': QgsProcessing.TEMPORARY_OUTPUT,
                'areas': QgsProcessing.TEMPORARY_OUTPUT,
                'format': 0
            }, context=context, feedback=feedback)['output']
            
            flow_accumulation = processing.run("grass7:r.watershed", {
                'elevation': filled_dem,
                'accumulation': QgsProcessing.TEMPORARY_OUTPUT,
                'drainage': QgsProcessing.TEMPORARY_OUTPUT,
                'threshold': resampled_dem.rasterUnitsPerPixelX(),
                '-s': True,
                '-m': True
            }, context=context, feedback=feedback)['accumulation']
            
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
            
            ordered_streams = self.calculate_stream_orders(smoothed_streams, context, feedback)
            
            stream_sink, stream_dest_id = self.parameterAsSink(parameters, self.OUTPUT_STREAMS, context,
                                                               ordered_streams.fields(), QgsWkbTypes.LineString, dem.crs())
            if stream_sink is None:
                raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT_STREAMS))
            
            feature_count = ordered_streams.featureCount()
            for current, f in enumerate(ordered_streams.getFeatures()):
                if feedback.isCanceled():
                    break
                stream_sink.addFeature(f, QgsFeatureSink.FastInsert)
                feedback.setProgress(int((current + 1) / feature_count * 100))
            
            return {self.OUTPUT_STREAMS: stream_dest_id}
        
        except Exception as e:
            QgsMessageLog.logMessage(f"Error in processAlgorithm: {str(e)}", level=Qgis.Critical)
            raise

    def resample_dem(self, dem, new_cell_size, context, feedback):
        try:
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
        except Exception as e:
            QgsMessageLog.logMessage(f"Error in resample_dem: {str(e)}", level=Qgis.Critical)
            raise

    def calculate_stream_orders(self, stream_layer, context, feedback):
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
            QgsMessageLog.logMessage(f"Error in calculate_stream_orders: {str(e)}", level=Qgis.Critical)
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

    def name(self):
        return 'watershedanalysiswithsmooth'

    def displayName(self):
        return self.tr('Stream Network with Order')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'arcgeekcalculator'

    def shortHelpString(self):
        return self.tr("""
        This algorithm generates a stream network, applies smoothing, and calculates both Strahler and Shreve stream orders.
        It uses GRASS GIS algorithms for stream extraction, QGIS native smooth algorithm, and implements custom Strahler and Shreve order calculations.
        
        For large, high-resolution DEMs, the algorithm may automatically resample the input to a lower resolution to enable processing.
        
        Parameters:
            Input DEM: A raster layer representing the terrain elevation
            Flow Accumulation Threshold: Minimum number of cells to form a stream
            Smooth Iterations: Number of smoothing iterations to apply
            Smooth Offset: Controls the smoothness of the output
        Outputs:
            Output Stream Network: A line layer representing the smoothed stream network with Strahler and Shreve orders
        """)

    def createInstance(self):
        return WatershedAnalysisAlgorithm()

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)