from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessing, QgsFeatureSink, QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource, QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterRasterLayer, QgsProcessingParameterField,
                       QgsVectorLayer, QgsRasterLayer, QgsFeature, QgsGeometry, QgsField,
                       QgsWkbTypes, QgsRasterBandStats, QgsPoint, QgsPointXY,
                       QgsFields, QgsProcessingParameterNumber, QgsProcessingUtils)
import processing
import math
import os
from .basin_processes import calculate_parameters, get_basin_area_interpretation, get_mean_slope_interpretation
from .hypsometric_curve import generate_hypsometric_curve

class BasinAnalysisAlgorithm(QgsProcessingAlgorithm):
    INPUT_BASIN = 'INPUT_BASIN'
    INPUT_STREAMS = 'INPUT_STREAMS'
    INPUT_DEM = 'INPUT_DEM'
    STREAM_ORDER_FIELD = 'STREAM_ORDER_FIELD'
    PRECISION = 'PRECISION'
    OUTPUT = 'OUTPUT'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT_BASIN, 'Basin layer', [QgsProcessing.TypeVectorPolygon]))
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT_STREAMS, 'Stream network', [QgsProcessing.TypeVectorLine]))
        self.addParameter(QgsProcessingParameterField(self.STREAM_ORDER_FIELD, 'Field containing stream order (Strahler)', optional=False, parentLayerParameterName=self.INPUT_STREAMS))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_DEM, 'Digital Elevation Model'))
        self.addParameter(QgsProcessingParameterNumber(self.PRECISION, 'Decimal precision', type=QgsProcessingParameterNumber.Integer, minValue=0, maxValue=15, defaultValue=4))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, 'Output Report', QgsProcessing.TypeVector))

    def processAlgorithm(self, parameters, context, feedback):
        try:
            basin_layer = self.parameterAsVectorLayer(parameters, self.INPUT_BASIN, context)
            streams_layer = self.parameterAsVectorLayer(parameters, self.INPUT_STREAMS, context)
            dem_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
            stream_order_field = self.parameterAsString(parameters, self.STREAM_ORDER_FIELD, context)
            precision = self.parameterAsInt(parameters, self.PRECISION, context)

            feedback.pushInfo(f"Basin layer: {basin_layer.name()}")
            feedback.pushInfo(f"Streams layer: {streams_layer.name()}")
            feedback.pushInfo(f"DEM layer: {dem_layer.name()}")
            feedback.pushInfo(f"Stream order field: {stream_order_field}")

            if not basin_layer.isValid() or not streams_layer.isValid() or not dem_layer.isValid():
                feedback.reportError('One or more input layers are invalid')
                return {}

            if basin_layer.crs() != streams_layer.crs() or basin_layer.crs() != dem_layer.crs():
                feedback.reportError('Input layers have different Coordinate Reference Systems (CRS). Please ensure all layers have the same CRS.')
                return {}

            dem_clipped = self.clip_dem_by_basin(dem_layer, basin_layer, context, feedback)
            slope_layer = self.calculate_slope(dem_clipped, context, feedback)
            slope_stats = self.get_slope_statistics(slope_layer, context, feedback)
            
            mean_slope_degrees = slope_stats['MEAN']
            mean_slope_percent = math.tan(math.radians(mean_slope_degrees)) * 100

            feedback.pushInfo(f"Mean slope (degrees): {mean_slope_degrees}")
            feedback.pushInfo(f"Mean slope (percent): {mean_slope_percent}")

            # Calculate the pour point (upstream point of the main channel)
            pour_point, upstream_point, downstream_point = self.calculate_pour_point(streams_layer, stream_order_field)
            
            feedback.pushInfo(f"Pour point: {pour_point.asWkt()}")
            feedback.pushInfo(f"Upstream point: {upstream_point.asWkt()}")
            feedback.pushInfo(f"Downstream point: {downstream_point.asWkt()}")

            results = calculate_parameters(basin_layer, streams_layer, dem_clipped, pour_point, stream_order_field, mean_slope_degrees, feedback)
            
            if results is None:
                feedback.reportError("Failed to calculate basin parameters.")
                return {}

            fields = QgsFields()
            fields.append(QgsField("Parameter", QVariant.String))
            fields.append(QgsField("Value", QVariant.Double))
            fields.append(QgsField("Unit", QVariant.String))
            fields.append(QgsField("Interpretation", QVariant.String))

            sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context, fields, QgsWkbTypes.Point, basin_layer.crs())

            for param, details in results.items():
                feature = QgsFeature()
                feature.setFields(fields)
                
                feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(pour_point)))
                
                feature.setAttribute("Parameter", param)
                feature.setAttribute("Value", round(details['value'], precision))
                feature.setAttribute("Unit", details['unit'])
                feature.setAttribute("Interpretation", details['interpretation'])
                sink.addFeature(feature, QgsFeatureSink.FastInsert)

            feedback.pushInfo("Output report generated successfully.")

            # Generate hypsometric curve
            temp_output_folder = QgsProcessingUtils.tempFolder()
            hypsometric_results = generate_hypsometric_curve(dem_clipped, basin_layer, temp_output_folder, feedback)

            # Add Hypsometric Integral to the results table
            if hypsometric_results and 'HI' in hypsometric_results and 'STAGE' in hypsometric_results:
                hi_feature = QgsFeature()
                hi_feature.setFields(fields)
                hi_feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(pour_point)))
                
                hi_feature.setAttribute("Parameter", "Hypsometric Integral (HI)")
                hi_feature.setAttribute("Value", round(hypsometric_results['HI'], precision))
                hi_feature.setAttribute("Unit", "dimensionless")
                hi_feature.setAttribute("Interpretation", hypsometric_results['STAGE'])
                sink.addFeature(hi_feature, QgsFeatureSink.FastInsert)

            # Create clickable links to the output files
            for file_type, file_path in hypsometric_results.items():
                if file_path:
                    feedback.pushInfo(f"{file_type}: {file_path}")

            feedback.pushInfo(f"Hypsometric curve analysis completed. Results saved in: {temp_output_folder}")

            return {self.OUTPUT: dest_id}

        except Exception as e:
            feedback.reportError(f"An error occurred: {str(e)}")
            import traceback
            feedback.pushInfo(traceback.format_exc())
            return {}

    def clip_dem_by_basin(self, dem_layer, basin_layer, context, feedback):
        params = {
            'ALPHA_BAND': False,
            'CROP_TO_CUTLINE': True,
            'KEEP_RESOLUTION': False,
            'INPUT': dem_layer,
            'MASK': basin_layer,
            'NODATA': None,
            'OPTIONS': '',
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }
        result = processing.run("gdal:cliprasterbymasklayer", params, context=context, feedback=feedback)
        return QgsRasterLayer(result['OUTPUT'], 'Clipped DEM')

    def calculate_pour_point(self, streams_layer, stream_order_field):
        max_order = max([f[stream_order_field] for f in streams_layer.getFeatures()])
        
        # Get all segments of the main channel
        main_channel_segments = [f.geometry() for f in streams_layer.getFeatures() if f[stream_order_field] == max_order]

        # Merge all segments into a single line
        main_channel = QgsGeometry.unaryUnion(main_channel_segments)

        # Ensure the result is a single line
        if main_channel.isMultipart():
            main_channel = main_channel.mergeLines()

        # Get the start and end points
        vertices = main_channel.asPolyline()
        upstream_point = vertices[0]
        downstream_point = vertices[-1]

        # The pour point is typically the downstream point
        pour_point = downstream_point

        return pour_point, upstream_point, downstream_point

    def calculate_slope(self, dem_layer, context, feedback):
        params = {
            'INPUT': dem_layer,
            'OUTPUT': 'TEMPORARY_OUTPUT',
            'Z_FACTOR': 1
        }
        result = processing.run("gdal:slope", params, context=context, feedback=feedback)
        return QgsRasterLayer(result['OUTPUT'], 'Slope')

    def get_slope_statistics(self, slope_layer, context, feedback):
        params = {
            'BAND': 1,
            'INPUT': slope_layer,
            'OUTPUT_HTML_FILE': 'TEMPORARY_OUTPUT'
        }
        return processing.run("qgis:rasterlayerstatistics", params, context=context, feedback=feedback)

    def name(self):
        return 'basinanalysis'

    def displayName(self):
        return 'Watershed Morphometric Analysis'

    def group(self):
        return 'ArcGeek Calculator'

    def groupId(self):
        return 'arcgeekcalculator'

    def shortHelpString(self):
        return """
        This algorithm performs a comprehensive analysis of a hydrological basin.
        It calculates various morphometric parameters and provides interpretations.

        Parameters:
            Basin layer: A polygon layer representing the basin boundary
            Stream network: A line layer representing the stream network within the basin
            Stream Order Field: Field containing stream order (Strahler)
            Digital Elevation Model: A raster layer representing the terrain elevation
            Decimal precision: Number of decimal places for the results (default: 4)

        Outputs:
            A table with calculated morphometric parameters and their interpretations

        Note: All input layers must have the same Coordinate Reference System (CRS).
        """

    def createInstance(self):
        return BasinAnalysisAlgorithm()