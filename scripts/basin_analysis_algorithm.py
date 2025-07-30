from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessing, QgsFeatureSink, QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource, QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterRasterLayer, QgsProcessingParameterField,
                       QgsVectorLayer, QgsRasterLayer, QgsFeature, QgsGeometry, QgsField,
                       QgsWkbTypes, QgsRasterBandStats, QgsPoint, QgsPointXY,
                       QgsFields, QgsProcessingParameterNumber, QgsProcessingUtils,
                       QgsProcessingParameterFileDestination, QgsProcessingParameterEnum)
import processing
import math
import os
import webbrowser
from .basin_processes import calculate_parameters, get_basin_area_interpretation, get_mean_slope_interpretation
from .hypsometric_curve import generate_hypsometric_curve

class BasinAnalysisAlgorithm(QgsProcessingAlgorithm):
    INPUT_BASIN = 'INPUT_BASIN'
    INPUT_STREAMS = 'INPUT_STREAMS'
    INPUT_DEM = 'INPUT_DEM'
    STREAM_ORDER_FIELD = 'STREAM_ORDER_FIELD'
    PRECISION = 'PRECISION'
    HYPSOMETRIC_METHOD = 'HYPSOMETRIC_METHOD'
    OUTPUT = 'OUTPUT'
    OUTPUT_HYPSOMETRIC = 'OUTPUT_HYPSOMETRIC'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT_BASIN, 'Basin layer', [QgsProcessing.TypeVectorPolygon]))
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT_STREAMS, 'Stream network', [QgsProcessing.TypeVectorLine]))
        self.addParameter(QgsProcessingParameterField(self.STREAM_ORDER_FIELD, 'Field containing stream order (Strahler)', optional=False, parentLayerParameterName=self.INPUT_STREAMS))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_DEM, 'Digital Elevation Model'))
        self.addParameter(QgsProcessingParameterNumber(self.PRECISION, 'Decimal precision', type=QgsProcessingParameterNumber.Integer, minValue=0, maxValue=15, defaultValue=4))
        self.addParameter(QgsProcessingParameterEnum(
            self.HYPSOMETRIC_METHOD,
            'Hypsometric calculation method',
            options=['QGIS Algorithm (Fast)', 'Pixel-by-Pixel (Precise)'],
            defaultValue=0
        ))
        
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, 'Morphometric Report', QgsProcessing.TypeVector))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_HYPSOMETRIC,
                'Hypsometric Curve Output',
                'HTML files (*.html)',
                defaultValue=None,
                optional=True
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        try:
            basin_layer = self.parameterAsVectorLayer(parameters, self.INPUT_BASIN, context)
            streams_layer = self.parameterAsVectorLayer(parameters, self.INPUT_STREAMS, context)
            dem_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
            stream_order_field = self.parameterAsString(parameters, self.STREAM_ORDER_FIELD, context)
            precision = self.parameterAsInt(parameters, self.PRECISION, context)
            method = self.parameterAsEnum(parameters, self.HYPSOMETRIC_METHOD, context)
            output_hypsometric = self.parameterAsFileOutput(parameters, self.OUTPUT_HYPSOMETRIC, context)

            if not basin_layer.isValid() or not streams_layer.isValid() or not dem_layer.isValid():
                feedback.reportError('One or more input layers are invalid')
                return {}

            if basin_layer.crs() != streams_layer.crs() or basin_layer.crs() != dem_layer.crs():
                feedback.reportError('Input layers have different Coordinate Reference Systems (CRS)')
                return {}

            feedback.pushInfo('Processing morphometric analysis...')

            dem_clipped = self.clip_dem_by_basin(dem_layer, basin_layer, context, feedback)
            slope_layer = self.calculate_slope(dem_clipped, context, feedback)
            slope_stats = self.get_slope_statistics(slope_layer, context, feedback)
            
            mean_slope_degrees = slope_stats['MEAN']
            mean_slope_percent = math.tan(math.radians(mean_slope_degrees)) * 100

            pour_point, upstream_point, downstream_point = self.calculate_pour_point(streams_layer, stream_order_field)
            
            results = calculate_parameters(basin_layer, streams_layer, dem_clipped, pour_point, stream_order_field, mean_slope_degrees, feedback)
            
            if results is None:
                feedback.reportError("Failed to calculate basin parameters")
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

            hypsometric_output_dir = os.path.dirname(output_hypsometric) if output_hypsometric else QgsProcessingUtils.tempFolder()
            hypsometric_results = generate_hypsometric_curve(dem_clipped, basin_layer, hypsometric_output_dir, feedback, method)

            if hypsometric_results and 'HI' in hypsometric_results and 'STAGE' in hypsometric_results:
                hi_feature = QgsFeature()
                hi_feature.setFields(fields)
                hi_feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(pour_point)))
                hi_feature.setAttribute("Parameter", "Hypsometric Integral (HI)")
                hi_feature.setAttribute("Value", round(hypsometric_results['HI'], precision))
                hi_feature.setAttribute("Unit", "dimensionless")
                hi_feature.setAttribute("Interpretation", hypsometric_results['STAGE'])
                sink.addFeature(hi_feature, QgsFeatureSink.FastInsert)

            if hypsometric_results and 'HTML' in hypsometric_results and hypsometric_results['HTML']:
                try:
                    html_path = hypsometric_results['HTML']
                    url = 'file://' + os.path.abspath(html_path)
                    webbrowser.open(url)
                except Exception as e:
                    feedback.pushInfo(f"Could not open HTML report automatically: {str(e)}")

            feedback.pushInfo('Analysis completed successfully')

            return {
                self.OUTPUT: dest_id,
                self.OUTPUT_HYPSOMETRIC: hypsometric_results.get('HTML', None) if hypsometric_results else None
            }

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
        main_channel_segments = [f.geometry() for f in streams_layer.getFeatures() if f[stream_order_field] == max_order]
        main_channel = QgsGeometry.unaryUnion(main_channel_segments)

        if main_channel.isMultipart():
            main_channel = main_channel.mergeLines()

        vertices = main_channel.asPolyline()
        upstream_point = vertices[0]
        downstream_point = vertices[-1]
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
            Hypsometric calculation method: Choose between QGIS Algorithm (fast) or Pixel-by-Pixel (precise)

        Outputs:
            Output Report: A table with calculated morphometric parameters and their interpretations
            Hypsometric Curve Output: HTML file containing the hypsometric curve visualization

        Note: All input layers must have the same Coordinate Reference System (CRS).
        """

    def createInstance(self):
        return BasinAnalysisAlgorithm()