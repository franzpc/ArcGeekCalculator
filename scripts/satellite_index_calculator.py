from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing, QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterEnum, QgsProcessingParameterRasterDestination,
                       QgsRasterLayer, QgsProcessingException, QgsProcessingOutputString,
                       QgsProcessingContext, QgsRasterBandStats)
from osgeo import gdal
import math

class SatelliteIndexCalculatorAlgorithm(QgsProcessingAlgorithm):
    SATELLITE_TYPE = 'SATELLITE_TYPE'
    INDEX_TYPE = 'INDEX_TYPE'
    INPUT_BAND_HIGHER = 'INPUT_BAND_HIGHER'
    INPUT_BAND_LOWER = 'INPUT_BAND_LOWER'
    OUTPUT_RASTER = 'OUTPUT_RASTER'
    INFO_MESSAGE = 'INFO_MESSAGE'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterEnum(
            self.SATELLITE_TYPE,
            self.tr('Satellite type'),
            options=['Sentinel-2', 'Landsat-7', 'Landsat-8/9'],
            defaultValue=0
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.INDEX_TYPE,
            self.tr('Index type'),
            options=['NDVI', 'NDWI', 'NDBI', 'NDMI', 'MSI', 'NBR', 'AVI', 'SAVI'],
            defaultValue=0
        ))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_BAND_HIGHER, self.tr('Higher number band')))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_BAND_LOWER, self.tr('Lower number band')))
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT_RASTER, self.tr('Output index raster')))
        self.addOutput(QgsProcessingOutputString(self.INFO_MESSAGE, self.tr('Info Message')))

    def processAlgorithm(self, parameters, context, feedback):
        satellite_type = self.parameterAsEnum(parameters, self.SATELLITE_TYPE, context)
        index_type = self.parameterAsEnum(parameters, self.INDEX_TYPE, context)
        band_higher = self.parameterAsRasterLayer(parameters, self.INPUT_BAND_HIGHER, context)
        band_lower = self.parameterAsRasterLayer(parameters, self.INPUT_BAND_LOWER, context)
        output_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_RASTER, context)

        index_info = {
            'Sentinel-2': {
                'NDVI': {'bands': ('8', '4')},
                'NDWI': {'bands': ('8', '3')},
                'NDBI': {'bands': ('11', '8')},
                'NDMI': {'bands': ('8', '11')},
                'MSI': {'bands': ('11', '8')},
                'NBR': {'bands': ('8', '12')},
                'AVI': {'bands': ('8', '4')},
                'SAVI': {'bands': ('8', '4')}
            },
            'Landsat-7': {
                'NDVI': {'bands': ('4', '3')},
                'NDWI': {'bands': ('4', '2')},
                'NDBI': {'bands': ('5', '4')},
                'NDMI': {'bands': ('4', '5')},
                'MSI': {'bands': ('5', '4')},
                'NBR': {'bands': ('4', '7')},
                'AVI': {'bands': ('4', '3')},
                'SAVI': {'bands': ('4', '3')}
            },
            'Landsat-8/9': {
                'NDVI': {'bands': ('5', '4')},
                'NDWI': {'bands': ('5', '3')},
                'NDBI': {'bands': ('6', '5')},
                'NDMI': {'bands': ('5', '6')},
                'MSI': {'bands': ('6', '5')},
                'NBR': {'bands': ('5', '7')},
                'AVI': {'bands': ('5', '4')},
                'SAVI': {'bands': ('5', '4')}
            }
        }

        satellite_names = ['Sentinel-2', 'Landsat-7', 'Landsat-8/9']
        index_names = ['NDVI', 'NDWI', 'NDBI', 'NDMI', 'MSI', 'NBR', 'AVI', 'SAVI']
        
        selected_satellite = satellite_names[satellite_type]
        selected_index = index_names[index_type]
        
        required_bands = index_info[selected_satellite][selected_index]['bands']

        warning_message = f"Calculating {selected_index} for {selected_satellite}:\n"
        warning_message += f"Higher number band should be: Band {max(required_bands)}\n"
        warning_message += f"Lower number band should be: Band {min(required_bands)}\n"
        warning_message += f"Selected:\n"
        warning_message += f"Higher number band: {band_higher.name()}\n"
        warning_message += f"Lower number band: {band_lower.name()}\n"

        feedback.pushWarning(warning_message)

        try:
            feedback.pushInfo("Starting index calculation...")
            
            # Read raster data
            ds_higher = gdal.Open(band_higher.source())
            ds_lower = gdal.Open(band_lower.source())
            
            # Check if resolutions match
            if (ds_higher.RasterXSize != ds_lower.RasterXSize or 
                ds_higher.RasterYSize != ds_lower.RasterYSize):
                raise QgsProcessingException("Input bands have different resolutions. Please use bands with matching resolutions.")

            band_a = ds_higher.GetRasterBand(1)
            band_b = ds_lower.GetRasterBand(1)

            # Get raster dimensions
            width = band_a.XSize
            height = band_a.YSize

            # Create output raster
            driver = gdal.GetDriverByName('GTiff')
            outds = driver.Create(output_raster, width, height, 1, gdal.GDT_Float32)
            outds.SetGeoTransform(ds_higher.GetGeoTransform())
            outds.SetProjection(ds_higher.GetProjection())
            outband = outds.GetRasterBand(1)

            # Process data in chunks to reduce memory usage
            chunk_size = 1024
            for y in range(0, height, chunk_size):
                if feedback.isCanceled():
                    break
                win_height = min(chunk_size, height - y)
                feedback.setProgress(int((y / height) * 100))

                a_data = band_a.ReadAsArray(0, y, width, win_height).astype(float)
                b_data = band_b.ReadAsArray(0, y, width, win_height).astype(float)

                if selected_index in ['NDVI', 'NDWI', 'NDBI', 'NDMI', 'NBR']:
                    result = (a_data - b_data) / (a_data + b_data + 1e-10)  # Add small number to avoid division by zero
                elif selected_index == 'MSI':
                    result = a_data / (b_data + 1e-10)  # Add small number to avoid division by zero
                elif selected_index == 'AVI':
                    result = ((a_data + 1) * (1.0 - (b_data + 1) / 10000.0) * (a_data - b_data)) ** (1/3)
                elif selected_index == 'SAVI':
                    result = 1.5 * (a_data - b_data) / (a_data + b_data + 0.5)

                outband.WriteArray(result, 0, y)

            outband.FlushCache()
            outds = None  # Close the dataset

            feedback.pushInfo("Index calculation completed successfully.")
            
            output_name = f"{selected_satellite}_{selected_index}"
            context.addLayerToLoadOnCompletion(output_raster, QgsProcessingContext.LayerDetails(output_name, context.project(), self.OUTPUT_RASTER))

        except Exception as e:
            raise QgsProcessingException(f"An error occurred during index calculation: {str(e)}")

        return {self.OUTPUT_RASTER: output_raster, self.INFO_MESSAGE: warning_message}

    def name(self):
        return 'satellite_index_calculator'

    def displayName(self):
        return self.tr('Satellite Index Calculator')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'spectral_indices'

    def createInstance(self):
        return SatelliteIndexCalculatorAlgorithm()

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def shortHelpString(self):
        return self.tr("This algorithm calculates various two-band spectral indices from satellite imagery.\n\n"
                       "1. Choose the satellite type.\n"
                       "2. Select the index you want to calculate.\n"
                       "3. Select the two raster bands required for the index.\n\n"
                       "Available indices and their required bands:\n"
                       "- NDVI (NIR, Red)\n"
                       "- NDWI (Green, NIR)\n"
                       "- NDBI (SWIR, NIR)\n"
                       "- NDMI (NIR, SWIR)\n"
                       "- MSI (SWIR, NIR)\n"
                       "- NBR (NIR, SWIR2)\n"
                       "- AVI (NIR, Red)\n"
                       "- SAVI (NIR, Red)\n\n"
                       "Band numbers for each satellite:\n"
                       "Sentinel-2: \n"
                       " Green: 3, Red: 4, NIR: 8, SWIR: 11, SWIR2: 12\n"
                       "Landsat-7: \n"
                       " Green: 2, Red: 3, NIR: 4, SWIR: 5, SWIR2: 7\n"
                       "Landsat-8/9: \n"
                       " Green: 3, Red: 4, NIR: 5, SWIR: 6, SWIR2: 7\n\n"
                       "Please ensure you select the correct bands for the chosen satellite and index.\n"
                       "Note: Input bands must have the same resolution.")