from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing, QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterNumber, QgsProcessingParameterRasterDestination,
                       QgsRasterLayer, QgsProcessingException, QgsRasterBlock,
                       QgsRasterFileWriter, Qgis, QgsRectangle, QgsProcessingParameterExtent,
                       QgsRasterBandStats)
import processing

class DamFloodSimulationAlgorithm(QgsProcessingAlgorithm):
    INPUT_DEM = 'INPUT_DEM'
    WATER_LEVEL = 'WATER_LEVEL'
    STUDY_AREA = 'STUDY_AREA'
    OUTPUT_RASTER = 'OUTPUT_RASTER'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_DEM, 'Input DEM'))
        self.addParameter(QgsProcessingParameterNumber(self.WATER_LEVEL, 'Water Level (m)', type=QgsProcessingParameterNumber.Double, defaultValue=25))
        self.addParameter(QgsProcessingParameterExtent(self.STUDY_AREA, 'Study Area', optional=True))
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT_RASTER, 'Output Water Depth Raster'))

    def processAlgorithm(self, parameters, context, feedback):
        original_dem = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
        water_level = self.parameterAsDouble(parameters, self.WATER_LEVEL, context)
        study_area = self.parameterAsExtent(parameters, self.STUDY_AREA, context)
        output_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_RASTER, context)

        if not original_dem.isValid():
            raise QgsProcessingException('Invalid DEM layer')

        # Determine which DEM to use
        if study_area and not study_area.isNull():
            feedback.pushInfo('Clipping DEM to specified study area...')
            clipped_dem = processing.run("gdal:cliprasterbyextent", {
                'INPUT': original_dem,
                'PROJWIN': study_area,
                'NODATA': original_dem.dataProvider().sourceNoDataValue(1),
                'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
            }, context=context, feedback=feedback)['OUTPUT']
            dem_layer = QgsRasterLayer(clipped_dem, 'Clipped DEM')
            if not dem_layer.isValid():
                raise QgsProcessingException('Failed to clip DEM')
        else:
            dem_layer = original_dem
            feedback.pushInfo('Using the entire DEM (no study area specified).')

        # Get DEM properties
        width = dem_layer.width()
        height = dem_layer.height()
        extent = dem_layer.extent()

        # Get DEM statistics
        dem_stats = dem_layer.dataProvider().bandStatistics(1, QgsRasterBandStats.All)
        dem_min = dem_stats.minimumValue
        dem_max = dem_stats.maximumValue
        feedback.pushInfo(f'DEM Min: {dem_min}, Max: {dem_max}')

        # Check if the DEM is very mountainous
        elevation_range = dem_max - dem_min
        if elevation_range > 1000:  # Arbitrary threshold, adjust as needed
            feedback.pushWarning('The DEM appears to be very mountainous. Due to the nature of the terrain, it may be challenging to find suitable areas for dam construction.')

        # Calculate flood level
        flood_level = dem_min + water_level
        feedback.pushInfo(f'Flood level: {flood_level}')

        # Create output raster
        writer = QgsRasterFileWriter(output_raster)
        writer.setOutputProviderKey('gdal')
        writer.setOutputFormat('GTiff')

        output_provider = writer.createOneBandRaster(
            Qgis.Float32, width, height, extent, dem_layer.crs())

        if not output_provider.isValid():
            raise QgsProcessingException("Could not create raster output provider.")

        NO_DATA = -9999
        output_provider.setNoDataValue(1, NO_DATA)
        output_provider.setEditable(True)

        # Calculate water depth
        feedback.pushInfo('Calculating water depth...')
        block = QgsRasterBlock(Qgis.Float32, width, 1)
        total_volume = 0
        flooded_area = 0
        max_depth = 0
        for row in range(height):
            if feedback.isCanceled():
                break
            feedback.setProgress(int((row / height) * 100))
            
            row_extent = QgsRectangle(extent.xMinimum(), extent.yMaximum() - (row + 1) * extent.height() / height,
                                      extent.xMaximum(), extent.yMaximum() - row * extent.height() / height)
            dem_data = dem_layer.dataProvider().block(1, row_extent, width, 1)
            
            for col in range(width):
                dem_value = dem_data.value(col)
                if dem_value is not None and dem_value != dem_layer.dataProvider().sourceNoDataValue(1):
                    water_depth = max(0, flood_level - dem_value)
                    if water_depth > 0:
                        block.setValue(0, col, water_depth)
                        total_volume += water_depth
                        flooded_area += 1
                        max_depth = max(max_depth, water_depth)
                    else:
                        block.setValue(0, col, NO_DATA)
                else:
                    block.setValue(0, col, NO_DATA)
            
            output_provider.writeBlock(block, 1, 0, row)

        output_provider.setEditable(False)

        # Calculate total water volume and flooded area
        cell_width = extent.width() / width
        cell_height = extent.height() / height
        cell_area = abs(cell_width * cell_height)
        total_volume *= cell_area  # Volume in cubic meters
        flooded_area *= cell_area  # Area in square meters

        feedback.pushInfo(f'The total volume of stored water is: {total_volume:.2f} cubic meters')
        feedback.pushInfo(f'The total flooded area is: {flooded_area:.2f} square meters')
        feedback.pushInfo(f'Maximum water depth: {max_depth:.2f} meters')
        feedback.pushInfo(f'Cell size used for calculation: {cell_width:.2f} x {cell_height:.2f} meters')

        return {self.OUTPUT_RASTER: output_raster}

    def name(self):
        return 'damfloodsimulation'

    def displayName(self):
        return self.tr('Dam Flood Simulation')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'arcgeekcalculator'

    def shortHelpString(self):
        return self.tr("""This algorithm simulates flooding by creating a water depth raster based on a DEM and specified water level.

        Parameters:
        - Input DEM: The Digital Elevation Model of the area.
        - Water Level (m): The water level above the minimum elevation of the DEM.
        - Study Area: The extent to process the DEM (optional). If not provided, the entire DEM will be used.
        - Output Water Depth Raster: The resulting raster showing water depth.

        The algorithm calculates the flood level by adding the specified water level to the minimum 
        DEM elevation within the study area (if specified) or the entire DEM. It then calculates 
        water depth where the elevation is below the flood level. Areas above the flood level are 
        set to NoData in the output raster.

        Note: For very mountainous terrain, the algorithm will provide a warning as it may be 
        challenging to find suitable areas for dam construction in such landscapes.""")

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return DamFloodSimulationAlgorithm()