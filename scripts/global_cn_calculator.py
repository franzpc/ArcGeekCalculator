from qgis.core import (QgsProcessingAlgorithm,
                       QgsProcessingParameterRasterDestination,
                       QgsProcessingParameterVectorLayer, QgsProcessing,
                       QgsProcessingException, QgsRasterLayer,
                       QgsCoordinateReferenceSystem, QgsRasterBandStats,
                       QgsProcessingMultiStepFeedback,
                       QgsProcessingParameterEnum)
from qgis.PyQt.QtCore import QCoreApplication
import processing
import requests
import os
import tempfile
import csv
from osgeo import gdal

class GlobalCNCalculator(QgsProcessingAlgorithm):
    """
    Calculates the Curve Number using global datasets from ESA WorldCover and ORNL HYSOG.
    """
    
    INPUT_AREA = 'INPUT_AREA'
    OUTPUT_CN = 'OUTPUT_CN'
    OUTPUT_LANDCOVER = 'OUTPUT_LANDCOVER'
    OUTPUT_SOIL = 'OUTPUT_SOIL'
    HC = 'HC'
    ARC = 'ARC'
    
    def initAlgorithm(self, config=None):
        self.hc = ["Poor", "Fair", "Good"]
        self.arc = ["I", "II", "III"]
        
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_AREA,
                self.tr('Study Area'),
                [QgsProcessing.TypeVectorPolygon]
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                self.HC,
                self.tr('Hydrologic Condition'),
                options=self.hc,
                defaultValue=1,
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                self.ARC,
                self.tr('Antecedent Runoff Condition'),
                options=self.arc,
                defaultValue=1,
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_LANDCOVER,
                self.tr('ESA Land Cover'),
                optional=True,
                createByDefault=False
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_SOIL,
                self.tr('Hydrologic Soil Groups'),
                optional=True,
                createByDefault=False
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_CN,
                self.tr('Curve Number'),
                optional=False,
                createByDefault=True
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        feedback.pushInfo(self.tr('Starting Curve Number calculation...'))
        
        steps = 6
        multi_feedback = QgsProcessingMultiStepFeedback(steps, feedback)
        
        # Get parameters
        aoi = self.parameterAsVectorLayer(parameters, self.INPUT_AREA, context)
        if not aoi.isValid():
            raise QgsProcessingException(self.tr('Invalid study area'))

        # Transform to EPSG:4326 if needed
        if aoi.crs().authid() != 'EPSG:4326':
            feedback.pushInfo(self.tr('Reprojecting study area to EPSG:4326...'))
            result = processing.run('native:reprojectlayer', {
                'INPUT': aoi,
                'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:4326'),
                'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
            }, context=context, feedback=feedback)
            aoi = result['OUTPUT']

        # Get VRT file path
        vrt_path = os.path.join(os.path.dirname(__file__), 'data', 'esa_worldcover_2021.vrt')
        if not os.path.exists(vrt_path):
            raise QgsProcessingException(self.tr('ESA WorldCover VRT file not found'))

        results = {}
        try:
            # Process Land Cover
            multi_feedback.setCurrentStep(0)
            feedback.pushInfo(self.tr('Step 1/6: Processing ESA WorldCover data...'))
            landcover = self.process_landcover(aoi, vrt_path, parameters, context, feedback)
            if parameters.get(self.OUTPUT_LANDCOVER, None):
                results[self.OUTPUT_LANDCOVER] = landcover

            # Process Soil Data
            multi_feedback.setCurrentStep(1)
            feedback.pushInfo(self.tr('Step 2/6: Processing ORNL HYSOG data...'))
            soil = self.process_soil_data(aoi, parameters, context, feedback)
            if parameters.get(self.OUTPUT_SOIL, None):
                results[self.OUTPUT_SOIL] = soil

            # Align rasters
            multi_feedback.setCurrentStep(2)
            feedback.pushInfo(self.tr('Step 3/6: Aligning datasets...'))
            aligned_soil = self.align_rasters(soil, landcover, context, feedback)

            # Calculate initial CN
            multi_feedback.setCurrentStep(3)
            feedback.pushInfo(self.tr('Step 4/6: Calculating initial Curve Number...'))
            temp_cn_raster = self.calculate_cn(landcover, aligned_soil, QgsProcessing.TEMPORARY_OUTPUT, 
                                               parameters.get(self.HC, 1),
                                               parameters.get(self.ARC, 1),
                                               context, feedback)

            # Clip CN raster with input polygon
            multi_feedback.setCurrentStep(4)
            feedback.pushInfo(self.tr('Step 5/6: Clipping final Curve Number to study area...'))
            
            # Clip the raster
            clipped_cn = processing.run("gdal:cliprasterbymasklayer", {
                'INPUT': temp_cn_raster,
                'MASK': aoi,
                'SOURCE_CRS': QgsCoordinateReferenceSystem('EPSG:4326'),
                'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:4326'),
                'NODATA': None,
                'ALPHA_BAND': False,
                'CROP_TO_CUTLINE': True,
                'KEEP_RESOLUTION': True,
                'SET_RESOLUTION': False,
                'OUTPUT': parameters[self.OUTPUT_CN]
            }, context=context, feedback=feedback)['OUTPUT']
            
            # Set the layer name to "Curve Number"
            output_layer = QgsRasterLayer(clipped_cn, 'Curve Number')

            results[self.OUTPUT_CN] = clipped_cn

            # Calculate statistics
            multi_feedback.setCurrentStep(5)
            feedback.pushInfo(self.tr('Step 6/6: Calculating statistics...'))
            self.calculate_statistics(clipped_cn, feedback)

            return results

        except Exception as e:
            raise QgsProcessingException(str(e))

    def process_landcover(self, aoi, vrt_path, parameters, context, feedback):
        """Process landcover data from VRT"""
        extent = aoi.extent()
        extent_str = f"{extent.xMinimum()},{extent.xMaximum()},{extent.yMinimum()},{extent.yMaximum()} [EPSG:4326]"
        
        output = parameters.get(self.OUTPUT_LANDCOVER, QgsProcessing.TEMPORARY_OUTPUT)
        
        return processing.run("gdal:cliprasterbyextent", {
            'INPUT': vrt_path,
            'PROJWIN': extent_str,
            'NODATA': None,
            'OPTIONS': '',
            'DATA_TYPE': 0,
            'OUTPUT': output
        }, context=context, feedback=feedback)['OUTPUT']

    def process_soil_data(self, aoi, parameters, context, feedback):
        """Download and process ORNL soil data"""
        extent = aoi.extent()
        bbox = f"{extent.xMinimum()},{extent.yMinimum()},{extent.xMaximum()},{extent.yMaximum()}"
        width = int((extent.xMaximum() - extent.xMinimum()) / 0.002083333)
        height = int(width * (extent.yMaximum() - extent.yMinimum()) / 
                    (extent.xMaximum() - extent.xMinimum()))

        url = f"https://webmap.ornl.gov/ogcbroker/wcs?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage&FORMAT=GeoTIFF_BYTE&COVERAGE=1566_1&WIDTH={width}&HEIGHT={height}&BBOX={bbox}&CRS=epsg:4326&RESPONSE_CRS=epsg:4326"

        try:
            temp_file = os.path.join(tempfile.gettempdir(), f'ornl_soil_{os.getpid()}.tif')
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            output = parameters.get(self.OUTPUT_SOIL, QgsProcessing.TEMPORARY_OUTPUT)
            
            result = processing.run("gdal:translate", {
                'INPUT': temp_file,
                'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:4326'),
                'NODATA': None,
                'COPY_SUBDATASETS': False,
                'OPTIONS': 'COMPRESS=LZW',
                'DATA_TYPE': 0,
                'OUTPUT': output
            }, context=context, feedback=feedback)['OUTPUT']
            
            try:
                os.remove(temp_file)
            except:
                pass
                
            return result
            
        except Exception as e:
            raise QgsProcessingException(f'Error downloading soil data: {str(e)}')

    def align_rasters(self, soil_raster, lc_raster, context, feedback):
        """Align soil raster to land cover resolution and extent"""
        lc = QgsRasterLayer(lc_raster)
        extent = lc.extent()
        pixel_size = lc.rasterUnitsPerPixelX()
        
        return processing.run("gdal:warpreproject", {
            'INPUT': soil_raster,
            'SOURCE_CRS': QgsCoordinateReferenceSystem('EPSG:4326'),
            'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:4326'),
            'RESAMPLING': 0,
            'TARGET_RESOLUTION': pixel_size,
            'TARGET_EXTENT': f"{extent.xMinimum()},{extent.xMaximum()},{extent.yMinimum()},{extent.yMaximum()} [EPSG:4326]",
            'TARGET_EXTENT_CRS': QgsCoordinateReferenceSystem('EPSG:4326'),
            'OPTIONS': '',
            'DATA_TYPE': 0,
            'MULTITHREADING': False,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }, context=context, feedback=feedback)['OUTPUT']

    def calculate_cn(self, landcover, soil, output_path, hc_index, arc_index, context, feedback):
        """Calculate CN values"""
        cn_values = self.get_cn_values(hc_index, arc_index)
        
        expressions = []
        for lc_code, soil_values in cn_values.items():
            for soil_code, cn in soil_values.items():
                expressions.append(
                    f'(A=={lc_code})*(B=={soil_code})*{cn}'
                )
        
        formula = '+'.join(expressions)
        
        return processing.run("gdal:rastercalculator", {
            'INPUT_A': landcover,
            'BAND_A': 1,
            'INPUT_B': soil,
            'BAND_B': 1,
            'FORMULA': formula,
            'NO_DATA': None,
            'RTYPE': 0,  # Byte data type
            'OPTIONS': '',
            'OUTPUT': output_path
        }, context=context, feedback=feedback)['OUTPUT']

    def get_cn_values(self, hc_index, arc_index):
        """Return CN lookup table reading from CSV files"""
        csv_filename = f"default_lookup_{(self.hc[hc_index][:1].lower())}_{(self.arc[arc_index].lower())}.csv"
        csv_path = os.path.join(os.path.dirname(__file__), 'data', csv_filename)
    
        cn_values = {}
    
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Skip empty lines
                    if not row['grid_code'].strip():
                        continue
                    
                    lc, soil = row['grid_code'].split('_')
                    lc = int(lc)
                    if lc not in cn_values:
                        cn_values[lc] = {}
                    soil_code = '1' if soil == 'A' else '2' if soil == 'B' else '3' if soil == 'C' else '4'
                    cn_values[lc][soil_code] = int(row['cn'])
                    
            if not cn_values:
                raise QgsProcessingException(f'No valid CN values found in {csv_filename}')
            
            return cn_values
        
        except Exception as e:
            raise QgsProcessingException(f'Error reading CN values from {csv_filename}: {str(e)}')

    def calculate_statistics(self, raster_path, feedback):
        """Calculate and report CN statistics with derived hydrological parameters"""
        try:
            raster = QgsRasterLayer(raster_path)
            if not raster.isValid():
                feedback.pushWarning(self.tr('Could not open output raster for statistics'))
                return
                
            provider = raster.dataProvider()
            stats = provider.bandStatistics(1, QgsRasterBandStats.All)
            
            # Calculate additional hydrological parameters
            mean_cn = stats.mean
            S = (25400 / mean_cn) - 254  # Storage potential in mm
            P = 100  # Example rainfall in mm (can be modified)
            Q = ((P - 0.2 * S) ** 2) / (P + 0.8 * S) if P > 0.2 * S else 0  # Direct runoff in mm
            
            feedback.pushInfo('\n=== Curve Number Statistics ===')
            feedback.pushInfo(f'Mean CN: {mean_cn:.2f}')
            feedback.pushInfo(f'S = Maximum potential retention (mm): {S:.2f}')
            feedback.pushInfo('============================')
            
        except Exception as e:
            feedback.pushWarning(f'Error calculating statistics: {str(e)}')

    def name(self):
        return 'globalcurvenumber'

    def displayName(self):
        return self.tr('Global Curve Number')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'arcgeekcalculator'

    def shortHelpString(self):
        return self.tr("""
        Calculates Curve Number using global datasets:
        - ESA WorldCover 2021 for land cover
        - ORNL HYSOG for hydrologic soil groups
        
        Parameters:
        - Study Area: Polygon of the area to calculate CN for
        - Hydrologic Condition: Poor, Fair (default), or Good
        - Antecedent Runoff Condition: I (dry), II (normal, default), or III (wet)
            
        Outputs:
        - ESA Land Cover (optional): Land cover classification raster
        - Hydrologic Soil Groups (optional): Soil groups raster
        - Curve Number: Final CN raster
        
        The tool will also display statistics including mean, minimum, maximum,
        and standard deviation of the calculated CN values.
            
        Data Sources:
        - Land Cover: ESA WorldCover 2021 (10m resolution)
        - Soil Groups: ORNL HYSOG Global Hydrologic Soil Groups
        
        Note: Internet connection required to download soil and land cover data.
        Processing time depends on the size of your study area.

        This tool is based on the "Curve Number Generator" plugin by Abdul Raheem.
        """)

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return GlobalCNCalculator()