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
        
        aoi = self.parameterAsVectorLayer(parameters, self.INPUT_AREA, context)
        if not aoi.isValid():
            raise QgsProcessingException(self.tr('Invalid study area'))

        if aoi.crs().authid() != 'EPSG:4326':
            feedback.pushInfo(self.tr('Reprojecting study area to EPSG:4326...'))
            result = processing.run('native:reprojectlayer', {
                'INPUT': aoi,
                'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:4326'),
                'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
            }, context=context, feedback=feedback)
            aoi = result['OUTPUT']

        vrt_path = os.path.join(os.path.dirname(__file__), 'data', 'esa_worldcover_2021.vrt')
        if not os.path.exists(vrt_path):
            raise QgsProcessingException(self.tr('ESA WorldCover VRT file not found'))

        results = {}
        try:
            multi_feedback.setCurrentStep(0)
            feedback.pushInfo(self.tr('Step 1/6: Processing ESA WorldCover data...'))
            landcover = self.process_landcover(aoi, vrt_path, parameters, context, feedback)
            if parameters.get(self.OUTPUT_LANDCOVER, None):
                results[self.OUTPUT_LANDCOVER] = landcover

            multi_feedback.setCurrentStep(1)
            feedback.pushInfo(self.tr('Step 2/6: Processing ORNL HYSOG data from tiles...'))
            soil = self.process_soil_data_tiles(aoi, parameters, context, feedback)
            if parameters.get(self.OUTPUT_SOIL, None):
                results[self.OUTPUT_SOIL] = soil

            multi_feedback.setCurrentStep(2)
            feedback.pushInfo(self.tr('Step 3/6: Aligning datasets...'))
            aligned_soil = self.align_rasters(soil, landcover, context, feedback)

            multi_feedback.setCurrentStep(3)
            feedback.pushInfo(self.tr('Step 4/6: Calculating initial Curve Number...'))
            temp_cn_raster = self.calculate_cn(landcover, aligned_soil, QgsProcessing.TEMPORARY_OUTPUT, 
                                               parameters.get(self.HC, 1),
                                               parameters.get(self.ARC, 1),
                                               context, feedback)

            multi_feedback.setCurrentStep(4)
            feedback.pushInfo(self.tr('Step 5/6: Clipping final Curve Number to study area...'))
            
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
            
            results[self.OUTPUT_CN] = clipped_cn

            multi_feedback.setCurrentStep(5)
            feedback.pushInfo(self.tr('Step 6/6: Calculating statistics...'))
            self.calculate_statistics(clipped_cn, feedback)

            return results

        except Exception as e:
            raise QgsProcessingException(str(e))

    def process_landcover(self, aoi, vrt_path, parameters, context, feedback):
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

    def process_soil_data_tiles(self, aoi, parameters, context, feedback):
        extent = aoi.extent()
        min_lon, max_lon = extent.xMinimum(), extent.xMaximum()
        min_lat, max_lat = extent.yMinimum(), extent.yMaximum()
        
        base_url = "https://arcgeek.com/hysog_tiles/"
        tiles_info_url = f"{base_url}tiles_info.csv"
        
        try:
            feedback.pushInfo('Downloading tiles information...')
            response = requests.get(tiles_info_url, timeout=30)
            response.raise_for_status()
            
            required_tiles = []
            lines = response.text.strip().split('\n')
            header = lines[0].split(',')
            
            for line in lines[1:]:
                if not line.strip():
                    continue
                    
                parts = line.split(',')
                if len(parts) < 6:
                    continue
                    
                tile_min_lon = float(parts[2])
                tile_max_lon = float(parts[3])
                tile_min_lat = float(parts[4])
                tile_max_lat = float(parts[5])
                filename = parts[1]
                
                if not (max_lon < tile_min_lon or min_lon > tile_max_lon or 
                        max_lat < tile_min_lat or min_lat > tile_max_lat):
                    required_tiles.append(filename)
            
            if not required_tiles:
                raise QgsProcessingException('No HYSOG tiles found for the study area')
            
            feedback.pushInfo(f'Found {len(required_tiles)} required tiles: {", ".join(required_tiles)}')
            
            temp_dir = tempfile.mkdtemp()
            downloaded_tiles = []
            
            for i, tile_name in enumerate(required_tiles):
                feedback.pushInfo(f'Downloading tile {i+1}/{len(required_tiles)}: {tile_name}')
                tile_url = f"{base_url}{tile_name}"
                temp_tile_path = os.path.join(temp_dir, tile_name)
                
                response = requests.get(tile_url, stream=True, timeout=60)
                response.raise_for_status()
                
                with open(temp_tile_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                downloaded_tiles.append(temp_tile_path)
            
            if len(downloaded_tiles) == 1:
                merged_file = downloaded_tiles[0]
            else:
                feedback.pushInfo('Merging multiple tiles...')
                merged_file = os.path.join(temp_dir, 'merged_hysog.tif')
                
                vrt_file = os.path.join(temp_dir, 'tiles.vrt')
                vrt_ds = gdal.BuildVRT(vrt_file, downloaded_tiles)
                vrt_ds = None
                
                translate_options = gdal.TranslateOptions(
                    format='GTiff',
                    creationOptions=['COMPRESS=LZW']
                )
                gdal.Translate(merged_file, vrt_file, options=translate_options)
            
            feedback.pushInfo('Clipping HYSOG data to study area...')
            output = parameters.get(self.OUTPUT_SOIL, QgsProcessing.TEMPORARY_OUTPUT)
            
            extent_str = f"{min_lon},{max_lon},{min_lat},{max_lat} [EPSG:4326]"
            
            clipped_output = processing.run("gdal:cliprasterbyextent", {
                'INPUT': merged_file,
                'PROJWIN': extent_str,
                'NODATA': None,
                'OPTIONS': 'COMPRESS=LZW',
                'DATA_TYPE': 0,
                'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
            }, context=context, feedback=feedback)['OUTPUT']
            
            feedback.pushInfo('Processing NoData and invalid soil group values...')
            
            filled_temp = processing.run("gdal:fillnodata", {
                'INPUT': clipped_output,
                'BAND': 1,
                'DISTANCE': 10,
                'ITERATIONS': 0,
                'NO_MASK': False,
                'MASK_LAYER': None,
                'OPTIONS': '',
                'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
            }, context=context, feedback=feedback)['OUTPUT']
            
            output = parameters.get(self.OUTPUT_SOIL, QgsProcessing.TEMPORARY_OUTPUT)
            result = processing.run("gdal:rastercalculator", {
                'INPUT_A': filled_temp,
                'BAND_A': 1,
                'FORMULA': 'numpy.where((A == 255) | (A == 13) | (A == 14) | (A > 4), 3, A)',
                'NO_DATA': None,
                'RTYPE': 0,
                'OPTIONS': '',
                'OUTPUT': output
            }, context=context, feedback=feedback)['OUTPUT']
            
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass
                
            return result
            
        except requests.RequestException as e:
            raise QgsProcessingException(f'Error downloading HYSOG tiles: {str(e)}')
        except Exception as e:
            raise QgsProcessingException(f'Error processing HYSOG data: {str(e)}')

    def align_rasters(self, soil_raster, lc_raster, context, feedback):
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
            'RTYPE': 0,
            'OPTIONS': '',
            'OUTPUT': output_path
        }, context=context, feedback=feedback)['OUTPUT']

    def get_cn_values(self, hc_index, arc_index):
        csv_filename = f"default_lookup_{(self.hc[hc_index][:1].lower())}_{(self.arc[arc_index].lower())}.csv"
        csv_path = os.path.join(os.path.dirname(__file__), 'data', csv_filename)
    
        cn_values = {}
    
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
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
        try:
            raster = QgsRasterLayer(raster_path)
            if not raster.isValid():
                feedback.pushWarning(self.tr('Could not open output raster for statistics'))
                return
                
            provider = raster.dataProvider()
            stats = provider.bandStatistics(1, QgsRasterBandStats.All)
            
            mean_cn = stats.mean
            S = (25400 / mean_cn) - 254
            P = 100
            Q = ((P - 0.2 * S) ** 2) / (P + 0.8 * S) if P > 0.2 * S else 0
            
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
        - ORNL HYSOG for hydrologic soil groups (tiled system)
        
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
        - Soil Groups: ORNL HYSOG Global Hydrologic Soil Groups (tiled system)
        
        Note: Internet connection required to download soil data tiles.
        Processing time depends on the size of your study area and number of tiles needed.

        This tool is based on the "Curve Number Generator" plugin by Abdul Raheem.
        """)

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return GlobalCNCalculator()