from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterRasterDestination, QgsProcessingParameterNumber,
                       QgsProcessingException, QgsRasterLayer, QgsColorRampShader,
                       QgsRasterShader, QgsSingleBandPseudoColorRenderer, QgsRasterBandStats,
                       QgsPointXY)
from qgis.analysis import QgsRasterCalculator, QgsRasterCalculatorEntry
from qgis.PyQt.QtGui import QColor
import numpy as np

class LandUseChangeDetectionAlgorithm(QgsProcessingAlgorithm):
    INPUT_RASTER_BEFORE = 'INPUT_RASTER_BEFORE'
    INPUT_RASTER_AFTER = 'INPUT_RASTER_AFTER'
    CATEGORY_TO_ANALYZE = 'CATEGORY_TO_ANALYZE'
    OUTPUT_DETAILED_RASTER = 'OUTPUT_DETAILED_RASTER'
    OUTPUT_SIMPLIFIED_RASTER = 'OUTPUT_SIMPLIFIED_RASTER'
    OUTPUT_GAIN_RASTER = 'OUTPUT_GAIN_RASTER'
    OUTPUT_LOSS_RASTER = 'OUTPUT_LOSS_RASTER'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER_BEFORE, 'Input raster layer (before)'))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER_AFTER, 'Input raster layer (after)'))
        self.addParameter(QgsProcessingParameterNumber(self.CATEGORY_TO_ANALYZE, 'Category to analyze', 
                                                   type=QgsProcessingParameterNumber.Integer, 
                                                   minValue=0,  # Allow zero, but will validate against actual categories later
                                                   defaultValue=1))
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT_DETAILED_RASTER, 'Output detailed change raster', optional=True))
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT_SIMPLIFIED_RASTER, 'Output simplified change raster', optional=True))
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT_GAIN_RASTER, 'Output gain raster', optional=True))
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT_LOSS_RASTER, 'Output loss raster', optional=True))

    def processAlgorithm(self, parameters, context, feedback):
        # Get input parameters
        raster_before = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER_BEFORE, context)
        raster_after = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER_AFTER, context)
        category = self.parameterAsInt(parameters, self.CATEGORY_TO_ANALYZE, context)
        output_detailed_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_DETAILED_RASTER, context)
        output_simplified_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_SIMPLIFIED_RASTER, context)
        output_gain_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_GAIN_RASTER, context)
        output_loss_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_LOSS_RASTER, context)

        if raster_before is None or raster_after is None:
            raise QgsProcessingException('Invalid input layers')

        # Get actual categories (not just range) from both rasters
        provider = raster_before.dataProvider()
        nodata_value = provider.sourceNoDataValue(1)  # Get NoData value for band 1
        
        categories_before = self.get_actual_categories(raster_before)
        categories_after = self.get_actual_categories(raster_after)
        all_categories = sorted(set(categories_before + categories_after))
        
        # Count total categories and filter out nodata values
        all_categories = [cat for cat in all_categories if cat != nodata_value and cat > -1e30 and cat < 1e30]
        num_categories = len(all_categories)
        
        # Store categories for use in symbology
        setattr(context, 'categoriesBefore', all_categories)
        setattr(context, 'categoriesAfter', all_categories)
        
        feedback.pushInfo(f"Actual categories detected: {all_categories}")
        feedback.pushInfo(f"Number of categories detected: {num_categories}")

        # Validate that the selected category exists in at least one of the rasters
        if category not in all_categories:
            # Format the available categories more nicely
            formatted_categories = ", ".join(map(str, all_categories))
            raise QgsProcessingException(
                f'Selected category ({category}) does not exist in either raster. '
                f'Available categories are: {formatted_categories}.'
            )

        # Prepare raster entries for calculation
        entries = []
        ras_before = QgsRasterCalculatorEntry()
        ras_before.ref = 'ras_before@1'
        ras_before.raster = raster_before
        ras_before.bandNumber = 1
        entries.append(ras_before)

        ras_after = QgsRasterCalculatorEntry()
        ras_after.ref = 'ras_after@1'
        ras_after.raster = raster_after
        ras_after.bandNumber = 1
        entries.append(ras_after)

        # Adjust formula based on number of categories
        if num_categories <= 9:
            detailed_formula = '(ras_after@1 * 10) + ras_before@1'
            multiplier = 10
        else:
            detailed_formula = '(ras_after@1 * 1000) + ras_before@1'
            multiplier = 1000

        # Define formulas for different outputs
        formulas_and_outputs = [
            (detailed_formula, output_detailed_raster),
            ('(ras_after@1 != ras_before@1)', output_simplified_raster),
            (f'(ras_after@1 = {category}) * (ras_before@1 != {category})', output_gain_raster),
            (f'(ras_before@1 = {category}) * (ras_after@1 != {category})', output_loss_raster)
        ]

        # Process each formula and output
        for formula, output in formulas_and_outputs:
            if not output:
                continue
            feedback.pushInfo(f"Processing formula: {formula}")
            calc = QgsRasterCalculator(formula, output, 'GTiff', 
                                       raster_before.extent(), raster_before.width(), raster_before.height(), 
                                       entries)
            result = calc.processCalculation(feedback)
            if result != 0:
                feedback.pushInfo(f"Error code: {result}")
                raise QgsProcessingException(f'Error calculating raster: {output}')
            feedback.pushInfo(f"Successfully created: {output}")

            # Apply appropriate symbology based on output type
            if output == output_detailed_raster:
                self.apply_detailed_symbology(output_detailed_raster, 'Detailed Change Raster', 
                                           num_categories, multiplier, context, feedback)
            elif output == output_gain_raster:
                self.apply_symbology(output_gain_raster, 'Gain Raster', 
                                     [(0, QColor(33, 47, 60), 'No Gain'),
                                      (1, QColor(26, 255, 1), 'Gain')], 
                                     context, feedback)
            elif output == output_loss_raster:
                self.apply_symbology(output_loss_raster, 'Loss Raster', 
                                     [(0, QColor(33, 47, 60), 'No Loss'),
                                      (1, QColor(249, 4, 73), 'Loss')], 
                                     context, feedback)
            elif output == output_simplified_raster:
                self.apply_symbology(output_simplified_raster, 'Simplified Change Raster', 
                                     [(0, QColor(33, 47, 60), 'No Change'),
                                      (1, QColor(220, 16, 43), 'Change')], 
                                     context, feedback)

        return {
            self.OUTPUT_DETAILED_RASTER: output_detailed_raster,
            self.OUTPUT_SIMPLIFIED_RASTER: output_simplified_raster,
            self.OUTPUT_GAIN_RASTER: output_gain_raster,
            self.OUTPUT_LOSS_RASTER: output_loss_raster
        }

    def get_unique_values(self, raster_layer):
        """Get unique values range from raster layer (min to max)"""
        provider = raster_layer.dataProvider()
        stats = provider.bandStatistics(1, QgsRasterBandStats.All)
        min_val = int(stats.minimumValue)
        max_val = int(stats.maximumValue)
        return sorted(set(range(min_val, max_val + 1)))
        
    def get_actual_categories(self, raster_layer):
        """Get actual categories that exist in the raster layer (instead of all possible values in range)"""
        # We need to sample the actual values present in the raster
        provider = raster_layer.dataProvider()
        nodata_value = provider.sourceNoDataValue(1)  # Get NoData value for band 1
        
        block = provider.block(1, raster_layer.extent(), raster_layer.width(), raster_layer.height())
        
        # Get unique values that actually exist
        actual_values = set()
        for row in range(block.height()):
            for col in range(block.width()):
                value = block.value(row, col)
                # Check if it's a valid value (not NoData and not extremely large/small)
                if not np.isnan(value) and value != nodata_value and value > -1e30 and value < 1e30:
                    # Convert to integer if it appears to be one
                    if value == int(value):
                        actual_values.add(int(value))
                    else:
                        actual_values.add(value)
        
        return sorted(actual_values)

    def get_unique_values_from_raster(self, raster_layer):
        """Extract unique values directly from a raster layer"""
        provider = raster_layer.dataProvider()
        nodata_value = provider.sourceNoDataValue(1)
        
        # Get the extent, width and height
        extent = raster_layer.extent()
        width = raster_layer.width()
        height = raster_layer.height()
        
        # Sample a reasonable number of pixels (full sampling could be too slow for large rasters)
        sample_width = min(width, 1000)
        sample_height = min(height, 1000)
        
        # Create sampling parameters
        x_step = max(1, width // sample_width)
        y_step = max(1, height // sample_height)
        
        # Sample the raster
        unique_values = set()
        for y in range(0, height, y_step):
            for x in range(0, width, x_step):
                value = provider.sample(QgsPointXY(
                    extent.xMinimum() + (x / width) * extent.width(),
                    extent.yMinimum() + (y / height) * extent.height()
                ), 1)[0]
                
                if not np.isnan(value) and value != nodata_value and value > -1e30 and value < 1e30:
                    unique_values.add(int(value))
        
        return sorted(unique_values)

    def apply_symbology(self, raster_path, layer_name, color_map, context, feedback):
        """Apply basic symbology to raster layer"""
        layer = QgsRasterLayer(raster_path, layer_name)
        if layer.isValid():
            shader = QgsRasterShader()
            color_ramp = QgsColorRampShader()
            color_ramp.setColorRampType(QgsColorRampShader.Discrete)
            color_ramp.setColorRampItemList([QgsColorRampShader.ColorRampItem(value, color, label) for value, color, label in color_map])
            shader.setRasterShaderFunction(color_ramp)
            renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
            layer.setRenderer(renderer)
            layer.triggerRepaint()

            context.project().addMapLayer(layer)
            feedback.pushInfo(f"Custom symbology applied to {layer_name}")
        else:
            feedback.pushWarning(f"Failed to apply custom symbology to {layer_name}")

    def apply_detailed_symbology(self, raster_path, layer_name, num_categories, multiplier, context, feedback):
        """Apply detailed symbology to change raster"""
        layer = QgsRasterLayer(raster_path, layer_name)
        if layer.isValid():
            shader = QgsRasterShader()
            color_ramp = QgsColorRampShader()
            color_ramp.setColorRampType(QgsColorRampShader.Discrete)
            
            # Create a color map using Viridis-like colors
            color_map = []
            viridis_colors = [
                (68, 1, 84), (72, 35, 116), (64, 67, 135), (52, 94, 141),
                (41, 120, 142), (32, 144, 141), (34, 168, 132), (68, 190, 112),
                (121, 209, 81), (189, 222, 38), (253, 231, 37)
            ]
            
            # Try to get the actual categories from the result raster
            result_provider = layer.dataProvider()
            
            # Read categories from context instead of creating arbitrary ones
            categories_before = []
            categories_after = []
            
            # Get categories from context if available
            if hasattr(context, 'categoriesBefore') and hasattr(context, 'categoriesAfter'):
                categories_before = getattr(context, 'categoriesBefore')
                categories_after = getattr(context, 'categoriesAfter')
            else:
                # Fallback to sample the result raster for unique values
                unique_values = self.get_unique_values_from_raster(layer)
                categories_before = sorted(set([value % multiplier for value in unique_values if value % multiplier > 0]))
                categories_after = sorted(set([value // multiplier for value in unique_values if value // multiplier > 0]))
            
            # Create combinations only for actual categories
            color_map_items = []
            total_items = len(categories_before) * len(categories_after)
            idx = 0
            
            for to_category in categories_after:
                for from_category in categories_before:
                    value = to_category * multiplier + from_category
                    
                    # Generate color index
                    color_idx = int(idx * (len(viridis_colors) - 1) / max(1, total_items - 1))
                    color = QColor(*viridis_colors[color_idx])
                    label = f'From {from_category} to {to_category}'
                    color_map_items.append(QgsColorRampShader.ColorRampItem(value, color, label))
                    idx += 1
            
            color_ramp.setColorRampItemList(color_map_items)
            shader.setRasterShaderFunction(color_ramp)
            renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
            layer.setRenderer(renderer)
            layer.triggerRepaint()

            context.project().addMapLayer(layer)
            feedback.pushInfo(f"Custom detailed symbology applied to {layer_name}")
        else:
            feedback.pushWarning(f"Failed to apply custom detailed symbology to {layer_name}")

    def name(self):
        return 'landusechangedetection'

    def displayName(self):
        return 'Land Use Change Detection'

    def group(self):
        return 'ArcGeek Calculator'

    def groupId(self):
        return 'arcgeek_calculator'

    def shortHelpString(self):
        return """
        Calculates changes between two raster images representing land use/cover at different times.

        Parameters:
        - Input raster layer (before): Initial state
        - Input raster layer (after): Final state
        - Category to analyze: Specific category for gain/loss analysis (must exist in at least one raster)

        Outputs:
        1. Detailed change raster: Shows category transitions
           - For ≤9 categories: value = (current year * 10) + previous year
           - For >9 categories: value = (current year * 1000) + previous year

        2. Simplified raster: Areas with and without changes
        3. Gain raster: Where specified category was gained
        4. Loss raster: Where specified category was lost

        Note: Supports up to 99 categories with automatic symbology.
        """

    def createInstance(self):
        return LandUseChangeDetectionAlgorithm()