from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterRasterDestination, QgsProcessingParameterNumber,
                       QgsProcessingException, QgsRasterLayer, QgsColorRampShader,
                       QgsRasterShader, QgsSingleBandPseudoColorRenderer, QgsRasterBandStats)
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
        self.addParameter(QgsProcessingParameterNumber(self.CATEGORY_TO_ANALYZE, 'Category to analyze', type=QgsProcessingParameterNumber.Integer, minValue=1, defaultValue=1))
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT_DETAILED_RASTER, 'Output detailed change raster', optional=True))
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT_SIMPLIFIED_RASTER, 'Output simplified change raster', optional=True))
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT_GAIN_RASTER, 'Output gain raster', optional=True))
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT_LOSS_RASTER, 'Output loss raster', optional=True))

    def processAlgorithm(self, parameters, context, feedback):
        raster_before = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER_BEFORE, context)
        raster_after = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER_AFTER, context)
        category = self.parameterAsInt(parameters, self.CATEGORY_TO_ANALYZE, context)
        output_detailed_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_DETAILED_RASTER, context)
        output_simplified_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_SIMPLIFIED_RASTER, context)
        output_gain_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_GAIN_RASTER, context)
        output_loss_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_LOSS_RASTER, context)

        if raster_before is None or raster_after is None:
            raise QgsProcessingException('Invalid input layers')

        # Determine the number of categories
        categories_before = self.get_unique_values(raster_before)
        categories_after = self.get_unique_values(raster_after)
        all_categories = sorted(set(categories_before + categories_after))
        num_categories = len(all_categories)

        feedback.pushInfo(f"Number of categories detected: {num_categories}")

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

        formulas_and_outputs = [
            ('(ras_after@1 * 10) + ras_before@1', output_detailed_raster),
            ('(ras_after@1 != ras_before@1)', output_simplified_raster),
            (f'(ras_after@1 = {category}) * (ras_before@1 != {category})', output_gain_raster),
            (f'(ras_before@1 = {category}) * (ras_after@1 != {category})', output_loss_raster)
        ]

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

            if output == output_detailed_raster:
                self.apply_detailed_symbology(output_detailed_raster, 'Detailed Change Raster', num_categories, context, feedback)
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
        provider = raster_layer.dataProvider()
        stats = provider.bandStatistics(1, QgsRasterBandStats.All)
        min_val = int(stats.minimumValue)
        max_val = int(stats.maximumValue)
        return sorted(set(range(min_val, max_val + 1)))

    def apply_symbology(self, raster_path, layer_name, color_map, context, feedback):
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

    def apply_detailed_symbology(self, raster_path, layer_name, num_categories, context, feedback):
        layer = QgsRasterLayer(raster_path, layer_name)
        if layer.isValid():
            shader = QgsRasterShader()
            color_ramp = QgsColorRampShader()
            color_ramp.setColorRampType(QgsColorRampShader.Discrete)
            
            # Create a color map for all possible combinations using Viridis-like colors
            color_map = []
            viridis_colors = [
                (68, 1, 84), (72, 35, 116), (64, 67, 135), (52, 94, 141),
                (41, 120, 142), (32, 144, 141), (34, 168, 132), (68, 190, 112),
                (121, 209, 81), (189, 222, 38), (253, 231, 37)
            ]
            
            total_combinations = num_categories * num_categories
            for idx in range(total_combinations):
                from_category = (idx % num_categories) + 1
                to_category = (idx // num_categories) + 1
                value = to_category * 10 + from_category
                color_idx = int(idx * (len(viridis_colors) - 1) / (total_combinations - 1))
                color = QColor(*viridis_colors[color_idx])
                label = f'From {from_category} to {to_category}'
                color_map.append(QgsColorRampShader.ColorRampItem(value, color, label))
            
            color_ramp.setColorRampItemList(color_map)
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
        This algorithm calculates various aspects of change between two raster images representing land use or land cover at different times.

        Parameters:
        - Input raster layer (before): The raster representing the earlier state.
        - Input raster layer (after): The raster representing the later state.
        - Category to analyze: The specific land use category to analyze for gain and loss.

        Outputs:
        1. Detailed change raster: Shows the detailed change between the two input layers.
           - Formula: (Current year * 10) + Previous year
           - Example: 32 means change from category 2 to 3
           - Colored using a Viridis-like palette for better distinction

        2. Simplified change raster: Shows areas of change and no change.
           - Value 0 (Dark purple): No change
           - Value 1 (Yellow): Change occurred

        3. Gain raster: Shows areas where the specified category was gained.
           - Value 0 (Dark purple): No gain
           - Value 1 (Turquoise): Gain

        4. Loss raster: Shows areas where the specified category was lost.
           - Value 0 (Dark purple): No loss
           - Value 1 (Yellow): Loss

        Note: 
        - The algorithm automatically detects the number of categories in your input data.
        - Ensure that your input rasters have integer values representing land use classes.
        - The gain and loss rasters are specific to the category you choose to analyze.
        - All output rasters are automatically styled using a Viridis-like color palette for easy interpretation.

        Use this tool to detect and analyze changes in land use or land cover over time, with a focus on specific categories of interest.
        """

    def createInstance(self):
        return LandUseChangeDetectionAlgorithm()