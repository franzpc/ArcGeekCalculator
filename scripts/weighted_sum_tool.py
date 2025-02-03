from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing, QgsProcessingAlgorithm, QgsProcessingParameterMultipleLayers,
                       QgsProcessingParameterString, QgsProcessingParameterRasterDestination, 
                       QgsProcessingException, QgsRasterLayer)
from qgis.analysis import QgsRasterCalculator, QgsRasterCalculatorEntry

class WeightedSumTool(QgsProcessingAlgorithm):
    INPUT_RASTERS = 'INPUT_RASTERS'
    WEIGHTS = 'WEIGHTS'
    OUTPUT_WEIGHTED_SUM = 'OUTPUT_WEIGHTED_SUM'  # Changed from OUTPUT to OUTPUT_WEIGHTED_SUM

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterMultipleLayers(
                self.INPUT_RASTERS,
                self.tr('Input rasters'),
                QgsProcessing.TypeRaster
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.WEIGHTS,
                self.tr('Weights (comma-separated)'),
                multiLine=False
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_WEIGHTED_SUM,  # Changed from OUTPUT to OUTPUT_WEIGHTED_SUM
                self.tr('Output Weighted Sum Raster')  # Changed description to be more descriptive
            )
        )

    def checkParameterValues(self, parameters, context):
        raster_layers = self.parameterAsLayerList(parameters, self.INPUT_RASTERS, context)
        weights_str = self.parameterAsString(parameters, self.WEIGHTS, context)

        if not raster_layers:
            return False, self.tr('No input rasters provided')

        num_rasters = len(raster_layers)
        example_weights = self.get_example_weights(num_rasters)

        try:
            weights = [float(w.strip()) for w in weights_str.split(',') if w.strip()]
        except ValueError:
            return False, self.tr(f'Invalid weight values. Please enter {num_rasters} numeric values separated by commas. '
                                  f'For example: {example_weights}')

        if len(weights) != num_rasters:
            return False, self.tr(f'Number of weights ({len(weights)}) does not match number of input rasters ({num_rasters}). '
                                  f'Please enter {num_rasters} weights. For example: {example_weights}')

        weight_sum = sum(weights)
        if abs(weight_sum - 100) > 0.001:
            return False, self.tr(f'Weights sum to {weight_sum:.2f}, but they must sum to 100. '
                                  f'Please adjust your weights. For example: {example_weights}')

        return super().checkParameterValues(parameters, context)

    def get_example_weights(self, num_rasters):
        if num_rasters == 2:
            return "50,50"
        elif num_rasters == 3:
            return "30,30,40"
        elif num_rasters == 4:
            return "25,25,25,25"
        elif num_rasters == 5:
            return "20,20,20,20,20"
        elif num_rasters == 6:
            return "16,16,17,17,17,17"
        else:
            base_weight = int(100 / num_rasters)
            remainder = 100 - (base_weight * num_rasters)
            weights = [base_weight] * num_rasters
            for i in range(remainder):
                weights[i] += 1
            return ",".join(map(str, weights))

    def processAlgorithm(self, parameters, context, feedback):
        raster_layers = self.parameterAsLayerList(parameters, self.INPUT_RASTERS, context)
        weights_str = self.parameterAsString(parameters, self.WEIGHTS, context)
        output = self.parameterAsOutputLayer(parameters, self.OUTPUT_WEIGHTED_SUM, context)  # Changed from OUTPUT to OUTPUT_WEIGHTED_SUM

        weights = [float(w.strip()) for w in weights_str.split(',') if w.strip()]

        # Normalize weights to sum to 1
        weights = [w / 100 for w in weights]

        # Create raster calculator entries
        entries = []
        for i, raster in enumerate(raster_layers):
            entry = QgsRasterCalculatorEntry()
            entry.ref = f'ras{i}@1'
            entry.raster = raster
            entry.bandNumber = 1
            entries.append(entry)

        # Construct formula
        formula = ' + '.join([f'({entry.ref} * {w})' for entry, w in zip(entries, weights)])

        # Create and execute raster calculator
        calc = QgsRasterCalculator(formula, output, 'GTiff', 
                                   raster_layers[0].extent(), raster_layers[0].width(), raster_layers[0].height(), 
                                   entries)
        result = calc.processCalculation(feedback)

        if result != 0:
            raise QgsProcessingException(self.tr(f'Error calculating output raster: {result}'))

        return {self.OUTPUT_WEIGHTED_SUM: output}  # Changed from OUTPUT to OUTPUT_WEIGHTED_SUM

    def name(self):
        return 'weightedsum'

    def displayName(self):
        return self.tr('Weighted Sum')

    def group(self):
        return self.tr('Raster analysis')

    def groupId(self):
        return 'rasteranalysis'

    def shortHelpString(self):
        return self.tr("This algorithm performs a weighted sum of multiple raster layers.\n\n"
                       "1. Select multiple input raster layers.\n"
                       "2. Enter the weights as comma-separated values (e.g., 50,50 for two layers).\n"
                       "3. The weights must sum to 100%.\n"
                       "4. IMPORTANT: The order of the weights must match the order of the selected rasters.\n"
                       "   The first weight corresponds to the first raster, the second weight to the second raster, and so on.\n\n"
                       "For example, if you select rasters A, B, and C in that order, and enter weights '30,30,40',\n"
                       "the algorithm will apply 30% to raster A, 30% to raster B, and 40% to raster C.\n\n"
                       "The tool will automatically calculate the weighted sum based on your inputs.")

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return WeightedSumTool()