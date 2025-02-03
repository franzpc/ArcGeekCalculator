from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterBoolean, QgsProcessingParameterCrs,
                       QgsProcessingParameterEnum, QgsField, QgsDistanceArea,
                       QgsProject, QgsWkbTypes, QgsProcessingParameterNumber,
                       QgsFeatureSink, QgsProcessingException,
                       QgsProcessingParameterFeatureSink, QgsUnitTypes, QgsExpression,
                       QgsExpressionContext, QgsExpressionContextUtils, QgsFeature,
                       QgsCoordinateTransform)

class CalculateLineGeometryAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    CRS = 'CRS'
    UNITS = 'UNITS'
    PRECISION = 'PRECISION'
    CALCULATION_METHOD = 'CALCULATION_METHOD'
    SELECTED_ONLY = 'SELECTED_ONLY'
    UPDATE_EXISTING = 'UPDATE_EXISTING'
    OUTPUT = 'OUTPUT'

    def __init__(self):
        super().__init__()
        self.units = [
            QgsUnitTypes.DistanceMeters,
            QgsUnitTypes.DistanceKilometers,
            QgsUnitTypes.DistanceFeet,
            QgsUnitTypes.DistanceYards,
            QgsUnitTypes.DistanceMiles,
            QgsUnitTypes.DistanceNauticalMiles,
            QgsUnitTypes.DistanceCentimeters,
            QgsUnitTypes.DistanceMillimeters
        ]

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return CalculateLineGeometryAlgorithm()

    def name(self):
        return 'calculatelinegeometry'

    def displayName(self):
        return self.tr('Calculate Line Geometry')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'arcgeekcalculator'

    def shortHelpString(self):
        return self.tr("""Calculates and adds a length field to line layers.

        Parameters:
        - Input line layer: Select the layer to process.
        - Length units: Choose the desired unit for length calculation.
        - Precision: Set the number of decimal places for results.
        - Calculation method: Choose between Cartesian (planar) or Ellipsoidal (curved surface) calculations.
        - Selected features only: Process only selected features if checked.
        - Update existing layer: Modify the input layer instead of creating a new one.
        - CRS for calculations: Optionally specify a CRS for the calculations.
        - Output layer: Specify the output layer (if not updating existing).

        Supports various units and calculation methods (Cartesian/Ellipsoidal).""")

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT,
                self.tr('Input line layer'),
                [QgsWkbTypes.LineGeometry]
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.UNITS,
                self.tr('Length units'),
                options=[QgsUnitTypes.toString(unit) for unit in self.units],
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PRECISION,
                self.tr('Precision'),
                type=QgsProcessingParameterNumber.Integer,
                minValue=0,
                maxValue=15,
                defaultValue=2
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.CALCULATION_METHOD,
                self.tr('Calculation method'),
                options=[self.tr('Cartesian'), self.tr('Ellipsoidal')],
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SELECTED_ONLY,
                self.tr('Selected features only'),
                defaultValue=False
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.UPDATE_EXISTING,
                self.tr('Update existing layer'),
                defaultValue=False
            )
        )
        self.addParameter(
            QgsProcessingParameterCrs(
                self.CRS,
                self.tr('CRS for calculations'),
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr('Output layer')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        crs = self.parameterAsCrs(parameters, self.CRS, context)
        unit_index = self.parameterAsEnum(parameters, self.UNITS, context)
        precision = self.parameterAsInt(parameters, self.PRECISION, context)
        calculation_method = self.parameterAsEnum(parameters, self.CALCULATION_METHOD, context)
        selected_only = self.parameterAsBool(parameters, self.SELECTED_ONLY, context)
        update_existing = self.parameterAsBool(parameters, self.UPDATE_EXISTING, context)

        if not crs.isValid():
            crs = source.crs()

        unit = self.units[unit_index]
        unit_suffix = self.get_abbreviated_unit_name(unit)
        length_field = f'l_{unit_suffix}'

        conv_factor = QgsUnitTypes.fromUnitToUnitFactor(crs.mapUnits(), unit)

        fields = source.fields()
        if length_field not in fields.names():
            fields.append(QgsField(length_field, QVariant.Double, len=20, prec=precision))

        if update_existing:
            if length_field not in source.fields().names():
                source.dataProvider().addAttributes([QgsField(length_field, QVariant.Double, len=20, prec=precision)])
                source.updateFields()
            sink = source
            sink_id = self.INPUT
        else:
            (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                                   fields, source.wkbType(), source.crs())

        features = source.getSelectedFeatures() if selected_only else source.getFeatures()
        total = 100.0 / (source.selectedFeatureCount() if selected_only else source.featureCount())

        distance_area = QgsDistanceArea()
        distance_area.setEllipsoid(crs.ellipsoidAcronym())
        distance_area.setSourceCrs(source.crs(), QgsProject.instance().transformContext())

        transform = QgsCoordinateTransform(source.crs(), crs, QgsProject.instance()) if source.crs() != crs else None

        if update_existing:
            source.startEditing()

        for current, feature in enumerate(features):
            if feedback.isCanceled():
                break

            if calculation_method == 1:  # Ellipsoidal
                length = distance_area.measureLength(feature.geometry())
            else:  # Cartesian
                if transform:
                    transformed_geom = feature.geometry()
                    transformed_geom.transform(transform)
                    length = transformed_geom.length()
                else:
                    length = feature.geometry().length()

            length = length * conv_factor
            length = round(length, precision)

            if update_existing:
                idx = source.fields().indexOf(length_field)
                source.changeAttributeValue(feature.id(), idx, length)
            else:
                new_feature = QgsFeature(fields)
                new_feature.setGeometry(feature.geometry())
                for field in source.fields():
                    new_feature[field.name()] = feature[field.name()]
                new_feature[length_field] = length
                sink.addFeature(new_feature, QgsFeatureSink.FastInsert)

            feedback.setProgress(int(current * total))

        if update_existing:
            source.commitChanges()

        return {self.OUTPUT: sink.id() if update_existing else dest_id}

    def get_abbreviated_unit_name(self, unit):
        unit_str = QgsUnitTypes.toString(unit).lower()
        if unit_str == 'meters':
            return 'm'
        elif unit_str == 'kilometers':
            return 'km'
        elif unit_str == 'feet':
            return 'ft'
        elif unit_str == 'yards':
            return 'yd'
        elif unit_str == 'miles':
            return 'mi'
        elif unit_str == 'nautical miles':
            return 'nmi'
        elif unit_str == 'centimeters':
            return 'cm'
        elif unit_str == 'millimeters':
            return 'mm'
        else:
            return unit_str