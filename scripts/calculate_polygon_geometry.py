from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterBoolean, QgsProcessingParameterCrs,
                       QgsProcessingParameterEnum, QgsField, QgsDistanceArea,
                       QgsProject, QgsWkbTypes, QgsProcessingParameterNumber,
                       QgsFeatureSink, QgsProcessingException,
                       QgsProcessingParameterFeatureSink, QgsUnitTypes, QgsExpression,
                       QgsExpressionContext, QgsExpressionContextUtils, QgsFeature,
                       QgsCoordinateTransform, QgsVectorFileWriter)

class CalculatePolygonGeometryAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    CRS = 'CRS'
    AREA_UNITS = 'AREA_UNITS'
    PERIMETER_UNITS = 'PERIMETER_UNITS'
    PRECISION = 'PRECISION'
    CALCULATION_METHOD = 'CALCULATION_METHOD'
    SELECTED_ONLY = 'SELECTED_ONLY'
    UPDATE_EXISTING = 'UPDATE_EXISTING'
    OUTPUT = 'OUTPUT'

    def __init__(self):
        super().__init__()
        self.area_units = [
            QgsUnitTypes.AreaSquareMeters,
            QgsUnitTypes.AreaSquareKilometers,
            QgsUnitTypes.AreaHectares,
            QgsUnitTypes.AreaSquareFeet,
            QgsUnitTypes.AreaSquareYards,
            QgsUnitTypes.AreaAcres,
            QgsUnitTypes.AreaSquareMiles,
            QgsUnitTypes.AreaSquareNauticalMiles,
            QgsUnitTypes.AreaSquareCentimeters,
            QgsUnitTypes.AreaSquareMillimeters
        ]
        self.perimeter_units = [
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
        return CalculatePolygonGeometryAlgorithm()

    def name(self):
        return 'calculatepolygongeometry'

    def displayName(self):
        return self.tr('Calculate Polygon Geometry')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'arcgeekcalculator'

    def shortHelpString(self):
        return self.tr("""Calculates and adds area and perimeter fields to polygon layers.

        Parameters:
        - Input polygon layer: Select the layer to process.
        - Area units: Choose the desired unit for area calculation.
        - Perimeter units: Choose the desired unit for perimeter calculation.
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
                self.tr('Input polygon layer'),
                [QgsWkbTypes.PolygonGeometry]
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.AREA_UNITS,
                self.tr('Area units'),
                options=[QgsUnitTypes.toString(unit) for unit in self.area_units],
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.PERIMETER_UNITS,
                self.tr('Perimeter units'),
                options=[QgsUnitTypes.toString(unit) for unit in self.perimeter_units],
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
        area_unit_index = self.parameterAsEnum(parameters, self.AREA_UNITS, context)
        perimeter_unit_index = self.parameterAsEnum(parameters, self.PERIMETER_UNITS, context)
        precision = self.parameterAsInt(parameters, self.PRECISION, context)
        calculation_method = self.parameterAsEnum(parameters, self.CALCULATION_METHOD, context)
        selected_only = self.parameterAsBool(parameters, self.SELECTED_ONLY, context)
        update_existing = self.parameterAsBool(parameters, self.UPDATE_EXISTING, context)

        if not crs.isValid():
            crs = source.crs()

        area_unit = self.area_units[area_unit_index]
        perimeter_unit = self.perimeter_units[perimeter_unit_index]
        area_unit_suffix = self.get_abbreviated_unit_name(area_unit, is_area=True)
        perimeter_unit_suffix = self.get_abbreviated_unit_name(perimeter_unit, is_area=False)
        area_field = f'a_{area_unit_suffix}'
        perimeter_field = f'p_{perimeter_unit_suffix}'

        source_area_unit = QgsUnitTypes.distanceToAreaUnit(crs.mapUnits())
        area_conv_factor = QgsUnitTypes.fromUnitToUnitFactor(source_area_unit, area_unit)
        perimeter_conv_factor = QgsUnitTypes.fromUnitToUnitFactor(crs.mapUnits(), perimeter_unit)

        fields = source.fields()
        if area_field not in fields.names():
            fields.append(QgsField(area_field, QVariant.Double, len=20, prec=precision))
        if perimeter_field not in fields.names():
            fields.append(QgsField(perimeter_field, QVariant.Double, len=20, prec=precision))

        if update_existing:
            if area_field not in source.fields().names():
                source.dataProvider().addAttributes([QgsField(area_field, QVariant.Double, len=20, prec=precision)])
            if perimeter_field not in source.fields().names():
                source.dataProvider().addAttributes([QgsField(perimeter_field, QVariant.Double, len=20, prec=precision)])
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
                area = distance_area.measureArea(feature.geometry())
                perimeter = distance_area.measurePerimeter(feature.geometry())
            else:  # Cartesian
                if transform:
                    transformed_geom = feature.geometry()
                    transformed_geom.transform(transform)
                    area = transformed_geom.area()
                    perimeter = transformed_geom.length()
                else:
                    area = feature.geometry().area()
                    perimeter = feature.geometry().length()

            area = area * area_conv_factor
            perimeter = perimeter * perimeter_conv_factor
            area = round(area, precision)
            perimeter = round(perimeter, precision)

            if update_existing:
                area_idx = source.fields().indexOf(area_field)
                perimeter_idx = source.fields().indexOf(perimeter_field)
                source.changeAttributeValue(feature.id(), area_idx, area)
                source.changeAttributeValue(feature.id(), perimeter_idx, perimeter)
            else:
                new_feature = QgsFeature(fields)
                new_feature.setGeometry(feature.geometry())
                for field in source.fields():
                    new_feature[field.name()] = feature[field.name()]
                new_feature[area_field] = area
                new_feature[perimeter_field] = perimeter
                sink.addFeature(new_feature, QgsFeatureSink.FastInsert)

            feedback.setProgress(int(current * total))

        if update_existing:
            source.commitChanges()

        return {self.OUTPUT: sink.id() if update_existing else dest_id}

    def get_abbreviated_unit_name(self, unit, is_area=True):
        unit_str = QgsUnitTypes.toString(unit).lower()
        if is_area:
            if 'square' in unit_str:
                unit_str = unit_str.replace('square ', '')
            if unit_str == 'meters':
                return 'sqr_m'
            elif unit_str == 'kilometers':
                return 'sqr_km'
            elif unit_str == 'feet':
                return 'sqr_ft'
            elif unit_str == 'yards':
                return 'sqr_yd'
            elif unit_str == 'miles':
                return 'sqr_mi'
            elif unit_str == 'nautical miles':
                return 'sqr_nmi'
            elif unit_str == 'centimeters':
                return 'sqr_cm'
            elif unit_str == 'millimeters':
                return 'sqr_mm'
            elif unit_str == 'hectares':
                return 'ha'
            elif unit_str == 'acres':
                return 'ac'
            else:
                return unit_str
        else:
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