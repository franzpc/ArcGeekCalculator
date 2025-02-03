from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QComboBox, QDialog, QDialogButtonBox
from qgis.core import (QgsProcessing, QgsFeatureSink, QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource, QgsProcessingParameterNumber,
                       QgsProcessingParameterFeatureSink, QgsFeature, QgsGeometry,
                       QgsWkbTypes, QgsProcessingException, QgsField, QgsFields,
                       QgsProcessingParameterEnum)
import processing
import math

try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

class ParcelSplitAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    SPLIT_TYPE = 'SPLIT_TYPE'
    WIDTH = 'WIDTH'
    LENGTH = 'LENGTH'
    NUM_PARTS = 'NUM_PARTS'
    TARGET_AREA = 'TARGET_AREA'
    MIN_AREA_RATIO = 'MIN_AREA_RATIO'
    OUTPUT = 'OUTPUT'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT, 'Input polygon layer', [QgsProcessing.TypeVectorPolygon]))
        self.addParameter(QgsProcessingParameterEnum(self.SPLIT_TYPE, 'Split type', options=['By measures', 'Equal parts', 'By area'], defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.WIDTH, 'Desired lot width', QgsProcessingParameterNumber.Double, optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.LENGTH, 'Desired lot length', QgsProcessingParameterNumber.Double, optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.NUM_PARTS, 'Number of parts to split into', QgsProcessingParameterNumber.Integer, optional=True, minValue=2))
        self.addParameter(QgsProcessingParameterNumber(self.TARGET_AREA, 'Target area for each lot', QgsProcessingParameterNumber.Double, optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.MIN_AREA_RATIO, 'Minimum area ratio for annexation', QgsProcessingParameterNumber.Double, defaultValue=0.05, minValue=0, maxValue=0.5))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, 'Output lots'))

    def processAlgorithm(self, parameters, context, feedback):
        if not SHAPELY_AVAILABLE:
            feedback.reportError("Shapely is not installed. Please install Shapely to use this tool.")
            return {}

        source = self.parameterAsSource(parameters, self.INPUT, context)
        split_type = self.parameterAsEnum(parameters, self.SPLIT_TYPE, context)
        min_area_ratio = self.parameterAsDouble(parameters, self.MIN_AREA_RATIO, context)

        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        fields = source.fields()
        fields.append(QgsField('split_id', QgsField.Integer))
        fields.append(QgsField('area', QgsField.Double))

        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                               fields, QgsWkbTypes.Polygon, source.sourceCrs())

        if sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT))

        total = 100.0 / source.featureCount() if source.featureCount() else 0
        features = source.getFeatures()

        for current, feature in enumerate(features):
            if feedback.isCanceled():
                break

            geom = feature.geometry()
            if geom.isMultipart():
                geometries = [Polygon([(p.x(), p.y()) for p in part]) for part in geom.asMultiPolygon()]
                shapely_geom = MultiPolygon(geometries)
            else:
                shapely_geom = Polygon([(p.x(), p.y()) for p in geom.asPolygon()[0]])
            
            if split_type == 0:  # By measures
                width = self.parameterAsDouble(parameters, self.WIDTH, context)
                length = self.parameterAsDouble(parameters, self.LENGTH, context)
                if width <= 0 or length <= 0:
                    raise QgsProcessingException("Width and length must be greater than 0 for 'By measures' split type.")
                lots = self.split_parcel_by_measures(shapely_geom, width, length, min_area_ratio, feedback)
            elif split_type == 1:  # Equal parts
                num_parts = self.parameterAsInt(parameters, self.NUM_PARTS, context)
                if num_parts < 2:
                    raise QgsProcessingException("Number of parts must be at least 2 for 'Equal parts' split type.")
                lots = self.split_parcel_equal_parts(shapely_geom, num_parts, min_area_ratio, feedback)
            else:  # By area
                target_area = self.parameterAsDouble(parameters, self.TARGET_AREA, context)
                if target_area <= 0:
                    raise QgsProcessingException("Target area must be greater than 0 for 'By area' split type.")
                lots = self.split_parcel_by_area(shapely_geom, target_area, min_area_ratio, feedback)

            for i, lot in enumerate(lots):
                f = QgsFeature()
                f.setGeometry(QgsGeometry.fromWkt(lot.wkt))
                attributes = feature.attributes()
                attributes.append(i + 1)
                attributes.append(lot.area)
                f.setAttributes(attributes)
                sink.addFeature(f, QgsFeatureSink.FastInsert)

            feedback.setProgress(int(current * total))

        return {self.OUTPUT: dest_id}

    def split_parcel_by_measures(self, shapely_geom, width, length, min_area_ratio, feedback):
        lots = []
        bounds = shapely_geom.bounds
        parcel_width = bounds[2] - bounds[0]
        parcel_length = bounds[3] - bounds[1]
        
        num_width = math.floor(parcel_width / width)
        num_length = math.floor(parcel_length / length)
        
        for i in range(num_width):
            for j in range(num_length):
                lot = Polygon([
                    (bounds[0] + i * width, bounds[1] + j * length),
                    (bounds[0] + (i+1) * width, bounds[1] + j * length),
                    (bounds[0] + (i+1) * width, bounds[1] + (j+1) * length),
                    (bounds[0] + i * width, bounds[1] + (j+1) * length),
                    (bounds[0] + i * width, bounds[1] + j * length)
                ])
                if shapely_geom.intersects(lot):
                    lots.append(shapely_geom.intersection(lot))

        return self.handle_small_areas(lots, min_area_ratio * shapely_geom.area, feedback)

    def split_parcel_equal_parts(self, shapely_geom, num_parts, min_area_ratio, feedback):
        target_area = shapely_geom.area / num_parts
        bounds = shapely_geom.bounds
        width = (bounds[2] - bounds[0]) / math.sqrt(num_parts)
        length = (bounds[3] - bounds[1]) / math.sqrt(num_parts)
        
        return self.split_parcel_by_measures(shapely_geom, width, length, min_area_ratio, feedback)

    def split_parcel_by_area(self, shapely_geom, target_area, min_area_ratio, feedback):
        num_parts = math.ceil(shapely_geom.area / target_area)
        return self.split_parcel_equal_parts(shapely_geom, num_parts, min_area_ratio, feedback)

    def handle_small_areas(self, lots, min_area, feedback):
        small_lots = [lot for lot in lots if lot.area < min_area]
        large_lots = [lot for lot in lots if lot.area >= min_area]
        
        for small_lot in small_lots:
            best_neighbor = max(large_lots, key=lambda x: x.boundary.intersection(small_lot.boundary).length)
            best_neighbor = unary_union([best_neighbor, small_lot])
            large_lots[large_lots.index(max(large_lots, key=lambda x: x.boundary.intersection(small_lot.boundary).length))] = best_neighbor
        
        return large_lots

    def name(self):
        return 'parcelsplit'

    def displayName(self):
        return 'Parcel Split'

    def group(self):
        return 'Vector'

    def groupId(self):
        return 'vector'

    def createInstance(self):
        return ParcelSplitAlgorithm()

    def shortHelpString(self):
        return "Splits parcels using different methods: by measures, equal parts, or by area."

    def createCustomParametersWidget(self, parent):
        return ParcelSplitParametersDialog(self, parent)

class ParcelSplitParametersWidget(QWidget):
    def __init__(self, alg, parent=None):
        super().__init__(parent)
        self.alg = alg
        self.setupUi()

    def setupUi(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.split_type_combo = QComboBox()
        self.split_type_combo.addItems(['By measures', 'Equal parts', 'By area'])
        self.split_type_combo.currentIndexChanged.connect(self.updateFieldVisibility)
        layout.addWidget(QLabel('Split type:'))
        layout.addWidget(self.split_type_combo)

        self.width_input = QLineEdit()
        self.length_input = QLineEdit()
        self.num_parts_input = QLineEdit()
        self.target_area_input = QLineEdit()

        layout.addWidget(QLabel('Desired lot width:'))
        layout.addWidget(self.width_input)
        layout.addWidget(QLabel('Desired lot length:'))
        layout.addWidget(self.length_input)
        layout.addWidget(QLabel('Number of parts to split into:'))
        layout.addWidget(self.num_parts_input)
        layout.addWidget(QLabel('Target area for each lot:'))
        layout.addWidget(self.target_area_input)

        self.min_area_ratio_input = QLineEdit()
        self.min_area_ratio_input.setText('0.05')
        layout.addWidget(QLabel('Minimum area ratio for annexation:'))
        layout.addWidget(self.min_area_ratio_input)

        self.updateFieldVisibility(0)

    def updateFieldVisibility(self, index):
        self.width_input.setVisible(index == 0)
        self.length_input.setVisible(index == 0)
        self.num_parts_input.setVisible(index == 1)
        self.target_area_input.setVisible(index == 2)

    def getParameters(self):
        parameters = {}
        parameters[self.alg.SPLIT_TYPE] = self.split_type_combo.currentIndex()
        parameters[self.alg.WIDTH] = self.width_input.text()
        parameters[self.alg.LENGTH] = self.length_input.text()
        parameters[self.alg.NUM_PARTS] = self.num_parts_input.text()
        parameters[self.alg.TARGET_AREA] = self.target_area_input.text()
        parameters[self.alg.MIN_AREA_RATIO] = self.min_area_ratio_input.text()
        return parameters

    def setParameters(self, parameters):
        if self.alg.SPLIT_TYPE in parameters:
            self.split_type_combo.setCurrentIndex(int(parameters[self.alg.SPLIT_TYPE]))
        if self.alg.WIDTH in parameters:
            self.width_input.setText(str(parameters[self.alg.WIDTH]))
        if self.alg.LENGTH in parameters:
            self.length_input.setText(str(parameters[self.alg.LENGTH]))
        if self.alg.NUM_PARTS in parameters:
            self.num_parts_input.setText(str(parameters[self.alg.NUM_PARTS]))
        if self.alg.TARGET_AREA in parameters:
            self.target_area_input.setText(str(parameters[self.alg.TARGET_AREA]))
        if self.alg.MIN_AREA_RATIO in parameters:
            self.min_area_ratio_input.setText(str(parameters[self.alg.MIN_AREA_RATIO]))

class ParcelSplitParametersDialog(QDialog):
    def __init__(self, alg, parent=None):
        super().__init__(parent)
        self.alg = alg
        self.setupUi()

    def setupUi(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.widget = ParcelSplitParametersWidget(self.alg, self)
        layout.addWidget(self.widget)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def getParameters(self):
        return self.widget.getParameters()

    def setParameters(self, parameters):
        self.widget.setParameters(parameters)