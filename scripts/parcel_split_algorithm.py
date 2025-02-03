from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing, QgsFeatureSink, QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource, QgsProcessingParameterNumber,
                       QgsProcessingParameterFeatureSink, QgsFeature, QgsGeometry,
                       QgsWkbTypes, QgsProcessingException, QgsField, QgsFields,
                       QgsProcessingParameterEnum)
import processing
import math

try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union, split
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
        fields.append(QgsField('split_id', QgsField.Int))
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
            
            try:
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

            except Exception as e:
                feedback.reportError(f"Error processing feature {current}: {str(e)}")

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
        total_area = shapely_geom.area
        target_area = total_area / num_parts
        
        lots = [shapely_geom]
        for _ in range(num_parts - 1):
            largest_lot = max(lots, key=lambda x: x.area)
            split_line = self.get_split_line(largest_lot, target_area)
            new_lots = list(split(largest_lot, split_line))
            lots.remove(largest_lot)
            lots.extend(new_lots)

        return self.handle_small_areas(lots, min_area_ratio * total_area, feedback)

    def split_parcel_by_area(self, shapely_geom, target_area, min_area_ratio, feedback):
        total_area = shapely_geom.area
        num_parts = math.ceil(total_area / target_area)
        return self.split_parcel_equal_parts(shapely_geom, num_parts, min_area_ratio, feedback)

    def get_split_line(self, polygon, target_area):
        bounds = polygon.bounds
        centroid = polygon.centroid
        angle = 0
        best_line = None
        best_diff = float('inf')

        for _ in range(36):  # Try 36 different angles
            dx = math.cos(angle)
            dy = math.sin(angle)
            line = LineString([(centroid.x - dx * 1000, centroid.y - dy * 1000),
                               (centroid.x + dx * 1000, centroid.y + dy * 1000)])
            split_polys = split(polygon, line)
            if len(split_polys) == 2:
                area_diff = abs(split_polys[0].area - target_area)
                if area_diff < best_diff:
                    best_diff = area_diff
                    best_line = line
            angle += math.pi / 36

        return best_line

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
        from .parcel_split_dialog import ParcelSplitParametersDialog
        return ParcelSplitParametersDialog(self, parent)