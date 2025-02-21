from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterFeatureSource, 
                       QgsProcessingParameterBoolean, QgsProcessingParameterVectorDestination, 
                       QgsProcessing, QgsProcessingException, QgsField, QgsFeature,
                       QgsVectorLayer, QgsFeatureRequest, QgsVectorFileWriter, 
                       QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject, QgsProcessingProvider, 
                       QgsProcessingParameterNumber, QgsProcessingParameterCrs, QgsCsException, QgsWkbTypes, QgsGeometry, 
                       QgsFeatureSink, QgsProcessingUtils)
from qgis.PyQt.QtCore import QVariant, QCoreApplication
import traceback
import os

class CoordinateCalculatorAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    MODIFY = 'MODIFY'
    OUTPUT = 'OUTPUT'
    CALCULATE_XY = 'CALCULATE_XY'
    FORMAT_DD = 'FORMAT_DD'
    FORMAT_DMS = 'FORMAT_DMS'
    FORMAT_DMS2 = 'FORMAT_DMS2'
    PRECISION = 'PRECISION'
    CRS = 'CRS'

    def __init__(self):
        super().__init__()

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return CoordinateCalculatorAlgorithm()

    def name(self):
        return 'coordinate_calculator'

    def displayName(self):
        return self.tr('Calculate Coordinates')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'arcgeek_calculator'

    def shortHelpString(self):
        return self.tr("""
        This algorithm calculates and adds coordinate information to a point layer in various formats.

        Parameters:
        - Input layer: The point layer to process.
        - Modify the current layer: If checked, modifies the input layer. Otherwise, creates a new layer.
        - Calculate XY: Adds X and Y fields with coordinates in the specified CRS.
        - Decimal Degrees: Adds latitude and longitude fields in decimal degrees.
        - Degrees Minutes Seconds (DDD° MM' SSS.ss" <N|S|E|W>): Adds fields with coordinates in DMS format.
        - Degrees Minutes Seconds (<N|S|E|W> DDD° MM' SSS.ss"): Adds fields with coordinates in alternative DMS format.
        - Precision: Set the number of decimal places for results.
        - CRS for calculations: Specify a CRS for the calculations. If not specified, the layer's CRS will be used.

        Note: Ensure that your input layer has a defined coordinate reference system (CRS) for accurate results.
        """)

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                self.tr('Input layer'),
                [QgsProcessing.TypeVectorPoint]
            )
        )
        
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.MODIFY,
                self.tr('Modify the current layer'),
                defaultValue=False
            )
        )
        
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT,
                self.tr('Output layer'),
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CALCULATE_XY,
                self.tr('Calculate XY'),
                defaultValue=True
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.FORMAT_DD,
                self.tr('Decimal Degrees'),
                defaultValue=False
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.FORMAT_DMS,
                self.tr('Degrees Minutes Seconds (DDD° MM\' SSS.ss" <N|S|E|W>)'),
                defaultValue=False
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.FORMAT_DMS2,
                self.tr('Degrees Minutes Seconds (<N|S|E|W> DDD° MM\' SSS.ss")'),
                defaultValue=False
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
            QgsProcessingParameterCrs(
                self.CRS,
                self.tr('CRS for calculations'),
                optional=True
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        try:
            source = self.parameterAsSource(parameters, self.INPUT, context)
            if not source:
                raise QgsProcessingException(self.tr("Input layer could not be loaded"))

            modify_layer = self.parameterAsBool(parameters, self.MODIFY, context)
            calculate_xy = self.parameterAsBool(parameters, self.CALCULATE_XY, context)
            format_dd = self.parameterAsBool(parameters, self.FORMAT_DD, context)
            format_dms = self.parameterAsBool(parameters, self.FORMAT_DMS, context)
            format_dms2 = self.parameterAsBool(parameters, self.FORMAT_DMS2, context)
            precision = self.parameterAsInt(parameters, self.PRECISION, context)
            crs = self.parameterAsCrs(parameters, self.CRS, context)

            if not crs.isValid():
                crs = source.sourceCrs()

            feedback.pushInfo(f"Input CRS: {crs.authid()}")
            feedback.pushInfo(f"Input layer feature count: {source.featureCount()}")

            # Prepare fields
            field_definitions = {
                'X': QgsField('X', QVariant.Double, len=20, prec=precision),
                'Y': QgsField('Y', QVariant.Double, len=20, prec=precision),
                'DD_Lat': QgsField('DD_Lat', QVariant.Double, len=20, prec=precision),
                'DD_Lon': QgsField('DD_Lon', QVariant.Double, len=20, prec=precision),
                'DMS_Lat': QgsField('DMS_Lat', QVariant.String, len=20),
                'DMS_Lon': QgsField('DMS_Lon', QVariant.String, len=20),
                'Lat_DMS': QgsField('Lat_DMS', QVariant.String, len=20),
                'Lon_DMS': QgsField('Lon_DMS', QVariant.String, len=20)
            }

            fields_to_add = []
            if calculate_xy:
                fields_to_add.extend(['X', 'Y'])
            if format_dd:
                fields_to_add.extend(['DD_Lat', 'DD_Lon'])
            if format_dms:
                fields_to_add.extend(['DMS_Lat', 'DMS_Lon'])
            if format_dms2:
                fields_to_add.extend(['Lat_DMS', 'Lon_DMS'])

            feedback.pushInfo(f"Fields to be added or updated: {', '.join(fields_to_add)}")

            # Prepare transformations
            transform_to_crs = QgsCoordinateTransform(source.sourceCrs(), crs, QgsProject.instance())
            transform_to_wgs84 = QgsCoordinateTransform(crs, QgsCoordinateReferenceSystem("EPSG:4326"), QgsProject.instance())

            if modify_layer:
                # Modifying the original layer
                layer = self.parameterAsVectorLayer(parameters, self.INPUT, context)
                if not layer:
                    raise QgsProcessingException(self.tr("Could not retrieve the input layer for modification"))
                
                # Remove existing fields if they exist
                fields_to_remove = []
                for field_name in fields_to_add:
                    if layer.fields().indexOf(field_name) != -1:
                        fields_to_remove.append(layer.fields().indexOf(field_name))
                
                if fields_to_remove:
                    layer.dataProvider().deleteAttributes(fields_to_remove)
                    layer.updateFields()

                # Add new fields
                new_fields = [field_definitions[field_name] for field_name in fields_to_add]
                layer.dataProvider().addAttributes(new_fields)
                layer.updateFields()
                
                # Update features
                total_features = layer.featureCount()
                layer.startEditing()

                # Process features in batches
                batch_size = 500
                for i in range(0, total_features, batch_size):
                    if feedback.isCanceled():
                        break
                    
                    start = i
                    end = min(i + batch_size, total_features+1)
                    
                    attr_map = {}
                    feature_ids = list(range(start, end))  # Convert range to list
                    for feature in layer.getFeatures(QgsFeatureRequest().setFilterFids(feature_ids)):
                        if feedback.isCanceled():
                            break
                        
                        geom = feature.geometry()
                        if geom.type() != QgsWkbTypes.PointGeometry:
                            continue
                        
                        point = geom.asPoint()
                        try:
                            point_in_crs = transform_to_crs.transform(point)
                            point_wgs84 = transform_to_wgs84.transform(point_in_crs)
                        except QgsCsException:
                            continue
                        
                        # Prepare attributes
                        attrs = {}
                        if calculate_xy:
                            attrs['X'] = round(point_in_crs.x(), precision)
                            attrs['Y'] = round(point_in_crs.y(), precision)
                        if format_dd:
                            attrs['DD_Lat'] = round(point_wgs84.y(), precision)
                            attrs['DD_Lon'] = round(point_wgs84.x(), precision)
                        if format_dms:
                            attrs['DMS_Lat'] = convert_to_dms(point_wgs84.y(), 'lat', precision)
                            attrs['DMS_Lon'] = convert_to_dms(point_wgs84.x(), 'lon', precision)
                        if format_dms2:
                            attrs['Lat_DMS'] = convert_to_dms2(point_wgs84.y(), 'lat', precision)
                            attrs['Lon_DMS'] = convert_to_dms2(point_wgs84.x(), 'lon', precision)
                        
                        # Prepare attribute map for the feature
                        feature_attr_map = {}
                        for field_name, value in attrs.items():
                            field_index = layer.fields().lookupField(field_name)
                            if field_index != -1:
                                feature_attr_map[field_index] = value
                        
                        attr_map[feature.id()] = feature_attr_map
                    
                    # Update attributes in batch
                    layer.dataProvider().changeAttributeValues(attr_map)
                    
                    feedback.setProgress(int(end / total_features * 100))
                
                layer.commitChanges()
                feedback.pushInfo(f"Modified input layer: {layer.id()}")
                return {self.OUTPUT: layer.id()}
            else:
                # Creating a new layer
                output_fields = source.fields()
                for field_name in fields_to_add:
                    output_fields.append(field_definitions[field_name])
                
                # Create the output layer
                (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                                       output_fields, source.wkbType(), source.sourceCrs())
                if sink is None:
                    raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT))
                
                # Process features
                total_features = source.featureCount()
                for current, feature in enumerate(source.getFeatures()):
                    if feedback.isCanceled():
                        break
                    
                    geom = feature.geometry()
                    if geom.type() != QgsWkbTypes.PointGeometry:
                        sink.addFeature(feature, QgsFeatureSink.FastInsert)
                        continue
                    
                    point = geom.asPoint()
                    try:
                        point_in_crs = transform_to_crs.transform(point)
                        point_wgs84 = transform_to_wgs84.transform(point_in_crs)
                    except QgsCsException:
                        sink.addFeature(feature, QgsFeatureSink.FastInsert)
                        continue
                    
                    new_feature = QgsFeature(output_fields)
                    new_feature.setGeometry(feature.geometry())
                    
                    # Copy existing attributes
                    for i in range(len(source.fields())):
                        new_feature.setAttribute(i, feature.attribute(i))
                    
                    # Add new attributes
                    if calculate_xy:
                        new_feature.setAttribute('X', round(point_in_crs.x(), precision))
                        new_feature.setAttribute('Y', round(point_in_crs.y(), precision))
                    if format_dd:
                        new_feature.setAttribute('DD_Lat', round(point_wgs84.y(), precision))
                        new_feature.setAttribute('DD_Lon', round(point_wgs84.x(), precision))
                    if format_dms:
                        new_feature.setAttribute('DMS_Lat', convert_to_dms(point_wgs84.y(), 'lat', precision))
                        new_feature.setAttribute('DMS_Lon', convert_to_dms(point_wgs84.x(), 'lon', precision))
                    if format_dms2:
                        new_feature.setAttribute('Lat_DMS', convert_to_dms2(point_wgs84.y(), 'lat', precision))
                        new_feature.setAttribute('Lon_DMS', convert_to_dms2(point_wgs84.x(), 'lon', precision))
                    
                    sink.addFeature(new_feature, QgsFeatureSink.FastInsert)
                    feedback.setProgress(int((current + 1) / total_features * 100))
                
                feedback.pushInfo(f"Created new layer: {dest_id}")
                return {self.OUTPUT: dest_id}

        except Exception as e:
            feedback.reportError(self.tr(f"Error in processAlgorithm: {str(e)}"), fatalError=True)
            feedback.pushInfo(traceback.format_exc())
            raise

def convert_to_dms(decimal_degree, coord_type, precision):
    # Converts decimal degrees to DMS format with dynamic precision
    is_positive = decimal_degree >= 0
    decimal_degree = abs(decimal_degree)
    degrees = int(decimal_degree)
    minutes = int((decimal_degree - degrees) * 60)
    seconds = round((decimal_degree - degrees - minutes / 60) * 3600, precision)
    if coord_type == 'lat':
        direction = 'N' if is_positive else 'S'
    else:
        direction = 'E' if is_positive else 'W'
    # Format seconds with dynamic precision
    seconds_formatted = f"{seconds:0{precision+3}.{precision}f}"
    return f"{degrees:2d}° {minutes:02d}' {seconds_formatted}\" {direction}"

def convert_to_dms2(decimal_degree, coord_type, precision):
    # Converts decimal degrees to alternative DMS format with dynamic precision
    is_positive = decimal_degree >= 0
    decimal_degree = abs(decimal_degree)
    degrees = int(decimal_degree)
    minutes = int((decimal_degree - degrees) * 60)
    seconds = round((decimal_degree - degrees - minutes / 60) * 3600, precision)
    if coord_type == 'lat':
        direction = 'N' if is_positive else 'S'
    else:
        direction = 'E' if is_positive else 'W'
    seconds_formatted = f"{seconds:0{precision+3}.{precision}f}"
    return f"{direction} {degrees:2d}° {minutes:02d}' {seconds_formatted}\""

class ArcGeekCalculatorProvider(QgsProcessingProvider):
    def loadAlgorithms(self):
        self.addAlgorithm(CoordinateCalculatorAlgorithm())

    def id(self):
        return "arcgeek_calculator"

    def name(self):
        return self.tr("ArcGeek Calculator")

    def longName(self):
        return self.name()
