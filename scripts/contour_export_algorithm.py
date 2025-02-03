from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessingAlgorithm,
                      QgsProcessingParameterVectorLayer,
                      QgsProcessingParameterField,
                      QgsProcessingParameterFileDestination,
                      QgsProcessingParameterCrs,
                      QgsProcessingParameterEnum,
                      QgsProcessing,
                      QgsVectorLayer,
                      QgsCoordinateReferenceSystem,
                      QgsProcessingException)
import processing
import os

class ContourExportAlgorithm(QgsProcessingAlgorithm):
    """
    Algorithm to export contour lines to 3D CAD while preserving elevation values.
    """
    
    INPUT = 'INPUT'
    ELEVATION_FIELD = 'ELEVATION_FIELD'
    TARGET_CRS = 'TARGET_CRS'
    OUTPUT_FORMAT = 'OUTPUT_FORMAT'
    OUTPUT = 'OUTPUT'

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT,
                self.tr('Input contour layer'),
                [QgsProcessing.TypeVectorLine]
            )
        )
        
        self.addParameter(
            QgsProcessingParameterField(
                self.ELEVATION_FIELD,
                self.tr('Elevation field'),
                type=QgsProcessingParameterField.Numeric,
                parentLayerParameterName=self.INPUT
            )
        )
        
        self.addParameter(
            QgsProcessingParameterCrs(
                self.TARGET_CRS,
                self.tr('Target CRS for CAD export'),
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                self.OUTPUT_FORMAT,
                self.tr('CAD Version'),
                options=['AutoCAD 2000', 'AutoCAD 2004', 'AutoCAD 2007', 'AutoCAD 2010', 'AutoCAD 2013', 'AutoCAD 2018'],
                defaultValue=5
            )
        )
        
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT,
                self.tr('Output CAD file'),
                'DXF files (*.dxf)'
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        elevation_field = self.parameterAsString(parameters, self.ELEVATION_FIELD, context)
        output_file = self.parameterAsString(parameters, self.OUTPUT, context)
        output_format = self.parameterAsEnum(parameters, self.OUTPUT_FORMAT, context)

        if not source:
            raise QgsProcessingException(self.tr('No input layer specified'))
            
        if not elevation_field:
            raise QgsProcessingException(self.tr('No elevation field specified'))
            
        # Validate elevation field
        field_idx = source.fields().indexFromName(elevation_field)
        if field_idx == -1:
            raise QgsProcessingException(self.tr('Elevation field not found in layer'))
            
        # Check if field contains numeric values
        if not source.fields().at(field_idx).isNumeric():
            raise QgsProcessingException(self.tr('Elevation field must contain numeric values'))
            
        # Use input layer CRS if no target CRS specified
        target_crs = self.parameterAsCrs(parameters, self.TARGET_CRS, context)
        if not target_crs.isValid():
            target_crs = source.crs()
            feedback.pushInfo(self.tr('Using input layer CRS: {}'.format(target_crs.authid())))

        # Step 1: Check if input is already 3D
        feedback.pushInfo(self.tr('Processing contours...'))
        
        is_3d = source.wkbType() > 1000  # WKB types > 1000 are 3D
        
        if is_3d:
            feedback.pushInfo(self.tr('Input layer is already 3D, using directly...'))
            input_3d_layer = source.source()
        else:
            feedback.pushInfo(self.tr('Converting to 3D using elevation field...'))
            grass_output = processing.run(
                "grass7:v.to.3d",
                {
                    'input': source,
                    'type': [1],  # line
                    'column': elevation_field,
                    'output': 'TEMPORARY_OUTPUT'
                },
                context=context,
                feedback=feedback
            )
            input_3d_layer = grass_output['output']

        if feedback.isCanceled():
            return {}
            
        # Step 2: Export to DXF
        feedback.pushInfo(self.tr('Exporting to CAD format...'))

        # Set DXF version
        dxf_versions = ['R2000', 'R2004', 'R2007', 'R2010', 'R2013', 'R2018']
        dxf_version = dxf_versions[output_format]
        
        try:
            # Configure export parameters
            dxf_params = {
                'LAYERS': [{
                    'layer': input_3d_layer,
                    'attributeIndex': -1,
                    'overriddenLayerName': '',
                    'buildDataDefinedBlocks': True,
                    'dataDefinedBlocksMaximumNumberOfClasses': -1
                }],
                'CRS': target_crs,
                'ENCODING': 'cp1252',
                'FORCE_2D': False,
                'EXPORT_LINES_WITH_ZERO_WIDTH': False,
                'SYMBOLOGY_MODE': 0,
                'SYMBOLOGY_SCALE': 1000000,
                'MAP_THEME': None,
                'EXTENT': None,
                'MTEXT': False,
                'SELECTED_FEATURES_ONLY': False,
                'DXF_VERSION': dxf_version,
                'OUTPUT': output_file
            }
            
            # Execute the export
            result_dxf = processing.run(
                "native:dxfexport",
                dxf_params,
                context=context,
                feedback=feedback
            )
            
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                feedback.pushInfo(
                    self.tr('Successfully exported 3D contours to: {} ({:.2f} MB)'.format(
                        output_file, file_size / (1024 * 1024)))
                )
            else:
                feedback.pushWarning(self.tr('Output file was not created'))
            
            return {self.OUTPUT: output_file}
            
        except Exception as e:
            feedback.reportError(f"Error during DXF export: {str(e)}")
            raise e

    def name(self):
        return 'contourexport'

    def displayName(self):
        return self.tr('Export Contours to 3D CAD')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'arcgeekcalculator'

    def shortHelpString(self):
        return self.tr("""
        This algorithm exports contour lines to a 3D CAD file (DXF format) that preserves elevation values,
        making it compatible with CAD software like AutoCAD, Civil 3D, BricsCAD, and other CAD applications
        that support 3D DXF files.

        Parameters:
            Input contour layer: Vector line layer containing contours
            Elevation field: Numeric field containing elevation values
            Target CRS: Coordinate Reference System for the CAD export (optional)
            CAD Version: AutoCAD version compatibility
            Output CAD file: Destination path for the exported DXF

        The algorithm automatically detects if your input is already 3D and:
        - If 3D: Uses it directly
        - If 2D: Converts it to 3D using the elevation field

        The output DXF file will maintain Z-coordinates from your elevation values,
        making it ready to use in CAD software.

        Note: For best compatibility with older CAD software, use AutoCAD 2000 format.
        """)

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return ContourExportAlgorithm()