from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterNumber, QgsProcessingParameterEnum,
                       QgsProcessingParameterString, QgsVectorLayer, QgsFeature,
                       QgsProcessingParameterFeatureSink, QgsGeometry, QgsPointXY,
                       QgsProcessingOutputNumber, QgsWkbTypes, QgsFeatureSink,
                       QgsField, QgsFields, QgsProcessingUtils, QgsProcessingException,
                       QgsProcessing, QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterBoolean, QgsRasterLayer,
                       QgsCoordinateTransform, QgsProject, QgsRasterBandStats, QgsRaster)
import math
import re
import processing

class TreePlantingPatternAlgorithm(QgsProcessingAlgorithm):
    """
    This algorithm generates tree planting points with different patterns inside polygon features.
    """
    
    INPUT = 'INPUT'
    PATTERN_TYPE = 'PATTERN_TYPE'
    BORDER_MARGIN = 'BORDER_MARGIN'
    SPACING = 'SPACING'
    DEM_LAYER = 'DEM_LAYER'
    SLOPE_ANALYSIS = 'SLOPE_ANALYSIS'
    OUTPUT = 'OUTPUT'
    OUTPUT_TRIANGULAR_COUNT = 'OUTPUT_TRIANGULAR_COUNT'
    OUTPUT_RECTANGULAR_COUNT = 'OUTPUT_RECTANGULAR_COUNT'
    OUTPUT_CINCOOROS_COUNT = 'OUTPUT_CINCOOROS_COUNT'
    OUTPUT_KEYLINE_COUNT = 'OUTPUT_KEYLINE_COUNT'
    OUTPUT_RANDOM_COUNT = 'OUTPUT_RANDOM_COUNT'

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT,
                self.tr('Input polygon layer'),
                [QgsProcessing.TypeVectorPolygon]
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                self.PATTERN_TYPE,
                self.tr('Planting pattern'),
                options=['Triangular/Quincunx (Tresbolillo)', 'Rectangular/Square', 'Five of Diamonds (Cinco de Oros)', 'Keyline (requires DEM)', 'Random Distribution'],
                defaultValue=0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BORDER_MARGIN,
                self.tr('Border margin (distance from boundary)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.5,
                minValue=0.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterString(
                self.SPACING,
                self.tr('Spacing (supports 3x3, 3*3, 3-3)'),
                defaultValue='3x3'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.DEM_LAYER,
                self.tr('DEM Layer (optional - if provided: keyline pattern available + automatic slope analysis)'),
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr('Output planting points'),
                QgsProcessing.TypeVectorPoint
            )
        )
        
        self.addOutput(QgsProcessingOutputNumber(self.OUTPUT_TRIANGULAR_COUNT, self.tr('Triangular/Quincunx planting count')))
        self.addOutput(QgsProcessingOutputNumber(self.OUTPUT_RECTANGULAR_COUNT, self.tr('Rectangular/Square planting count')))
        self.addOutput(QgsProcessingOutputNumber(self.OUTPUT_CINCOOROS_COUNT, self.tr('Five of Diamonds planting count')))
        self.addOutput(QgsProcessingOutputNumber(self.OUTPUT_KEYLINE_COUNT, self.tr('Keyline planting count')))
        self.addOutput(QgsProcessingOutputNumber(self.OUTPUT_RANDOM_COUNT, self.tr('Random distribution planting count')))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        pattern_type = self.parameterAsEnum(parameters, self.PATTERN_TYPE, context)
        border_margin = self.parameterAsDouble(parameters, self.BORDER_MARGIN, context)
        spacing_str = self.parameterAsString(parameters, self.SPACING, context)
        dem_layer = self.parameterAsRasterLayer(parameters, self.DEM_LAYER, context)
        
        # NUEVA LÓGICA: Si hay DEM = análisis automático, si no hay = sin análisis
        slope_analysis = dem_layer is not None
        
        try:
            spacing_x, spacing_y = self.parse_spacing(spacing_str)
        except (ValueError, TypeError) as e:
            raise QgsProcessingException(f"Invalid spacing format: {e}")
        
        if source is None:
            raise QgsProcessingException("Could not load source layer")
        
        # Ya no necesitamos validar slope_analysis por separado porque es automático
        has_dem = dem_layer is not None
        if pattern_type == 3 and not has_dem:
            raise QgsProcessingException("Keyline pattern requires a DEM layer")
        
        # Create output layer fields - INCLUIR slope field desde el inicio si es necesario
        fields = QgsFields()
        fields.append(QgsField('id', QVariant.Int))
        fields.append(QgsField('pattern', QVariant.String))
        fields.append(QgsField('x', QVariant.Double))
        fields.append(QgsField('y', QVariant.Double))
        
        # Agregar campo slope desde el inicio si se solicita análisis
        if slope_analysis:
            fields.append(QgsField('slope', QVariant.Double, 'Slope (degrees)', 10, 2))
        
        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT, context, fields, QgsWkbTypes.Point, source.crs()
        )
        
        if sink is None:
            raise QgsProcessingException("Could not create output layer")
        
        # Initialize counters
        triangular_count = 0
        rectangular_count = 0
        cincooros_count = 0
        keyline_count = 0
        random_count = 0
        total_area = 0
        
        # Generate slope raster AUTOMÁTICAMENTE si hay DEM
        slope_layer = None
        if has_dem:  # Si hay DEM, hacer análisis automáticamente
            feedback.pushInfo("DEM detected - performing automatic slope analysis...")
            slope_layer = self.generate_slope_map(dem_layer, source, context, feedback)
            if not slope_layer or not slope_layer.isValid():
                feedback.pushWarning("Could not generate slope map, continuing without slope analysis...")
                slope_analysis = False  # Desactivar si falla
            else:
                feedback.pushInfo("Slope analysis will be included automatically.")
        else:
            feedback.pushInfo("No DEM provided - slope analysis disabled.")
        
        # Prepare CRS transformation if needed
        transform = None
        if slope_analysis and slope_layer and source.crs() != slope_layer.crs():
            transform = QgsCoordinateTransform(source.crs(), slope_layer.crs(), context.project())
            feedback.pushInfo(f"CRS transform for slope sampling prepared ({source.crs().authid()} -> {slope_layer.crs().authid()}).")
        
        # Prepare streams layer for keyline if needed - OPTIMIZADO PARA REUTILIZAR DEM
        streams_layer = None
        clipped_dem = None
        if pattern_type == 3 and has_dem:
            feedback.pushInfo("Preparing Keyline pattern (optimized for performance)...")
            # OPTIMIZACIÓN: Reutilizar el DEM ya clipeado si existe, sino usar el original
            if slope_layer:
                # Intentar reutilizar el área ya clipeada para pendientes
                feedback.pushInfo("Reusing clipped DEM area for Keyline processing...")
                clipped_dem = slope_layer  # Reutilizar si es posible
            else:
                # Solo clipear si no se hizo antes
                union_geom = QgsGeometry()
                for feature in source.getFeatures():
                    if union_geom.isEmpty():
                        union_geom = feature.geometry()
                    else:
                        union_geom = union_geom.combine(feature.geometry())
                
                buffered_union = union_geom.buffer(max(spacing_x, spacing_y) * 2, 5)
                clipped_dem = self.clip_dem_to_polygon(dem_layer, buffered_union, context, feedback)
            
            # Generar streams solo si tenemos DEM válido
            if clipped_dem and clipped_dem.isValid():
                streams_layer = self.generate_stream_network(clipped_dem, context, feedback)
            else:
                feedback.pushWarning("Could not prepare DEM for Keyline, using simplified approach...")
                clipped_dem = dem_layer  # Usar DEM original como fallback
        
        # Process each feature
        total_features = source.featureCount()
        feature_id_counter = 1
        
        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
                
            feedback.setProgress(int(current / total_features * 100))
            
            geom = feature.geometry()
            if geom.isEmpty():
                continue
            
            feature_area = geom.area()
            total_area += feature_area
            
            # Apply buffer
            working_geom = geom
            if border_margin > 0:
                buffered_geom = geom.buffer(-border_margin, 5)
                if buffered_geom.isEmpty():
                    feedback.pushInfo(f"Feature {feature.id()} became empty after applying border margin")
                    continue
                working_geom = buffered_geom
            
            bbox = working_geom.boundingBox()
            
            # Generate points for all patterns to count them
            triangular_points = self.generate_triangular_points(working_geom, bbox, min(spacing_x, spacing_y))
            rectangular_points = self.generate_rectangular_points(working_geom, bbox, spacing_x, spacing_y)
            all_cincooros_points = self.generate_cincooros_points(working_geom, bbox, spacing_x, spacing_y)
            
            triangular_count += len(triangular_points)
            rectangular_count += len(rectangular_points)
            cincooros_count += len(all_cincooros_points)
            
            # Generate keyline points if selected
            keyline_points = []
            if pattern_type == 3:
                keyline_points = self.generate_keyline_points_original(working_geom, clipped_dem, streams_layer, spacing_y, spacing_x, context, feedback)
                keyline_count += len(keyline_points)
            
            # Generate random points if selected
            random_points = []
            if pattern_type == 4:
                random_points = self.generate_random_points(working_geom, spacing_x, spacing_y, source.crs(), context, feedback)
                random_count += len(random_points)
            
            # Add points to output based on selected pattern
            if pattern_type == 0:  # Triangular
                pattern_name = 'Triangular'
                selected_points = [(p, pattern_name) for p in triangular_points]
            elif pattern_type == 1:  # Rectangular
                pattern_name = 'Rectangular' if spacing_x != spacing_y else 'Square'
                selected_points = [(p, pattern_name) for p in rectangular_points]
            elif pattern_type == 2:  # Five of Diamonds
                pattern_name = 'Five of Diamonds'
                selected_points = [(p[0], f"{pattern_name}_{p[1]}") for p in all_cincooros_points]
            elif pattern_type == 3:  # Keyline
                pattern_name = 'Keyline'
                selected_points = [(p, pattern_name) for p in keyline_points]
            elif pattern_type == 4:  # Random
                pattern_name = 'Random'
                selected_points = [(p, pattern_name) for p in random_points]
            
            # Add features to sink CON slope analysis directamente - METODOLOGÍA DEL NUEVO SCRIPT
            for point, p_type in selected_points:
                feat = QgsFeature()
                feat.setGeometry(QgsGeometry.fromPointXY(point))
                
                # Basic attributes
                attributes = [feature_id_counter, p_type, point.x(), point.y()]
                
                # Add slope analysis if requested - USANDO LA METODOLOGÍA DIRECTA DEL NUEVO SCRIPT
                if slope_analysis and slope_layer:
                    slope_value = self.sample_slope_at_point(point, slope_layer, transform)
                    attributes.append(slope_value)
                elif slope_analysis:
                    attributes.append(None)
                
                feat.setAttributes(attributes)
                sink.addFeature(feat, QgsFeatureSink.FastInsert)
                feature_id_counter += 1
        
        # Calculate areas in hectares
        total_area_ha = total_area / 10000
        
        # Generate report
        self.generate_report(feedback, pattern_type, spacing_x, spacing_y, total_area_ha, 
                            triangular_count, rectangular_count, cincooros_count, keyline_count, random_count, slope_analysis)
        
        return {
            self.OUTPUT: dest_id,
            self.OUTPUT_TRIANGULAR_COUNT: triangular_count,
            self.OUTPUT_RECTANGULAR_COUNT: rectangular_count,
            self.OUTPUT_CINCOOROS_COUNT: cincooros_count,
            self.OUTPUT_KEYLINE_COUNT: keyline_count,
            self.OUTPUT_RANDOM_COUNT: random_count
        }

    def generate_slope_map(self, dem_layer, source_layer, context, feedback):
        """Generates a slope map from the DEM, clipped to the source layer's extent. - COPIADO DEL NUEVO SCRIPT"""
        try:
            feedback.pushInfo("Creating union geometry of all polygons...")
            union_geom = QgsGeometry()
            for feature in source_layer.getFeatures():
                union_geom = union_geom.combine(feature.geometry()) if not union_geom.isEmpty() else feature.geometry()
            
            if union_geom.isEmpty():
                feedback.pushWarning("The union geometry of the polygons is empty.")
                return None

            buffered_union = union_geom.buffer(50, 5) # Buffer to avoid edge effects
            
            feedback.pushInfo("Clipping DEM to polygon area...")
            clipped_dem = self.clip_dem_to_polygon(dem_layer, buffered_union, context, feedback)
            
            if not clipped_dem or not clipped_dem.isValid():
                feedback.pushWarning("DEM clipping failed. Cannot generate slope map.")
                return None
            
            feedback.pushInfo(f"Clipped DEM created. Extent: {clipped_dem.extent().toString()}")
            feedback.pushInfo("Calculating slope from clipped DEM...")
            
            result = processing.run('native:slope', {'INPUT': clipped_dem, 'Z_FACTOR': 1, 'OUTPUT': 'TEMPORARY_OUTPUT'}, 
                                    context=context, feedback=feedback, is_child_algorithm=True)
            
            slope_layer = QgsProcessingUtils.mapLayerFromString(result['OUTPUT'], context)
            if not slope_layer or not slope_layer.isValid():
                feedback.pushWarning("The resulting slope raster is not valid.")
                return None
            
            # Diagnostic: Report stats of the generated slope raster
            stats = slope_layer.dataProvider().bandStatistics(1, QgsRasterBandStats.All)
            feedback.pushInfo(f"Statistics for generated slope raster: MIN={stats.minimumValue:.2f}, MAX={stats.maximumValue:.2f}")

            feedback.pushInfo("Slope map generated successfully.")
            return slope_layer
            
        except Exception as e:
            feedback.pushInfo(f"Error generating slope map: {e}")
            return None

    def sample_slope_at_point(self, point, slope_layer, transform=None):
        """Sample slope value at a specific point - COPIADO DEL NUEVO SCRIPT"""
        try:
            point_for_sampling = QgsPointXY(point.x(), point.y())
            
            # Transform point if needed
            if transform:
                point_geom = QgsGeometry.fromPointXY(point_for_sampling)
                point_geom.transform(transform)
                point_for_sampling = point_geom.asPoint()
            
            # Use identify method for robust sampling
            identify_result = slope_layer.dataProvider().identify(point_for_sampling, QgsRaster.IdentifyFormatValue)
            
            if identify_result and identify_result.isValid():
                slope_value = list(identify_result.results().values())[0]
                return float(slope_value) if slope_value is not None else None
            else:
                return None
                
        except Exception as e:
            return None

    def parse_spacing(self, spacing_str):
        """Parse spacing string with flexible separators"""
        separators = ['x', '*', '-', ':']
        
        match = None
        for sep in separators:
            pattern = rf'(\d+(?:\.\d+)?){re.escape(sep)}(\d+(?:\.\d+)?)'
            match = re.match(pattern, spacing_str)
            if match:
                spacing_x = float(match.group(1))
                spacing_y = float(match.group(2))
                break
        
        if not match:
            try:
                spacing_x = float(spacing_str)
                spacing_y = spacing_x
            except ValueError:
                raise ValueError(f"Invalid spacing format: '{spacing_str}'")
        
        if spacing_x <= 0 or spacing_y <= 0:
            raise ValueError("Spacing values must be positive")
        
        return spacing_x, spacing_y

    def clip_dem_to_polygon(self, dem_layer, polygon_geom, context, feedback):
        """Clip DEM to polygon boundary"""
        try:
            temp_layer = QgsVectorLayer(f"Polygon?crs={dem_layer.crs().authid()}", "temp_clip", "memory")
            temp_provider = temp_layer.dataProvider()
            temp_feature = QgsFeature()
            temp_feature.setGeometry(polygon_geom)
            temp_provider.addFeatures([temp_feature])
            
            clip_params = {
                'INPUT': dem_layer,
                'MASK': temp_layer,
                'SOURCE_CRS': dem_layer.crs(),
                'TARGET_CRS': dem_layer.crs(),
                'NODATA': -9999,
                'ALPHA_BAND': False,
                'CROP_TO_CUTLINE': True,
                'KEEP_RESOLUTION': True,
                'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
            }
            
            result = processing.run("gdal:cliprasterbymasklayer", clip_params, 
                                   context=context, feedback=feedback, is_child_algorithm=True)
            
            clipped_layer = QgsRasterLayer(result['OUTPUT'], 'Clipped DEM')
            if not clipped_layer.isValid():
                feedback.pushWarning("DEM clipping resulted in an invalid layer.")
                return None
            return clipped_layer
            
        except Exception as e:
            feedback.pushInfo(f"Error clipping DEM: {e}")
            return dem_layer

    def generate_stream_network(self, dem_layer, context, feedback):
        """Generate stream network from DEM"""
        try:
            feedback.pushInfo("Generating stream network for keyline analysis...")
            
            filled_dem = processing.run("grass7:r.fill.dir", {
                'input': dem_layer,
                'output': QgsProcessing.TEMPORARY_OUTPUT,
                'direction': QgsProcessing.TEMPORARY_OUTPUT,
                'areas': QgsProcessing.TEMPORARY_OUTPUT,
                'format': 0
            }, context=context, feedback=feedback)['output']
            
            flow_accumulation = processing.run("grass7:r.watershed", {
                'elevation': filled_dem,
                'accumulation': QgsProcessing.TEMPORARY_OUTPUT,
                'drainage': QgsProcessing.TEMPORARY_OUTPUT,
                'threshold': dem_layer.rasterUnitsPerPixelX(),
                '-s': True,
                '-m': True
            }, context=context, feedback=feedback)['accumulation']
            
            area_km2 = dem_layer.extent().width() * dem_layer.extent().height() / 1000000
            threshold = max(500, int(area_km2 * 100))
            
            streams = processing.run("grass7:r.stream.extract", {
                'elevation': filled_dem,
                'accumulation': flow_accumulation,
                'threshold': threshold,
                'stream_vector': QgsProcessing.TEMPORARY_OUTPUT,
                'stream_raster': QgsProcessing.TEMPORARY_OUTPUT,
                'direction': QgsProcessing.TEMPORARY_OUTPUT,
                'GRASS_OUTPUT_TYPE_PARAMETER': 2
            }, context=context, feedback=feedback)['stream_vector']
            
            streams_layer = QgsProcessingUtils.mapLayerFromString(streams, context)
            feedback.pushInfo(f"Generated {streams_layer.featureCount()} stream segments")
            return streams_layer
            
        except Exception as e:
            feedback.pushInfo(f"Error generating stream network: {e}")
            return None

    def generate_keyline_points_original(self, geom, dem_layer, streams_layer, contour_interval, spacing, context, feedback):
        """Generate keyline points: continuous around stream network + contour continuation"""
        points = []
        
        try:
            feedback.pushInfo("Generating keyline pattern...")
            
            # Step 1: Generate points around entire stream network continuously
            stream_points = []
            if streams_layer:
                # Collect all stream geometries within polygon
                all_stream_lines = []
                for stream_feature in streams_layer.getFeatures():
                    stream_geom = stream_feature.geometry()
                    
                    if stream_geom.intersects(geom):
                        intersection = stream_geom.intersection(geom)
                        
                        if not intersection.isEmpty() and intersection.type() == QgsWkbTypes.LineGeometry:
                            if intersection.isMultipart():
                                lines = intersection.asMultiPolyline()
                                all_stream_lines.extend(lines)
                            else:
                                all_stream_lines.append(intersection.asPolyline())
                
                # Generate continuous points around all streams
                if all_stream_lines:
                    stream_points = self.generate_continuous_stream_points(all_stream_lines, spacing, geom)
            
            feedback.pushInfo(f"Generated {len(stream_points)} continuous points around streams")
            
            # Step 2: Generate contour lines
            contour_points = []
            try:
                contour_params = {
                    'INPUT': dem_layer,
                    'BAND': 1,
                    'INTERVAL': contour_interval,
                    'FIELD_NAME': 'ELEV',
                    'OUTPUT': 'TEMPORARY_OUTPUT'
                }
                
                contour_result = processing.run('gdal:contour', contour_params, 
                                              context=context, feedback=feedback, is_child_algorithm=True)
                
                contour_layer = QgsProcessingUtils.mapLayerFromString(contour_result['OUTPUT'], context)
                
                # Step 3: Find closest stream point to start contour continuation
                if stream_points and contour_layer.featureCount() > 0:
                    closest_stream_point = self.find_closest_point_to_contours(stream_points, contour_layer, geom)
                    
                    if closest_stream_point:
                        contour_points = self.generate_contour_continuation_points(
                            contour_layer, closest_stream_point, spacing, geom, stream_points
                        )
                
                feedback.pushInfo(f"Generated {len(contour_points)} contour continuation points")
                
            except Exception as e:
                feedback.pushInfo(f"Contour generation error: {e}")
            
            # Step 4: Combine points intelligently avoiding duplicates between sources
            if stream_points and contour_points:
                # First add all stream points (they have priority)
                final_points = stream_points.copy()
                
                # Then add contour points that are not too close to stream points
                for contour_point in contour_points:
                    too_close_to_stream_point = False
                    
                    for stream_point in stream_points:
                        if contour_point.distance(stream_point) < spacing:
                            too_close_to_stream_point = True
                            break
                    
                    if not too_close_to_stream_point:
                        final_points.append(contour_point)
                
                points = final_points
            else:
                # If only one type of points exists, use them directly
                points = stream_points + contour_points
            
            # Final cleanup: remove points too close to stream lines (keep this as additional safety)
            if streams_layer and points:
                cleaned_points = self.remove_points_too_close_to_streams(points, streams_layer, geom, spacing * 0.8)  # Use 80% tolerance
                points = cleaned_points
            
            feedback.pushInfo(f"Total keyline points after intelligent combination: {len(points)}")
            
        except Exception as e:
            feedback.pushInfo(f"Keyline generation error: {e}")
        
        return points

    def remove_points_too_close_to_streams(self, all_points, streams_layer, polygon_geom, min_distance):
        """Remove all points that are closer than min_distance to any stream"""
        cleaned_points = []
        
        try:
            # Create a list of all stream geometries within the polygon
            stream_geometries = []
            for stream_feature in streams_layer.getFeatures():
                stream_geom = stream_feature.geometry()
                
                if stream_geom.intersects(polygon_geom):
                    intersection = stream_geom.intersection(polygon_geom)
                    
                    if not intersection.isEmpty() and intersection.type() == QgsWkbTypes.LineGeometry:
                        stream_geometries.append(intersection)
            
            # Check each point against all streams
            for point in all_points:
                point_geom = QgsGeometry.fromPointXY(point)
                keep_point = True
                
                for stream_geom in stream_geometries:
                    distance_to_stream = point_geom.distance(stream_geom)
                    
                    if distance_to_stream < min_distance:
                        keep_point = False
                        break
                
                if keep_point:
                    cleaned_points.append(point)
        
        except Exception as e:
            # If cleanup fails, return original points
            cleaned_points = all_points
        
        return cleaned_points

    def generate_continuous_stream_points(self, stream_lines, spacing, polygon_geom):
        """Generate continuous points around entire stream network with CORRECT spacing"""
        points = []
        
        try:
            # Process each stream line individually with correct spacing
            for line_idx, line in enumerate(stream_lines):
                if len(line) >= 2:
                    # Calculate total length of this line
                    total_length = 0
                    for i in range(len(line) - 1):
                        total_length += line[i].distance(line[i + 1])
                    
                    if total_length > 0:
                        # Calculate how many points we need along this line with CORRECT spacing
                        num_points_along_line = max(1, int(total_length / spacing))
                        
                        # Generate points along the line at correct intervals
                        for point_idx in range(num_points_along_line + 1):
                            # Calculate position along line
                            if num_points_along_line == 0:
                                ratio = 0.5
                            else:
                                ratio = point_idx / num_points_along_line
                            
                            target_distance = ratio * total_length
                            
                            # Find the exact position along the line
                            current_distance = 0
                            stream_point = None
                            
                            for i in range(len(line) - 1):
                                segment_length = line[i].distance(line[i + 1])
                                
                                if current_distance + segment_length >= target_distance:
                                    # Found the segment where our point should be
                                    if segment_length > 0:
                                        segment_ratio = (target_distance - current_distance) / segment_length
                                        stream_x = line[i].x() + segment_ratio * (line[i + 1].x() - line[i].x())
                                        stream_y = line[i].y() + segment_ratio * (line[i + 1].y() - line[i].y())
                                        
                                        # Calculate perpendicular direction for this segment
                                        dx = line[i + 1].x() - line[i].x()
                                        dy = line[i + 1].y() - line[i].y()
                                        length = math.sqrt(dx*dx + dy*dy)
                                        
                                        if length > 0:
                                            # Normalize and rotate 90 degrees for perpendicular
                                            norm_x = -dy / length
                                            norm_y = dx / length
                                            
                                            # Generate points on both sides at EXACT spacing distance
                                            for side in [-1, 1]:
                                                point_x = stream_x + norm_x * spacing * side
                                                point_y = stream_y + norm_y * spacing * side
                                                
                                                candidate_point = QgsPointXY(point_x, point_y)
                                                
                                                # Check if point is inside polygon
                                                if polygon_geom.contains(QgsGeometry.fromPointXY(candidate_point)):
                                                    # Check for duplicates with existing points
                                                    too_close = False
                                                    for existing_point in points:
                                                        if candidate_point.distance(existing_point) < spacing * 0.7:  # 70% tolerance
                                                            too_close = True
                                                            break
                                                    
                                                    if not too_close:
                                                        points.append(candidate_point)
                                    break
                                
                                current_distance += segment_length
        
        except Exception as e:
            pass
        
        return points

    def find_closest_point_to_contours(self, stream_points, contour_layer, polygon_geom):
        """Find the stream point closest to any contour line"""
        closest_point = None
        min_distance = float('inf')
        
        try:
            for stream_point in stream_points:
                for contour_feature in contour_layer.getFeatures():
                    contour_geom = contour_feature.geometry()
                    
                    if contour_geom.intersects(polygon_geom):
                        intersection = contour_geom.intersection(polygon_geom)
                        
                        if not intersection.isEmpty():
                            # Calculate distance to contour
                            distance = QgsGeometry.fromPointXY(stream_point).distance(intersection)
                            
                            if distance < min_distance:
                                min_distance = distance
                                closest_point = stream_point
        
        except Exception as e:
            pass
        
        return closest_point

    def generate_contour_continuation_points(self, contour_layer, start_point, spacing, polygon_geom, existing_points):
        """Generate points along contours starting from closest stream point"""
        points = []
        
        try:
            # Find the closest contour to start point
            closest_contour = None
            min_distance = float('inf')
            
            for contour_feature in contour_layer.getFeatures():
                contour_geom = contour_feature.geometry()
                
                if contour_geom.intersects(polygon_geom):
                    intersection = contour_geom.intersection(polygon_geom)
                    
                    if not intersection.isEmpty():
                        distance = QgsGeometry.fromPointXY(start_point).distance(intersection)
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_contour = intersection
            
            # Generate points along contours starting from the closest one
            if closest_contour:
                if closest_contour.isMultipart():
                    lines = closest_contour.asMultiPolyline()
                else:
                    lines = [closest_contour.asPolyline()]
                
                for line in lines:
                    if len(line) >= 2:
                        line_points = self.generate_points_along_contour_line(line, spacing, existing_points)
                        points.extend(line_points)
                
                # Continue with other contours
                for contour_feature in contour_layer.getFeatures():
                    contour_geom = contour_feature.geometry()
                    
                    if contour_geom.intersects(polygon_geom):
                        intersection = contour_geom.intersection(polygon_geom)
                        
                        if not intersection.isEmpty() and not intersection.equals(closest_contour):
                            if intersection.isMultipart():
                                lines = intersection.asMultiPolyline()
                            else:
                                lines = [intersection.asPolyline()]
                            
                            for line in lines:
                                if len(line) >= 2:
                                    line_points = self.generate_points_along_contour_line(line, spacing, existing_points + points)
                                    points.extend(line_points)
        
        except Exception as e:
            pass
        
        return points

    def generate_points_along_contour_line(self, line, spacing, existing_points):
        """Generate points along a single contour line avoiding existing points"""
        points = []
        
        try:
            if len(line) >= 2:
                # Calculate total length
                total_length = 0
                for i in range(len(line) - 1):
                    total_length += line[i].distance(line[i + 1])
                
                if total_length > 0:
                    num_points = max(1, int(total_length / spacing))
                    
                    for j in range(num_points):
                        if num_points == 1:
                            ratio = 0.5
                        else:
                            ratio = j / (num_points - 1) if num_points > 1 else 0
                        
                        # Interpolate point along line
                        current_dist = 0
                        target_dist = ratio * total_length
                        
                        for k in range(len(line) - 1):
                            segment_length = line[k].distance(line[k + 1])
                            if current_dist + segment_length >= target_dist:
                                if segment_length > 0:
                                    segment_ratio = (target_dist - current_dist) / segment_length
                                    x = line[k].x() + segment_ratio * (line[k + 1].x() - line[k].x())
                                    y = line[k].y() + segment_ratio * (line[k + 1].y() - line[k].y())
                                    
                                    candidate_point = QgsPointXY(x, y)
                                    
                                    # Check if point is far enough from existing points
                                    too_close = False
                                    for existing_point in existing_points:
                                        if candidate_point.distance(existing_point) < spacing * 0.7:  # 70% spacing tolerance
                                            too_close = True
                                            break
                                    
                                    if not too_close:
                                        points.append(candidate_point)
                                break
                            current_dist += segment_length
        
        except Exception as e:
            pass
        
        return points

    def generate_random_points(self, geom, spacing_x, spacing_y, source_crs, context, feedback):
        """Generate random points using native QGIS algorithm with min/max distance from spacing"""
        points = []
        
        try:
            # Use spacing_x as min distance and spacing_y as max distance
            min_distance = spacing_x
            max_distance = spacing_y
            
            # Calculate appropriate number of points based on area and spacing
            area = geom.area()
            avg_spacing = (min_distance + max_distance) / 2
            target_count = max(10, int((area / 10000) * (10000 / (avg_spacing ** 2))))
            
            # Create temporary layer for the polygon
            temp_layer = QgsVectorLayer(f"Polygon?crs={source_crs.authid()}", "temp", "memory")
            temp_provider = temp_layer.dataProvider()
            temp_feature = QgsFeature()
            temp_feature.setGeometry(geom)
            temp_provider.addFeatures([temp_feature])
            
            # Use native random points algorithm with minimum distance
            random_params = {
                'INPUT': temp_layer,
                'POINTS_NUMBER': target_count,
                'MIN_DISTANCE': min_distance,
                'SEED': None,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            }
            
            result = processing.run('native:randompointsinpolygons', random_params, 
                                   context=context, feedback=feedback, is_child_algorithm=True)
            
            # Extract points from result
            random_layer = QgsProcessingUtils.mapLayerFromString(result['OUTPUT'], context)
            for feature in random_layer.getFeatures():
                point_geom = feature.geometry()
                if point_geom and not point_geom.isEmpty():
                    point = point_geom.asPoint()
                    points.append(QgsPointXY(point.x(), point.y()))
            
            feedback.pushInfo(f"Generated {len(points)} random points (min: {min_distance}m, max: {max_distance}m)")
            
        except Exception as e:
            feedback.pushInfo(f"Random points generation error: {e}")
        
        return points

    def generate_triangular_points(self, geom, bbox, spacing):
        """Generate points in a triangular/quincunx pattern (staggered rows)"""
        points = []
        
        x_min = bbox.xMinimum() - spacing
        x_max = bbox.xMaximum() + spacing
        y_min = bbox.yMinimum() - spacing
        y_max = bbox.yMaximum() + spacing
        
        triangle_height = spacing * math.sqrt(3) / 2
        
        width = x_max - x_min
        height = y_max - y_min
        num_cols = math.ceil(width / spacing)
        num_rows = math.ceil(height / triangle_height)
        
        offset_x = (width - (num_cols - 1) * spacing) / 2
        offset_y = (height - (num_rows - 1) * triangle_height) / 2
        
        for row in range(num_rows):
            row_offset = spacing / 2 if row % 2 else 0
            y = y_min + offset_y + (row * triangle_height)
            
            for col in range(num_cols):
                x = x_min + offset_x + row_offset + (col * spacing)
                point = QgsPointXY(x, y)
                
                if geom.contains(point):
                    points.append(point)
        
        return points

    def generate_rectangular_points(self, geom, bbox, spacing_x, spacing_y):
        """Generate points in a rectangular/square pattern"""
        points = []
        
        x_min = bbox.xMinimum() - spacing_x
        x_max = bbox.xMaximum() + spacing_x
        y_min = bbox.yMinimum() - spacing_y
        y_max = bbox.yMaximum() + spacing_y
        
        width = x_max - x_min
        height = y_max - y_min
        num_cols = math.ceil(width / spacing_x)
        num_rows = math.ceil(height / spacing_y)
        
        offset_x = (width - (num_cols - 1) * spacing_x) / 2
        offset_y = (height - (num_rows - 1) * spacing_y) / 2
        
        for row in range(num_rows):
            y = y_min + offset_y + (row * spacing_y)
            
            for col in range(num_cols):
                x = x_min + offset_x + (col * spacing_x)
                point = QgsPointXY(x, y)
                
                if geom.contains(point):
                    points.append(point)
        
        return points

    def generate_cincooros_points(self, geom, bbox, spacing_x, spacing_y):
        """Generate points in a Five of Diamonds pattern (rectangular grid plus center points)"""
        points = []
        
        x_min = bbox.xMinimum() - spacing_x
        x_max = bbox.xMaximum() + spacing_x
        y_min = bbox.yMinimum() - spacing_y
        y_max = bbox.yMaximum() + spacing_y
        
        width = x_max - x_min
        height = y_max - y_min
        num_cols = math.ceil(width / spacing_x)
        num_rows = math.ceil(height / spacing_y)
        
        offset_x = (width - (num_cols - 1) * spacing_x) / 2
        offset_y = (height - (num_rows - 1) * spacing_y) / 2
        
        # Regular grid points
        grid_points = []
        for row in range(num_rows):
            y = y_min + offset_y + (row * spacing_y)
            
            for col in range(num_cols):
                x = x_min + offset_x + (col * spacing_x)
                point = QgsPointXY(x, y)
                
                if geom.contains(point):
                    grid_points.append((point, 'regular'))
        
        # Center points
        center_points = []
        for row in range(num_rows - 1):
            for col in range(num_cols - 1):
                center_x = x_min + offset_x + (col * spacing_x) + (spacing_x / 2)
                center_y = y_min + offset_y + (row * spacing_y) + (spacing_y / 2)
                
                center_point = QgsPointXY(center_x, center_y)
                
                if geom.contains(center_point):
                    center_points.append((center_point, 'center'))
        
        points = grid_points + center_points
        
        return points

    def generate_report(self, feedback, pattern_type, spacing_x, spacing_y, total_area_ha, 
                       triangular_count, rectangular_count, cincooros_count, keyline_count, random_count, slope_analysis):
        """Generate comprehensive report including slope analysis if enabled"""
        
        feedback.pushInfo("=" * 80)
        feedback.pushInfo("TREE PLANTING COMPLETE REPORT")
        feedback.pushInfo("=" * 80)
        
        feedback.pushInfo("📏 POLYGON INFORMATION:")
        feedback.pushInfo(f"   • Total area: {total_area_ha:.4f} ha")
        feedback.pushInfo("")
        
        feedback.pushInfo("📐 SPACING CONFIGURATION:")
        if spacing_x == spacing_y:
            feedback.pushInfo(f"   • Spacing: {spacing_x} x {spacing_y} m (Square)")
        else:
            feedback.pushInfo(f"   • Spacing: {spacing_x} x {spacing_y} m (Rectangular)")
        feedback.pushInfo("")
        
        # Slope analysis information if enabled
        if slope_analysis:
            feedback.pushInfo("🏔️ SLOPE ANALYSIS:")
            feedback.pushInfo("   • Slope analysis: AUTOMATIC (DEM detected)")
            feedback.pushInfo("   • 'slope' field added with values in degrees")
            feedback.pushInfo("   • Slope values sampled directly from DEM")
            feedback.pushInfo("")
        else:
            feedback.pushInfo("🏔️ SLOPE ANALYSIS:")
            feedback.pushInfo("   • Slope analysis: DISABLED (no DEM)")
            feedback.pushInfo("")
        
        # Theoretical calculations
        feedback.pushInfo("🧮 THEORETICAL CALCULATIONS (trees/ha):")
        theoretical_trees_per_ha = 10000 / (spacing_x * spacing_y)
        theoretical_triangular_trees_per_ha = 11547 / (spacing_x ** 2) if spacing_x == spacing_y else 11547 / (spacing_x * spacing_y)
        theoretical_cincooros_trees_per_ha = theoretical_trees_per_ha * 2
        
        feedback.pushInfo(f"   • Square/Rectangular pattern: {theoretical_trees_per_ha:.1f} trees/ha")
        feedback.pushInfo(f"   • Triangular/Quincunx pattern: {theoretical_triangular_trees_per_ha:.1f} trees/ha")
        feedback.pushInfo(f"   • Five of Diamonds pattern: {theoretical_cincooros_trees_per_ha:.1f} trees/ha")
        feedback.pushInfo("")
        
        # Actual results
        feedback.pushInfo("🌳 ACTUAL RESULTS (with border margin):")
        actual_triangular_per_ha = triangular_count / total_area_ha if total_area_ha > 0 else 0
        actual_rectangular_per_ha = rectangular_count / total_area_ha if total_area_ha > 0 else 0
        actual_cincooros_per_ha = cincooros_count / total_area_ha if total_area_ha > 0 else 0
        
        feedback.pushInfo(f"   • Square/Rectangular pattern: {actual_rectangular_per_ha:.1f} trees/ha")
        feedback.pushInfo(f"   • Triangular/Quincunx pattern: {actual_triangular_per_ha:.1f} trees/ha")
        feedback.pushInfo(f"   • Five of Diamonds pattern: {actual_cincooros_per_ha:.1f} trees/ha")
        if keyline_count > 0:
            actual_keyline_per_ha = keyline_count / total_area_ha if total_area_ha > 0 else 0
            feedback.pushInfo(f"   • Keyline pattern: {actual_keyline_per_ha:.1f} trees/ha")
        if random_count > 0:
            actual_random_per_ha = random_count / total_area_ha if total_area_ha > 0 else 0
            feedback.pushInfo(f"   • Random pattern: {actual_random_per_ha:.1f} trees/ha")
        feedback.pushInfo("")
        
        # Detailed count by pattern
        feedback.pushInfo("📊 DETAILED COUNT BY PATTERN:")
        
        if pattern_type == 0:
            feedback.pushInfo(f"   • Triangular/Quincunx: {triangular_count} trees [SELECTED]")
        else:
            feedback.pushInfo(f"   • Triangular/Quincunx: {triangular_count} trees")
        
        if pattern_type == 1:
            if spacing_x == spacing_y:
                feedback.pushInfo(f"   • Square: {rectangular_count} trees [SELECTED]")
            else:
                feedback.pushInfo(f"   • Rectangular: {rectangular_count} trees [SELECTED]")
        else:
            if spacing_x == spacing_y:
                feedback.pushInfo(f"   • Square: {rectangular_count} trees")
            else:
                feedback.pushInfo(f"   • Rectangular: {rectangular_count} trees")
        
        if pattern_type == 2:
            feedback.pushInfo(f"   • Five of Diamonds: {cincooros_count} trees [SELECTED]")
        else:
            feedback.pushInfo(f"   • Five of Diamonds: {cincooros_count} trees")
        
        if pattern_type == 3:
            feedback.pushInfo(f"   • Keyline: {keyline_count} trees [SELECTED]")
        elif keyline_count > 0:
            feedback.pushInfo(f"   • Keyline: {keyline_count} trees")
        
        if pattern_type == 4:
            feedback.pushInfo(f"   • Random: {random_count} trees [SELECTED]")
        elif random_count > 0:
            feedback.pushInfo(f"   • Random: {random_count} trees")
        
        feedback.pushInfo("")
        
        # Efficiency analysis
        feedback.pushInfo("📈 EFFICIENCY ANALYSIS (BORDER MARGIN IMPACT):")
        
        rect_efficiency = (actual_rectangular_per_ha / theoretical_trees_per_ha * 100) if theoretical_trees_per_ha > 0 else 0
        tri_efficiency = (actual_triangular_per_ha / theoretical_triangular_trees_per_ha * 100) if theoretical_triangular_trees_per_ha > 0 else 0
        cinco_efficiency = (actual_cincooros_per_ha / theoretical_cincooros_trees_per_ha * 100) if theoretical_cincooros_trees_per_ha > 0 else 0
        
        feedback.pushInfo(f"   • Square/Rectangular efficiency: {rect_efficiency:.1f}%")
        feedback.pushInfo(f"   • Triangular/Quincunx efficiency: {tri_efficiency:.1f}%")
        feedback.pushInfo(f"   • Five of Diamonds efficiency: {cinco_efficiency:.1f}%")
        feedback.pushInfo("")
        
        # Recommendations
        feedback.pushInfo("💡 RECOMMENDATIONS:")
        if tri_efficiency > rect_efficiency:
            feedback.pushInfo("   • Triangular/Quincunx pattern is more efficient for this spacing")
        elif rect_efficiency > tri_efficiency:
            feedback.pushInfo("   • Square/Rectangular pattern is more efficient for this spacing")
        else:
            feedback.pushInfo("   • Triangular and Square patterns have similar efficiency")
        
        if pattern_type == 3:
            feedback.pushInfo("   • Keyline pattern optimized for water management and erosion control")
        elif pattern_type == 4:
            feedback.pushInfo(f"   • Random pattern with minimum distance {spacing_x}m and maximum {spacing_y}m")
        
        if slope_analysis:
            feedback.pushInfo("   • Slope analysis completed successfully")
            feedback.pushInfo("   • Slope values available in 'slope' field (degrees)")
            feedback.pushInfo("   • Use these values to plan planting according to topography")
        else:
            feedback.pushInfo("   • For slope analysis, provide a DEM in next execution")
        
        feedback.pushInfo("=" * 80)

    def name(self):
        return 'treeplantingpattern'

    def displayName(self):
        return self.tr('Tree Planting Pattern Generator')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'arcgeekcalculator'

    def shortHelpString(self):
        return self.tr("Generates tree planting points with different patterns inside polygons. If DEM provided: automatic slope analysis and keyline pattern available. Output fields: id, pattern, x, y, slope (degrees).")

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return TreePlantingPatternAlgorithm()