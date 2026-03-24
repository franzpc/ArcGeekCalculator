from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer, 
                       QgsProcessingParameterFeatureSink, QgsProcessingParameterPoint, 
                       QgsWkbTypes, QgsField, QgsVectorLayer, QgsFeatureSink, 
                       QgsProcessing, QgsProcessingParameterVectorLayer,
                       QgsProcessingException, QgsMessageLog, Qgis,
                       QgsProcessingParameterNumber, QgsRasterLayer, QgsSnappingConfig,
                       QgsProcessingParameterBoolean, QgsFeature, QgsGeometry,
                       QgsPointXY, QgsSpatialIndex, QgsFields)
from qgis.PyQt.QtCore import QVariant, QCoreApplication
from qgis.utils import iface
import processing
from collections import deque
import math

class WatershedBasinDelineationAlgorithm(QgsProcessingAlgorithm):
    INPUT_DEM = 'INPUT_DEM'
    POUR_POINT = 'POUR_POINT'
    INPUT_STREAM = 'INPUT_STREAM'
    OUTPUT_BASIN = 'OUTPUT_BASIN'
    OUTPUT_STREAM = 'OUTPUT_STREAM'
    SMOOTH_ITERATIONS = 'SMOOTH_ITERATIONS'
    SMOOTH_OFFSET = 'SMOOTH_OFFSET'
    CALC_LONGEST_PATH = 'CALC_LONGEST_PATH'
    MAX_RASTER_SIZE = 100000000

    def __init__(self):
        super().__init__()
        self.activate_snapping()

    def activate_snapping(self):
        snapping_config = iface.mapCanvas().snappingUtils().config()
        snapping_config.setEnabled(True)
        try:
            snapping_config.setTypeFlag(QgsSnappingConfig.VertexFlag | QgsSnappingConfig.SegmentFlag)
        except TypeError:
            snapping_config.setType(QgsSnappingConfig.Vertex | QgsSnappingConfig.Segment)
        iface.mapCanvas().snappingUtils().setConfig(snapping_config)

    def deactivate_snapping(self):
        snapping_config = iface.mapCanvas().snappingUtils().config()
        snapping_config.setEnabled(False)
        try:
            snapping_config.setTypeFlag(QgsSnappingConfig.VertexFlag | QgsSnappingConfig.SegmentFlag)
        except TypeError:
            snapping_config.setType(QgsSnappingConfig.Vertex | QgsSnappingConfig.Segment)
        iface.mapCanvas().snappingUtils().setConfig(snapping_config)

    def get_points(self, geometry):
        if geometry.type() == QgsWkbTypes.LineGeometry:
            if geometry.isMultipart():
                multilines = geometry.asMultiPolyline()
                if multilines:
                    return multilines[0][0], multilines[-1][-1]
            else:
                polyline = geometry.asPolyline()
                if polyline:
                    return polyline[0], polyline[-1]
        return None, None

    def find_upstream_features(self, feature, layer, tolerance=0.0001):
        if not feature.geometry() or feature.geometry().isEmpty():
            return []
        start_point, _ = self.get_points(feature.geometry())
        if start_point is None:
            return []
        upstream_features = []
        for f in layer.getFeatures():
            if f.id() == feature.id():
                continue
            if not f.geometry() or f.geometry().isEmpty():
                continue
            _, end_point = self.get_points(f.geometry())
            if end_point is None:
                continue
            if abs(start_point.x() - end_point.x()) < tolerance and abs(start_point.y() - end_point.y()) < tolerance:
                upstream_features.append(f)
        return upstream_features

    def find_downstream_feature(self, feature, layer, tolerance=0.0001):
        if not feature.geometry() or feature.geometry().isEmpty():
            return None
        _, end_point = self.get_points(feature.geometry())
        if end_point is None:
            return None
        for f in layer.getFeatures():
            if f.id() == feature.id():
                continue
            if not f.geometry() or f.geometry().isEmpty():
                continue
            start_point, _ = self.get_points(f.geometry())
            if start_point is None:
                continue
            if abs(end_point.x() - start_point.x()) < tolerance and abs(end_point.y() - start_point.y()) < tolerance:
                return f
        return None

    def calculate_strahler(self, stream_layer, feedback):
        feedback.pushInfo("Starting Strahler order calculation...")
        if not stream_layer.startEditing():
            feedback.reportError("Could not start editing stream layer")
            return
        field_names = [field.name() for field in stream_layer.fields()]
        if 'sthr_original' not in field_names:
            if not stream_layer.dataProvider().addAttributes([QgsField('sthr_original', QVariant.Int)]):
                feedback.reportError("Could not add sthr_original field")
                return
            stream_layer.updateFields()
        for feature in stream_layer.getFeatures():
            feature['sthr_original'] = 1
            if not stream_layer.updateFeature(feature):
                feedback.reportError(f"Could not update feature {feature.id()}")
        processed_features = {}
        def get_strahler_order(feature_id):
            if feature_id in processed_features:
                return processed_features[feature_id]
            feature = stream_layer.getFeature(feature_id)
            if not feature.isValid():
                return 1
            upstream_features = self.find_upstream_features(feature, stream_layer)
            if not upstream_features:
                order = 1
            else:
                upstream_orders = [get_strahler_order(f.id()) for f in upstream_features if f.isValid()]
                if upstream_orders:
                    max_order = max(upstream_orders)
                    count_max = upstream_orders.count(max_order)
                    order = max_order + 1 if count_max > 1 else max_order
                else:
                    order = 1
            processed_features[feature_id] = order
            feature['sthr_original'] = order
            stream_layer.updateFeature(feature)
            return order
        outlet_features = []
        for feature in stream_layer.getFeatures():
            if not self.find_downstream_feature(feature, stream_layer):
                outlet_features.append(feature)
        if outlet_features:
            for outlet in outlet_features:
                get_strahler_order(outlet.id())
        if not stream_layer.commitChanges():
            feedback.reportError("Could not commit changes to stream layer")
        else:
            feedback.pushInfo("Strahler order calculation completed")

    def extend_main_channel(self, stream_layer, feedback):
        feedback.pushInfo("Starting main channel extension...")
        if not stream_layer.startEditing():
            feedback.reportError("Could not start editing stream layer for extension")
            return
        field_names = [field.name() for field in stream_layer.fields()]
        if 'sthr_extend' not in field_names:
            if not stream_layer.dataProvider().addAttributes([QgsField('sthr_extend', QVariant.Int)]):
                feedback.reportError("Could not add sthr_extend field")
                return
            stream_layer.updateFields()
        for feature in stream_layer.getFeatures():
            original_value = feature['sthr_original'] if feature['sthr_original'] is not None else 1
            feature['sthr_extend'] = original_value
            stream_layer.updateFeature(feature)
        outlets = []
        for feature in stream_layer.getFeatures():
            if not self.find_downstream_feature(feature, stream_layer):
                outlets.append(feature)
        if not outlets:
            if not stream_layer.commitChanges():
                feedback.reportError("Could not commit changes")
            return
        main_outlet = max(outlets, key=lambda feat: feat['sthr_extend'] if feat['sthr_extend'] is not None else 0)
        max_order = main_outlet['sthr_extend']
        for current_order in range(max_order, 1, -1):
            last_segments = []
            for feature in stream_layer.getFeatures():
                if feature['sthr_extend'] == current_order:
                    upstream_features = self.find_upstream_features(feature, stream_layer)
                    same_or_higher_order = [f for f in upstream_features if f['sthr_extend'] >= current_order]
                    if not same_or_higher_order:
                        last_segments.append(feature)
            for last_segment in last_segments:
                upstream_features = self.find_upstream_features(last_segment, stream_layer)
                next_order_features = [f for f in upstream_features if f['sthr_extend'] == current_order - 1]
                if next_order_features:
                    branches = []
                    for next_feature in next_order_features:
                        branch = self.find_complete_branch(next_feature, current_order - 1, stream_layer)
                        if branch:
                            total_length = sum(f.geometry().length() for f in branch if f.geometry() is not None)
                            branches.append((branch, total_length))
                    if branches:
                        longest_branch = max(branches, key=lambda x: x[1])[0]
                        for feature in longest_branch:
                            feature['sthr_extend'] = current_order
                            stream_layer.updateFeature(feature)
        if not stream_layer.commitChanges():
            feedback.reportError("Could not commit changes for extend")
        else:
            feedback.pushInfo("Main channel extension completed")

    def find_complete_branch(self, start_feature, target_order, stream_layer, visited=None):
        if visited is None:
            visited = set()
        if start_feature.id() in visited:
            return []
        visited.add(start_feature.id())
        if start_feature['sthr_extend'] != target_order:
            return []
        result = [start_feature]
        upstream_features = self.find_upstream_features(start_feature, stream_layer)
        target_order_upstream = [f for f in upstream_features if f['sthr_extend'] == target_order]
        for upstream in target_order_upstream:
            branch = self.find_complete_branch(upstream, target_order, stream_layer, visited)
            result.extend(branch)
        return result

    def extend_to_headwaters(self, stream_layer, feedback):
        feedback.pushInfo("Starting final channel extension to headwaters...")
        if not stream_layer.startEditing():
            feedback.reportError("Could not start editing stream layer for final extension")
            return
        field_names = [field.name() for field in stream_layer.fields()]
        if 'sthr_final' not in field_names:
            if not stream_layer.dataProvider().addAttributes([QgsField('sthr_final', QVariant.Int)]):
                feedback.reportError("Could not add sthr_final field")
                return
            stream_layer.updateFields()
        for feature in stream_layer.getFeatures():
            feature['sthr_final'] = 1
            stream_layer.updateFeature(feature)
        outlets = []
        for feature in stream_layer.getFeatures():
            if not self.find_downstream_feature(feature, stream_layer):
                outlets.append(feature)
        if not outlets:
            if not stream_layer.commitChanges():
                feedback.reportError("Could not commit changes")
            return
        main_outlet = max(outlets, key=lambda feat: feat['sthr_extend'] if feat['sthr_extend'] is not None else 0)
        max_order = main_outlet['sthr_extend'] if main_outlet['sthr_extend'] is not None else 1
        try:
            main_channel_path = self.find_main_channel_to_headwater(main_outlet, stream_layer)
            feedback.pushInfo(f"Found main channel path with {len(main_channel_path)} segments")
            for feature in main_channel_path:
                feature['sthr_final'] = max_order
                if not stream_layer.updateFeature(feature):
                    feedback.reportError(f"Could not update feature {feature.id()}")
        except Exception as e:
            feedback.reportError(f"Error tracing main channel: {str(e)}")
        if not stream_layer.commitChanges():
            feedback.reportError("Could not commit final changes")
        else:
            feedback.pushInfo("Final channel extension completed")

    def find_main_channel_to_headwater(self, start_feature, stream_layer, visited=None):
        if visited is None:
            visited = set()
        if not start_feature or not start_feature.isValid() or start_feature.id() in visited:
            return []
        visited.add(start_feature.id())
        result = [start_feature]
        try:
            upstream_features = self.find_upstream_features(start_feature, stream_layer)
            upstream_features = [f for f in upstream_features if f.isValid()]
            if not upstream_features:
                return result
            if len(upstream_features) == 1:
                upstream_path = self.find_main_channel_to_headwater(upstream_features[0], stream_layer, visited)
                result.extend(upstream_path)
            else:
                branches = []
                for upstream_feature in upstream_features:
                    if upstream_feature.id() not in visited:
                        branch_path = self.trace_complete_path(upstream_feature, stream_layer, set())
                        if branch_path:
                            total_length = sum(f.geometry().length() for f in branch_path if f.geometry() is not None)
                            branches.append((total_length, upstream_feature))
                if branches:
                    longest_branch = max(branches, key=lambda x: x[0])
                    upstream_path = self.find_main_channel_to_headwater(longest_branch[1], stream_layer, visited)
                    result.extend(upstream_path)
        except Exception as e:
            QgsMessageLog.logMessage(f"Error in find_main_channel_to_headwater: {str(e)}", level=Qgis.Critical)
        return result

    def trace_complete_path(self, start_feature, stream_layer, visited=None):
        if visited is None:
            visited = set()
        if not start_feature or not start_feature.isValid() or start_feature.id() in visited:
            return []
        visited.add(start_feature.id())
        result = [start_feature]
        try:
            upstream_features = self.find_upstream_features(start_feature, stream_layer)
            upstream_features = [f for f in upstream_features if f.isValid() and f.id() not in visited]
            for upstream in upstream_features:
                branch = self.trace_complete_path(upstream, stream_layer, visited)
                result.extend(branch)
        except Exception as e:
            QgsMessageLog.logMessage(f"Error in trace_complete_path: {str(e)}", level=Qgis.Critical)
        return result

    def get_elevation_at_point(self, dem_layer, point):
        """Get elevation value at a specific point using sample()"""
        try:
            if isinstance(point, QgsPointXY):
                pt = point
            else:
                pt = QgsPointXY(point.x(), point.y())
            val, success = dem_layer.dataProvider().sample(pt, 1)
            if success and val is not None and not math.isnan(val):
                return val
        except Exception as e:
            pass
        return None

    def get_outlet_point(self, stream_layer, feedback):
        """Get the outlet point of the stream network (end of main channel)"""
        for feature in stream_layer.getFeatures():
            if not self.find_downstream_feature(feature, stream_layer):
                _, end_point = self.get_points(feature.geometry())
                if end_point is not None:
                    feedback.pushInfo(f"Outlet point: {end_point.x():.2f}, {end_point.y():.2f}")
                    return QgsPointXY(end_point.x(), end_point.y())
        return None

    def get_headwater_point_of_main_channel(self, stream_layer, feedback):
        """Get the headwater point (start point) of the main channel (sthr_final = max_order)"""
        max_order = 0
        for feature in stream_layer.getFeatures():
            if feature['sthr_final'] is not None and feature['sthr_final'] > max_order:
                max_order = feature['sthr_final']
        
        if max_order == 0:
            return None, None
        
        main_channel_features = [f for f in stream_layer.getFeatures() if f['sthr_final'] == max_order]
        
        for feature in main_channel_features:
            upstream_features = self.find_upstream_features(feature, stream_layer)
            upstream_main = [f for f in upstream_features if f['sthr_final'] == max_order]
            if not upstream_main:
                start_point, _ = self.get_points(feature.geometry())
                if start_point is not None:
                    feedback.pushInfo(f"Headwater point of main channel: {start_point.x():.2f}, {start_point.y():.2f}")
                    return QgsPointXY(start_point.x(), start_point.y()), max_order
        return None, max_order

    def find_farthest_point_from_outlet(self, basin_layer, outlet_point, feedback):
        """Find the farthest point on basin boundary from outlet"""
        feedback.pushInfo("Finding farthest point from outlet...")
        
        basin_feature = next(basin_layer.getFeatures())
        basin_geom = basin_feature.geometry()
        
        furthest_point = None
        max_distance = 0
        
        for vertex in basin_geom.vertices():
            vertex_point = QgsPointXY(vertex.x(), vertex.y())
            distance = QgsGeometry.fromPointXY(vertex_point).distance(QgsGeometry.fromPointXY(outlet_point))
            if distance > max_distance:
                max_distance = distance
                furthest_point = vertex_point
        
        if furthest_point is not None:
            feedback.pushInfo(f"Farthest point: {furthest_point.x():.2f}, {furthest_point.y():.2f}")
            feedback.pushInfo(f"Distance from outlet: {max_distance:.2f} m")
        
        return furthest_point

    def calculate_longest_flow_path(self, stream_layer, basin_layer, dem_layer, context, feedback):
        """Calculate the longest flow path using r.drain from farthest point"""
        feedback.pushInfo("Calculating Longest Flow Path...")
        
        outlet_point = self.get_outlet_point(stream_layer, feedback)
        if outlet_point is None:
            feedback.reportError("Could not find outlet point")
            return None
        
        farthest_point = self.find_farthest_point_from_outlet(basin_layer, outlet_point, feedback)
        
        if farthest_point is None:
            feedback.reportError("Could not find farthest point on basin boundary")
            return None
        
        try:
            drain_coords = f"{farthest_point.x()},{farthest_point.y()}"
            feedback.pushInfo(f"Running r.drain from: {drain_coords}")
            
            drain_result = processing.run('grass7:r.drain', {
                'input': dem_layer,
                'start_coordinates': drain_coords,
                'output': 'TEMPORARY_OUTPUT',
                'drain': 'TEMPORARY_OUTPUT',
                '-c': False,
                '-a': False,
                '-n': False,
                '-d': False
            }, context=context, feedback=feedback)
            
            drain_output = drain_result['drain']
            feedback.pushInfo(f"r.drain vector output: {drain_output}")
            
            drain_layer = QgsVectorLayer(drain_output, 'drain_path', 'ogr')
            
            if not drain_layer.isValid():
                feedback.reportError("r.drain output is not valid")
                return None
            
            feedback.pushInfo(f"r.drain produced {drain_layer.featureCount()} feature(s)")
            
            return drain_layer
            
        except Exception as e:
            feedback.reportError(f"Error running r.drain: {str(e)}")
            import traceback
            feedback.reportError(traceback.format_exc())
            return None

    def find_intersection_with_stream(self, drain_layer, stream_layer, farthest_point, feedback):
        """Find the FIRST intersection point between drain and stream network"""
        feedback.pushInfo("Finding first intersection between drain and stream network...")
        
        try:
            drain_feature = next(drain_layer.getFeatures())
            drain_geom = drain_feature.geometry()
            
            # Get drain points in order
            if drain_geom.isMultipart():
                lines = drain_geom.asMultiPolyline()
                all_drain_points = []
                for line in lines:
                    all_drain_points.extend(line)
            else:
                all_drain_points = drain_geom.asPolyline()
            
            if not all_drain_points:
                feedback.reportError("No points in drain geometry")
                return None, None
            
            feedback.pushInfo(f"Drain has {len(all_drain_points)} vertices")
            
            # Determine which end is the boundary (farthest_point)
            first_pt = all_drain_points[0]
            last_pt = all_drain_points[-1]
            
            dist_first_to_boundary = math.sqrt((first_pt.x() - farthest_point.x())**2 + 
                                                (first_pt.y() - farthest_point.y())**2)
            dist_last_to_boundary = math.sqrt((last_pt.x() - farthest_point.x())**2 + 
                                               (last_pt.y() - farthest_point.y())**2)
            
            feedback.pushInfo(f"Distance first point to boundary: {dist_first_to_boundary:.2f} m")
            feedback.pushInfo(f"Distance last point to boundary: {dist_last_to_boundary:.2f} m")
            
            # Order points from boundary to outlet
            if dist_first_to_boundary < dist_last_to_boundary:
                ordered_drain_points = all_drain_points
                feedback.pushInfo("Drain flows from first to last point")
            else:
                ordered_drain_points = list(reversed(all_drain_points))
                feedback.pushInfo("Drain flows from last to first point (reversed)")
            
            # Combine all stream geometries
            stream_geoms = []
            for feat in stream_layer.getFeatures():
                if feat.geometry() and not feat.geometry().isEmpty():
                    stream_geoms.append(feat.geometry())
            
            if not stream_geoms:
                feedback.reportError("No valid stream geometries found")
                return None, None
            
            combined_stream = QgsGeometry.unaryUnion(stream_geoms)
            
            # Find first intersection point along drain (from boundary)
            first_intersection_point = None
            first_intersection_idx = None
            intersecting_feature = None
            
            tolerance = 5.0  # 5 meters tolerance
            
            for i, drain_pt in enumerate(ordered_drain_points):
                drain_pt_xy = QgsPointXY(drain_pt.x(), drain_pt.y())
                drain_pt_geom = QgsGeometry.fromPointXY(drain_pt_xy)
                
                # Check distance to combined stream
                distance = drain_pt_geom.distance(combined_stream)
                
                if distance < tolerance:
                    # Found intersection - snap to exact point on stream
                    nearest_on_stream = combined_stream.nearestPoint(drain_pt_geom)
                    first_intersection_point = QgsPointXY(nearest_on_stream.asPoint().x(), 
                                                         nearest_on_stream.asPoint().y())
                    first_intersection_idx = i
                    
                    # Find which specific feature intersects
                    for stream_feat in stream_layer.getFeatures():
                        if not stream_feat.geometry() or stream_feat.geometry().isEmpty():
                            continue
                        
                        if drain_pt_geom.distance(stream_feat.geometry()) < tolerance:
                            intersecting_feature = stream_feat
                            break
                    
                    feedback.pushInfo(f"Found first intersection at drain vertex {i}")
                    feedback.pushInfo(f"Intersection point: {first_intersection_point.x():.2f}, {first_intersection_point.y():.2f}")
                    break
            
            if first_intersection_point is None:
                feedback.reportError(f"No intersection found within {tolerance}m tolerance")
                return None, None
            
            return first_intersection_point, intersecting_feature
            
        except Exception as e:
            feedback.reportError(f"Error finding intersection: {str(e)}")
            import traceback
            feedback.reportError(traceback.format_exc())
            return None, None

    def split_drain_at_intersection(self, drain_layer, intersection_point, farthest_point, feedback):
        """Split the drain line at first intersection and return the upstream segment (boundary to intersection)"""
        feedback.pushInfo("Splitting drain at first stream intersection...")
        
        try:
            drain_feature = next(drain_layer.getFeatures())
            drain_geom = drain_feature.geometry()
            
            if drain_geom.isMultipart():
                lines = drain_geom.asMultiPolyline()
                all_points = []
                for line in lines:
                    all_points.extend(line)
            else:
                all_points = drain_geom.asPolyline()
            
            if not all_points:
                feedback.reportError("No points in drain geometry")
                return None
            
            # Find closest vertex to intersection point
            min_dist = float('inf')
            split_idx = 0
            
            for i, pt in enumerate(all_points):
                dist = math.sqrt((pt.x() - intersection_point.x())**2 + (pt.y() - intersection_point.y())**2)
                if dist < min_dist:
                    min_dist = dist
                    split_idx = i
            
            feedback.pushInfo(f"Splitting at vertex {split_idx}")
            
            # Determine which end is near boundary
            first_pt = all_points[0]
            last_pt = all_points[-1]
            
            dist_first_to_boundary = math.sqrt((first_pt.x() - farthest_point.x())**2 + 
                                                (first_pt.y() - farthest_point.y())**2)
            dist_last_to_boundary = math.sqrt((last_pt.x() - farthest_point.x())**2 + 
                                               (last_pt.y() - farthest_point.y())**2)
            
            # Create segment from boundary to intersection
            if dist_first_to_boundary < dist_last_to_boundary:
                # Boundary is at first, take from 0 to split_idx
                segment_points_xy = [QgsPointXY(p.x(), p.y()) for p in all_points[:split_idx]]
                segment_points_xy.append(intersection_point)
            else:
                # Boundary is at last, take from split_idx to end
                segment_points_xy = [intersection_point]
                segment_points_xy.extend([QgsPointXY(p.x(), p.y()) for p in all_points[split_idx + 1:]])
            
            if len(segment_points_xy) < 2:
                feedback.reportError("Segment has less than 2 points")
                return None
            
            segment_geom = QgsGeometry.fromPolylineXY(segment_points_xy)
            
            feedback.pushInfo(f"Created drain segment with {len(segment_points_xy)} vertices, length: {segment_geom.length():.2f} m")
            
            return segment_geom
            
        except Exception as e:
            feedback.reportError(f"Error splitting drain: {str(e)}")
            import traceback
            feedback.reportError(traceback.format_exc())
            return None

    def add_longest_path_to_stream(self, stream_layer, drain_segment, intersecting_feature, intersection_point, outlet_point, max_order, feedback):
        """Topologically merge drain segment with stream network at intersection"""
        feedback.pushInfo("Topologically merging longest flow path with stream network...")
        
        if drain_segment is None or drain_segment.isEmpty():
            feedback.reportError("No valid drain segment geometry")
            return False
        
        if intersecting_feature is None:
            feedback.reportError("No intersecting stream feature")
            return False
        
        if not stream_layer.startEditing():
            feedback.reportError("Could not start editing stream layer")
            return False
        
        try:
            # Get stream geometry
            stream_geom = QgsGeometry(intersecting_feature.geometry())
            
            if stream_geom.isMultipart():
                lines = stream_geom.asMultiPolyline()
                stream_points = []
                for line in lines:
                    stream_points.extend(line)
            else:
                stream_points = stream_geom.asPolyline()
            
            # Find split index in stream
            min_dist = float('inf')
            split_idx = 0
            for i, pt in enumerate(stream_points):
                dist = math.sqrt((pt.x() - intersection_point.x())**2 + (pt.y() - intersection_point.y())**2)
                if dist < min_dist:
                    min_dist = dist
                    split_idx = i
            
            feedback.pushInfo(f"Splitting stream at index {split_idx}")
            
            # Create two stream segments
            if split_idx == 0:
                stream_seg1 = None
                stream_seg2_points = [intersection_point] + [QgsPointXY(p.x(), p.y()) for p in stream_points[1:]]
                stream_seg2 = QgsGeometry.fromPolylineXY(stream_seg2_points)
            elif split_idx == len(stream_points) - 1:
                stream_seg1_points = [QgsPointXY(p.x(), p.y()) for p in stream_points[:-1]] + [intersection_point]
                stream_seg1 = QgsGeometry.fromPolylineXY(stream_seg1_points)
                stream_seg2 = None
            else:
                stream_seg1_points = [QgsPointXY(p.x(), p.y()) for p in stream_points[:split_idx]] + [intersection_point]
                stream_seg1 = QgsGeometry.fromPolylineXY(stream_seg1_points)
                stream_seg2_points = [intersection_point] + [QgsPointXY(p.x(), p.y()) for p in stream_points[split_idx + 1:]]
                stream_seg2 = QgsGeometry.fromPolylineXY(stream_seg2_points)
            
            # Determine which stream segment to keep: the one whose free end is closer to outlet
            stream_seg_to_keep = None
            
            if stream_seg1 and stream_seg2:
                # Get the free ends (not at intersection)
                seg1_free_end = stream_seg1_points[0]  # Start of seg1
                seg2_free_end = stream_seg2_points[-1]  # End of seg2
                
                # Calculate distances to outlet
                dist1_to_outlet = math.sqrt((seg1_free_end.x() - outlet_point.x())**2 + 
                                           (seg1_free_end.y() - outlet_point.y())**2)
                dist2_to_outlet = math.sqrt((seg2_free_end.x() - outlet_point.x())**2 + 
                                           (seg2_free_end.y() - outlet_point.y())**2)
                
                feedback.pushInfo(f"Segment 1 free end distance to outlet: {dist1_to_outlet:.2f} m")
                feedback.pushInfo(f"Segment 2 free end distance to outlet: {dist2_to_outlet:.2f} m")
                
                # Keep the segment whose free end is closer to outlet (flows toward outlet)
                if dist1_to_outlet < dist2_to_outlet:
                    stream_seg_to_keep = stream_seg1
                    feedback.pushInfo("Keeping segment 1 (closer to outlet)")
                else:
                    stream_seg_to_keep = stream_seg2
                    feedback.pushInfo("Keeping segment 2 (closer to outlet)")
            elif stream_seg1:
                stream_seg_to_keep = stream_seg1
            elif stream_seg2:
                stream_seg_to_keep = stream_seg2
            
            if not stream_seg_to_keep:
                feedback.reportError("No valid stream segment to keep")
                stream_layer.rollBack()
                return False
            
            # Get points from drain and stream segments
            drain_points = drain_segment.asPolyline() if not drain_segment.isMultipart() else []
            if drain_segment.isMultipart():
                for line in drain_segment.asMultiPolyline():
                    drain_points.extend(line)
            
            stream_keep_points = stream_seg_to_keep.asPolyline() if not stream_seg_to_keep.isMultipart() else []
            if stream_seg_to_keep.isMultipart():
                for line in stream_seg_to_keep.asMultiPolyline():
                    stream_keep_points.extend(line)
            
            drain_points_xy = [QgsPointXY(p.x(), p.y()) for p in drain_points]
            stream_points_xy = [QgsPointXY(p.x(), p.y()) for p in stream_keep_points]
            
            # Check which ends are at intersection
            tolerance = 5.0
            drain_start_at_int = math.sqrt((drain_points_xy[0].x() - intersection_point.x())**2 + 
                                           (drain_points_xy[0].y() - intersection_point.y())**2) < tolerance
            drain_end_at_int = math.sqrt((drain_points_xy[-1].x() - intersection_point.x())**2 + 
                                         (drain_points_xy[-1].y() - intersection_point.y())**2) < tolerance
            
            stream_start_at_int = math.sqrt((stream_points_xy[0].x() - intersection_point.x())**2 + 
                                            (stream_points_xy[0].y() - intersection_point.y())**2) < tolerance
            stream_end_at_int = math.sqrt((stream_points_xy[-1].x() - intersection_point.x())**2 + 
                                          (stream_points_xy[-1].y() - intersection_point.y())**2) < tolerance
            
            # Merge geometries
            merged_points = []
            
            if drain_end_at_int and stream_start_at_int:
                merged_points = drain_points_xy[:-1] + [intersection_point] + stream_points_xy[1:]
            elif drain_end_at_int and stream_end_at_int:
                merged_points = drain_points_xy[:-1] + [intersection_point] + list(reversed(stream_points_xy[:-1]))
            elif drain_start_at_int and stream_start_at_int:
                merged_points = stream_points_xy[:-1] + [intersection_point] + list(reversed(drain_points_xy[1:]))
            elif drain_start_at_int and stream_end_at_int:
                merged_points = stream_points_xy[:-1] + [intersection_point] + drain_points_xy[1:]
            else:
                feedback.pushInfo("Using default merge order")
                merged_points = drain_points_xy + stream_points_xy
            
            if len(merged_points) < 2:
                feedback.reportError("Merged geometry has less than 2 points")
                stream_layer.rollBack()
                return False
            
            merged_geom = QgsGeometry.fromPolylineXY(merged_points)
            
            feedback.pushInfo(f"Created merged geometry: {len(merged_points)} vertices, length: {merged_geom.length():.2f} m")
            
            # Delete original stream feature
            if not stream_layer.deleteFeature(intersecting_feature.id()):
                feedback.reportError(f"Could not delete stream feature {intersecting_feature.id()}")
                stream_layer.rollBack()
                return False
            
            # Add merged feature
            new_feature = QgsFeature(stream_layer.fields())
            new_feature.setGeometry(merged_geom)
            
            # Copy attributes
            for field in stream_layer.fields():
                field_name = field.name()
                if field_name in ['sthr_original', 'sthr_extend', 'sthr_final']:
                    new_feature[field_name] = max_order
                else:
                    new_feature[field_name] = intersecting_feature[field_name]
            
            if not stream_layer.addFeature(new_feature):
                feedback.reportError("Could not add merged feature")
                stream_layer.rollBack()
                return False
            
            if not stream_layer.commitChanges():
                feedback.reportError("Could not commit changes")
                return False
            
            feedback.pushInfo("Topological merge completed successfully")
            return True
            
        except Exception as e:
            feedback.reportError(f"Error during merge: {str(e)}")
            import traceback
            feedback.reportError(traceback.format_exc())
            stream_layer.rollBack()
            return False

    def canCancel(self):
        return True
        
    def onClose(self):
        self.deactivate_snapping()
        super().onClose()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_DEM, 'Input DEM'))
        self.addParameter(QgsProcessingParameterPoint(self.POUR_POINT, 'Pour Point (click on the river)'))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_STREAM, 'Input Stream Network', 
                                                           types=[QgsProcessing.TypeVectorLine], optional=False))
        self.addParameter(QgsProcessingParameterNumber(self.SMOOTH_ITERATIONS, 'Smoothing Iterations', 
                                                       type=QgsProcessingParameterNumber.Integer, 
                                                       minValue=0, maxValue=10, defaultValue=1))
        self.addParameter(QgsProcessingParameterNumber(self.SMOOTH_OFFSET, 'Smoothing Offset', 
                                                       type=QgsProcessingParameterNumber.Double, 
                                                       minValue=0.0, maxValue=0.5, defaultValue=0.25))
        self.addParameter(QgsProcessingParameterBoolean(self.CALC_LONGEST_PATH, 
                                                        'Calculate Longest Flow Path', 
                                                        defaultValue=False))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT_BASIN, 'Watershed Basin', 
                                                            QgsProcessing.TypeVectorPolygon))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT_STREAM, 'Basin Stream Network', 
                                                            QgsProcessing.TypeVectorLine))

    def processAlgorithm(self, parameters, context, feedback):
        dem = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
        pour_point = self.parameterAsPoint(parameters, self.POUR_POINT, context)
        input_stream = self.parameterAsVectorLayer(parameters, self.INPUT_STREAM, context)
        smooth_iterations = self.parameterAsInt(parameters, self.SMOOTH_ITERATIONS, context)
        smooth_offset = self.parameterAsDouble(parameters, self.SMOOTH_OFFSET, context)
        calc_longest_path = self.parameterAsBoolean(parameters, self.CALC_LONGEST_PATH, context)

        if not dem.isValid():
            raise QgsProcessingException(self.tr('Invalid input DEM'))

        if not input_stream or input_stream.geometryType() != QgsWkbTypes.LineGeometry:
            raise QgsProcessingException(self.tr('Input Stream Network must be a line layer'))

        original_cell_size = dem.rasterUnitsPerPixelX()
        cell_size_multiplier = 1
        max_attempts = 3
        
        for attempt in range(max_attempts):
            current_cell_size = original_cell_size * cell_size_multiplier
            resampled_dem = self.resample_dem(dem, current_cell_size, context, feedback)
            raster_size = resampled_dem.width() * resampled_dem.height()
            if raster_size <= self.MAX_RASTER_SIZE:
                if attempt > 0:
                    feedback.pushInfo(self.tr(f'DEM resampled to {current_cell_size:.2f} units per pixel for processing.'))
                break
            cell_size_multiplier *= 3
        
        if raster_size > self.MAX_RASTER_SIZE:
            raise QgsProcessingException(self.tr('Input DEM is too large to process efficiently even after resampling.'))

        filled_dem = processing.run('grass7:r.fill.dir', {
            'input': resampled_dem,
            'format': 0,
            'output': 'TEMPORARY_OUTPUT',
            'direction': 'TEMPORARY_OUTPUT',
            'areas': 'TEMPORARY_OUTPUT'
        }, context=context, feedback=feedback)['output']

        watershed_result = processing.run('grass7:r.watershed', {
            'elevation': filled_dem,
            'convergence': 5,
            'memory': 300,
            '-s': True,
            'accumulation': 'TEMPORARY_OUTPUT',
            'drainage': 'TEMPORARY_OUTPUT'
        }, context=context, feedback=feedback)
        
        drainage = watershed_result['drainage']

        pour_point_str = f'{pour_point.x()},{pour_point.y()}'
        basin_raster = processing.run('grass7:r.water.outlet', {
            'input': drainage,
            'coordinates': pour_point_str,
            'output': 'TEMPORARY_OUTPUT'
        }, context=context, feedback=feedback)['output']

        basin_vector_result = processing.run('grass7:r.to.vect', {
            'input': basin_raster,
            'type': 2,
            'column': 'value',
            '-s': True,
            'output': 'TEMPORARY_OUTPUT',
            'GRASS_OUTPUT_TYPE_PARAMETER': 3
        }, context=context, feedback=feedback)
        
        basin_vector = basin_vector_result['output']
        basin_layer = QgsVectorLayer(basin_vector, 'basin', 'ogr')

        if basin_layer.featureCount() > 1:
            max_area = 0
            largest_feature = None
            for feature in basin_layer.getFeatures():
                area = feature.geometry().area()
                if area > max_area:
                    max_area = area
                    largest_feature = feature
            if largest_feature:
                largest_polygon = QgsVectorLayer("Polygon?crs=" + basin_layer.crs().authid(), "largest_polygon", "memory")
                provider = largest_polygon.dataProvider()
                provider.addFeatures([largest_feature])
                largest_polygon.updateExtents()
            else:
                largest_polygon = basin_layer
        else:
            largest_polygon = basin_layer

        smoothed_basin = processing.run('native:smoothgeometry', {
            'INPUT': largest_polygon,
            'ITERATIONS': smooth_iterations,
            'OFFSET': smooth_offset,
            'MAX_ANGLE': 180,
            'OUTPUT': 'memory:'
        }, context=context, feedback=feedback)['OUTPUT']

        basin_layer = smoothed_basin

        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT_BASIN, context,
                                               basin_layer.fields(), QgsWkbTypes.Polygon, basin_layer.crs())
        
        if sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT_BASIN))

        features = basin_layer.getFeatures()
        for feature in features:
            sink.addFeature(feature, QgsFeatureSink.FastInsert)

        results = {self.OUTPUT_BASIN: dest_id}

        try:
            clipped_stream = processing.run('native:clip', {
                'INPUT': input_stream,
                'OVERLAY': basin_layer,
                'OUTPUT': 'memory:'
            }, context=context, feedback=feedback)['OUTPUT']

            self.calculate_strahler(clipped_stream, feedback)
            self.extend_main_channel(clipped_stream, feedback)
            self.extend_to_headwaters(clipped_stream, feedback)

            # Calculate Longest Flow Path if enabled
            if calc_longest_path:
                outlet_point = self.get_outlet_point(clipped_stream, feedback)
                
                if outlet_point is not None:
                    farthest_point = self.find_farthest_point_from_outlet(basin_layer, outlet_point, feedback)
                    
                    if farthest_point is not None:
                        # Run r.drain from farthest point
                        drain_layer = self.calculate_longest_flow_path(clipped_stream, basin_layer, dem, context, feedback)
                        
                        if drain_layer is not None and drain_layer.isValid():
                            # Find FIRST intersection between drain and stream network
                            intersection_point, intersecting_feature = self.find_intersection_with_stream(
                                drain_layer, clipped_stream, farthest_point, feedback)
                            
                            if intersection_point is not None and intersecting_feature is not None:
                                # Split drain at first intersection
                                drain_segment = self.split_drain_at_intersection(
                                    drain_layer, intersection_point, farthest_point, feedback)
                                
                                if drain_segment is not None:
                                    # Get max order
                                    max_order = 1
                                    for feat in clipped_stream.getFeatures():
                                        if feat['sthr_final'] is not None and feat['sthr_final'] > max_order:
                                            max_order = feat['sthr_final']
                                    
                                    # Topologically merge drain with stream
                                    if self.add_longest_path_to_stream(clipped_stream, drain_segment, 
                                                                       intersecting_feature, intersection_point,
                                                                       outlet_point, max_order, feedback):
                                        # Recalculate Strahler orders
                                        feedback.pushInfo("Recalculating Strahler orders after merge...")
                                        self.calculate_strahler(clipped_stream, feedback)
                                        self.extend_main_channel(clipped_stream, feedback)
                                        self.extend_to_headwaters(clipped_stream, feedback)
                                    else:
                                        feedback.pushInfo("Merge failed, continuing with original network")
                            else:
                                feedback.pushInfo("No intersection found, skipping longest flow path")
                else:
                    feedback.pushInfo("No outlet point found, skipping longest flow path")

            (stream_sink, stream_dest_id) = self.parameterAsSink(parameters, self.OUTPUT_STREAM, context,
                                                                 clipped_stream.fields(), QgsWkbTypes.LineString, 
                                                                 clipped_stream.crs())
            
            if stream_sink is None:
                raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT_STREAM))

            stream_features = clipped_stream.getFeatures()
            for feature in stream_features:
                stream_sink.addFeature(feature, QgsFeatureSink.FastInsert)

            results[self.OUTPUT_STREAM] = stream_dest_id
        except Exception as e:
            feedback.reportError(f"Error processing stream network: {str(e)}")
            import traceback
            feedback.reportError(traceback.format_exc())
            raise QgsProcessingException(f"Error processing stream network: {str(e)}")

        return results

    def resample_dem(self, dem, new_cell_size, context, feedback):
        extent = dem.extent()
        width = int(extent.width() / new_cell_size)
        height = int(extent.height() / new_cell_size)
        resampled = processing.run("gdal:warpreproject", {
            'INPUT': dem,
            'SOURCE_CRS': dem.crs(),
            'TARGET_CRS': dem.crs(),
            'RESAMPLING': 0,
            'TARGET_RESOLUTION': new_cell_size,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }, context=context, feedback=feedback)['OUTPUT']
        return QgsRasterLayer(resampled, 'resampled_dem', 'gdal')

    def name(self):
        return 'watershedbasindelineation'

    def displayName(self):
        return self.tr('Watershed Basin Delineation')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'arcgeekcalculator'

    def createInstance(self):
        return WatershedBasinDelineationAlgorithm()

    def shortHelpString(self):
        return self.tr("""
        This algorithm delineates a watershed basin based on a Digital Elevation Model (DEM) and a pour point.
        It uses GRASS GIS algorithms for hydrological analysis and watershed delineation.
        
        The stream network is required and will be clipped to the basin boundary with three Strahler stream order calculations:
        - sthr_original: Standard Strahler order calculation
        - sthr_extend: Extended main channel along longest paths
        - sthr_final: Main channel extended to all headwaters
        
        Parameters:
            Input DEM: A raster layer representing the terrain elevation
            Pour Point: The outlet point of the watershed. Snapping to streams is enabled
            Input Stream Network: A line vector layer representing the stream network (required)
            Smoothing Iterations: Number of iterations for smoothing the basin boundary (0-10)
            Smoothing Offset: Offset value for smoothing (0.0-0.5)
            Calculate Longest Flow Path: If enabled, extends the main channel from its headwater to the basin boundary
            
        Outputs:
            Output Basin: A polygon layer representing the delineated watershed basin
            Output Basin Stream Network: A line layer with Strahler orders and longest flow path extension
        """)

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)