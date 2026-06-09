from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (Qgis, QgsProcessing, QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource, QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterNumber, QgsFeature, QgsGeometry, QgsWkbTypes,
                       QgsProcessingException, QgsFeatureSink, QgsPointXY,
                       QgsSpatialIndex, QgsProcessingParameterBoolean,
                       QgsProcessingParameterEnum, QgsCoordinateTransform)
import math

# Qt5/Qt6 - QGIS 3/QGIS 4 compatibility for enums
try:
    LINE_GEOMETRY = Qgis.GeometryType.Line
    POLYGON_GEOMETRY = Qgis.GeometryType.Polygon
except AttributeError:
    LINE_GEOMETRY = QgsWkbTypes.LineGeometry
    POLYGON_GEOMETRY = QgsWkbTypes.PolygonGeometry

try:
    MULTIPOLYGON_WKB = Qgis.WkbType.MultiPolygon
except AttributeError:
    MULTIPOLYGON_WKB = QgsWkbTypes.MultiPolygon

try:
    JOIN_STYLE_ROUND = Qgis.JoinStyle.Round
except AttributeError:
    JOIN_STYLE_ROUND = QgsGeometry.JoinStyleRound


class OptimizedParcelDivisionAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'
    LOT_WIDTH = 'LOT_WIDTH'
    TARGET_AREA = 'TARGET_AREA'
    MERGE_THRESHOLD = 'MERGE_THRESHOLD'
    REMAINDER_MODE = 'REMAINDER_MODE'
    CORNER_WIDTH = 'CORNER_WIDTH'
    TWO_ROWS = 'TWO_ROWS'
    REFERENCE_LINES = 'REFERENCE_LINES'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT, self.tr('Input polygon layer'), [QgsProcessing.TypeVectorPolygon]))
        self.addParameter(QgsProcessingParameterNumber(self.LOT_WIDTH, self.tr('Desired lot width'), QgsProcessingParameterNumber.Double, 10.0))
        self.addParameter(QgsProcessingParameterNumber(self.TARGET_AREA, self.tr('Target lot area (0 = divide by width only)'), QgsProcessingParameterNumber.Double, 0.0, False, 0.0))
        self.addParameter(QgsProcessingParameterNumber(self.MERGE_THRESHOLD, self.tr('Merge threshold (% of average area)'), QgsProcessingParameterNumber.Double, 30.0, False, 0.0, 100.0))
        self.addParameter(QgsProcessingParameterEnum(self.REMAINDER_MODE, self.tr('Remainder distribution'),
                                                     options=[self.tr('Redistribute evenly (all lots equal width)'),
                                                              self.tr('Split between the two corner lots'),
                                                              self.tr('Place in the last lot')],
                                                     defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.CORNER_WIDTH, self.tr('Corner lot width (0 = same as lot width)'), QgsProcessingParameterNumber.Double, 0.0, False, 0.0))
        self.addParameter(QgsProcessingParameterBoolean(self.TWO_ROWS, self.tr('Two rows of lots (split block along its centerline)'), defaultValue=True))
        self.addParameter(QgsProcessingParameterFeatureSource(self.REFERENCE_LINES, self.tr('Frontage reference lines (optional)'), [QgsProcessing.TypeVectorLine], optional=True))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, self.tr('Output divided parcels')))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException(self.tr("Failed to load the input layer. Please check your input data."))

        if source.sourceCrs().isGeographic():
            raise QgsProcessingException(self.tr("The input layer uses a geographic CRS (degrees). Reproject it to a projected CRS (e.g. UTM) so the lot width can be measured in meters."))

        lot_width = self.parameterAsDouble(parameters, self.LOT_WIDTH, context)
        if lot_width <= 0:
            raise QgsProcessingException(self.tr("Lot width must be greater than zero."))
        target_area = self.parameterAsDouble(parameters, self.TARGET_AREA, context)

        merge_threshold = self.parameterAsDouble(parameters, self.MERGE_THRESHOLD, context) / 100.0
        remainder_mode = self.parameterAsEnum(parameters, self.REMAINDER_MODE, context)
        corner_width = self.parameterAsDouble(parameters, self.CORNER_WIDTH, context)
        two_rows = self.parameterAsBool(parameters, self.TWO_ROWS, context)
        ref_source = self.parameterAsSource(parameters, self.REFERENCE_LINES, context)

        remainder_labels = ['redistribute evenly', 'split between corner lots', 'place in last lot']
        feedback.pushInfo(f"Input layer loaded. Feature count: {source.featureCount()}")
        feedback.pushInfo(f"Lot width: {lot_width}")
        feedback.pushInfo(f"Target lot area: {target_area if target_area > 0 else 'not set (dividing by width)'}")
        feedback.pushInfo(f"Merge threshold: {merge_threshold * 100}% of average area")
        feedback.pushInfo(f"Remainder distribution: {remainder_labels[remainder_mode]}")
        feedback.pushInfo(f"Corner lot width: {corner_width if corner_width > 0 else 'same as lot width'}")
        feedback.pushInfo(f"Two rows of lots: {two_rows}")
        feedback.pushInfo(f"Frontage reference lines: {'yes' if ref_source is not None else 'no'}")

        fields = source.fields()
        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                               fields, MULTIPOLYGON_WKB, source.sourceCrs())
        if sink is None:
            raise QgsProcessingException(self.tr("Failed to create the output layer."))

        ref_index, ref_geoms = self._load_reference_lines(ref_source, source.sourceCrs(), context)

        total = 100.0 / source.featureCount() if source.featureCount() else 0
        lots_created = 0

        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            geom = feature.geometry()
            if geom is None or geom.isEmpty():
                feedback.pushInfo(f"Feature {current + 1} has no geometry. Skipping.")
                continue

            if not geom.isGeosValid():
                geom = geom.makeValid()

            parts = [g for g in geom.asGeometryCollection()
                     if not g.isEmpty() and int(QgsWkbTypes.geometryType(g.wkbType())) == int(POLYGON_GEOMETRY)]

            for part in parts:
                if part.area() <= 0:
                    continue

                # Split the block along its centerline (rear lot line) so each
                # row of lots faces its own street.
                if two_rows:
                    rows = self._split_into_rows(part, ref_index, ref_geoms, lot_width)
                else:
                    rows = [QgsGeometry(part)]

                for row in rows:
                    if row.area() <= 0:
                        continue

                    # Find the division axis of this row: its own frontage line
                    # if available, otherwise the long axis of the oriented
                    # minimum bounding box.
                    axis = None
                    half_len = 0.0
                    if ref_index is not None:
                        axis, half_len = self._frontage_axis(row, ref_index, ref_geoms, lot_width)
                    if axis is None:
                        axis, half_len = self._omb_axis(row)

                    if axis is None:
                        pieces = [QgsGeometry(row)]
                    else:
                        cuts = self._build_cuts(axis, row, lot_width, target_area, remainder_mode, corner_width, half_len)
                        pieces = self._split_polygon(row, cuts)
                        pieces = self._merge_small_pieces(pieces, merge_threshold)

                    for piece in pieces:
                        piece.convertToMultiType()
                        out_feature = QgsFeature(fields)
                        out_feature.setAttributes(feature.attributes())
                        out_feature.setGeometry(piece)
                        sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
                        lots_created += 1

            feedback.setProgress(int((current + 1) * total))

        feedback.pushInfo(f"Processing completed. {lots_created} parcels created.")
        return {self.OUTPUT: dest_id}

    def _load_reference_lines(self, ref_source, target_crs, context):
        """Load frontage reference lines into a spatial index, reprojected to the parcel CRS."""
        if ref_source is None:
            return None, None

        transform = None
        if ref_source.sourceCrs().isValid() and ref_source.sourceCrs() != target_crs:
            transform = QgsCoordinateTransform(ref_source.sourceCrs(), target_crs, context.transformContext())

        index = QgsSpatialIndex()
        geoms = {}
        for feature in ref_source.getFeatures():
            geom = feature.geometry()
            if geom is None or geom.isEmpty():
                continue
            geom = QgsGeometry(geom)
            if transform is not None:
                geom.transform(transform)
            indexed = QgsFeature(feature.id())
            indexed.setGeometry(geom)
            index.addFeature(indexed)
            geoms[feature.id()] = geom

        if not geoms:
            return None, None
        return index, geoms

    def _split_into_rows(self, part, ref_index, ref_geoms, lot_width):
        """Split a block along its centerline (rear lot line) into two rows of lots."""
        spine = self._block_centerline(part, ref_index, ref_geoms, lot_width)
        spine = self._as_single_line(spine)
        if spine is None:
            return [QgsGeometry(part)]

        bbox = part.boundingBox()
        margin = bbox.width() + bbox.height()
        extended = spine.extendLine(margin, margin)
        if extended is None or extended.isEmpty():
            extended = spine

        line_points = extended.asPolyline()
        if not line_points:
            return [QgsGeometry(part)]

        first_row = QgsGeometry(part)
        try:
            result_code, new_parts, _ = first_row.splitGeometry(line_points, False)
        except Exception:
            return [QgsGeometry(part)]
        if result_code != 0 or not new_parts:
            return [QgsGeometry(part)]
        rows = [first_row] + list(new_parts)

        # With reference streets available, keep the two-row layout only if
        # every row actually fronts its own street. A parcel served by a
        # single street gets a single row of full-depth lots.
        if ref_index is not None:
            for row in rows:
                axis, _ = self._frontage_axis(row, ref_index, ref_geoms, lot_width)
                if axis is None:
                    return [QgsGeometry(part)]
        return rows

    def _block_centerline(self, part, ref_index, ref_geoms, lot_width):
        """Find the rear lot line of a block.

        With streets on both sides: the line equidistant to both frontages,
        so the two rows get the same depth everywhere. With a single street:
        the frontage offset inwards by half the average block depth. Without
        reference lines: the long axis of the oriented minimum bounding box.
        """
        if ref_index is not None:
            candidates = self._frontage_candidates(part, ref_index, ref_geoms, lot_width)
            if candidates:
                primary = candidates[0]
                depth = part.area() / primary.length()

                # Look for the street on the opposite side of the block: the
                # longest candidate farther away than half the block depth
                # (side streets touch the primary one at corners and are
                # filtered out by this distance test).
                opposite = None
                for candidate in candidates[1:]:
                    if candidate.distance(primary) > depth * 0.5:
                        opposite = candidate
                        break
                if opposite is not None:
                    midline = self._midline_between(primary, opposite, part, lot_width)
                    if midline is not None:
                        return midline

                half_depth = depth / 2.0
                best = None
                best_inside = 0.0
                for sign in (1.0, -1.0):
                    offset = primary.offsetCurve(sign * half_depth, 8, JOIN_STYLE_ROUND, 2.0)
                    if offset is None or offset.isEmpty():
                        continue
                    inter = offset.intersection(part)
                    inside = sum(line.length() for line in self._line_parts(inter)) if inter is not None else 0.0
                    if inside > best_inside:
                        best_inside = inside
                        best = offset
                if best is not None:
                    return best

        axis, _ = self._omb_axis(part)
        return axis

    def _midline_between(self, primary, opposite, part, lot_width):
        """Rear lot line at half the polygon depth.

        At each station along the primary frontage, the segment towards the
        opposite street is intersected with the parcel and the midpoint of the
        interior chord is taken - the true middle of the polygon, regardless
        of how the street axes sit relative to the parcel edges.
        """
        length = primary.length()
        if length <= 0:
            return None

        step = max(min(lot_width / 2.0, length / 8.0), length / 200.0)
        num_samples = max(int(math.ceil(length / step)), 1)

        opposite_length = opposite.length()
        clamp_eps = min(step / 2.0, opposite_length * 0.01)

        samples = []
        for i in range(num_samples + 1):
            distance = min(i * step, length)
            point_a = primary.interpolate(distance).asPoint()
            _, point_b, _, _ = opposite.closestSegmentWithContext(QgsPointXY(point_a))

            # Skip stations the opposite face no longer covers: there the
            # projection clamps to the face endpoint, the midpoint hooks away
            # from the centerline direction, and the extended spine would
            # slash the corner lots diagonally.
            station_b = opposite.lineLocatePoint(QgsGeometry.fromPointXY(QgsPointXY(point_b)))
            if station_b <= clamp_eps or station_b >= opposite_length - clamp_eps:
                continue

            straight_mid = QgsPointXY((point_a.x() + point_b.x()) / 2.0,
                                      (point_a.y() + point_b.y()) / 2.0)

            chord_mid = None
            chord_len = 0.0
            chord = QgsGeometry.fromPolylineXY([QgsPointXY(point_a), QgsPointXY(point_b)]).intersection(part)
            if chord is not None and not chord.isEmpty():
                chord_line = self._as_single_line(chord)
                if chord_line is not None and chord_line.length() > 0:
                    center = chord_line.interpolate(chord_line.length() / 2.0)
                    if center is not None and not center.isEmpty():
                        chord_mid = center.asPoint()
                        chord_len = chord_line.length()

            samples.append((chord_mid, chord_len, straight_mid))

        if not samples:
            return None

        # Center the spine on the polygon only at stations where the chord
        # crosses the full block depth. Where corner chamfers clip the chord
        # its midpoint drifts off the centerline, so those stations are
        # dropped and the straight extension covers the corner instead.
        full_lengths = sorted(s[1] for s in samples if s[0] is not None)
        points = []
        if full_lengths:
            median_len = full_lengths[len(full_lengths) // 2]
            for chord_mid, chord_len, _ in samples:
                if chord_mid is None or chord_len < 0.8 * median_len:
                    continue
                if points and points[-1].distance(chord_mid) < 1e-9:
                    continue
                points.append(chord_mid)

        # Fallback: midpoints between the two street faces.
        if len(points) < 2:
            points = []
            for _, _, straight_mid in samples:
                if points and points[-1].distance(straight_mid) < 1e-9:
                    continue
                points.append(straight_mid)

        if len(points) < 2:
            return None
        midline = QgsGeometry.fromPolylineXY(points)
        # Remove sampling jitter so the spine (and its straight extension at
        # the block ends) keeps a stable direction.
        simplified = midline.simplify(step / 4.0)
        if simplified is not None and not simplified.isEmpty():
            return simplified
        return midline

    def _as_single_line(self, geom):
        """Return the geometry as a single polyline (longest part if it cannot be merged)."""
        if geom is None or geom.isEmpty():
            return None
        if geom.isMultipart():
            geom = geom.mergeLines()
        if geom.isMultipart():
            parts = self._line_parts(geom)
            if not parts:
                return None
            geom = max(parts, key=lambda g: g.length())
        return geom

    def _frontage_candidates(self, part, ref_index, ref_geoms, lot_width):
        """Stretches of reference lines near the parcel, trimmed to the face they front, longest first."""
        search_area = part.buffer(lot_width, 8)
        candidates = []
        for fid in ref_index.intersects(search_area.boundingBox()):
            ref_geom = ref_geoms.get(fid)
            if ref_geom is None:
                continue
            clipped = ref_geom.intersection(search_area)
            if clipped is None or clipped.isEmpty():
                continue
            for line in self._line_parts(clipped):
                for face in self._split_at_corners(line):
                    face = self._trim_to_part(face, part)
                    if face.length() >= lot_width:
                        candidates.append(face)
        candidates.sort(key=lambda g: g.length(), reverse=True)
        return candidates

    def _split_at_corners(self, line, max_angle_deg=45.0):
        """Split a polyline at sharp vertices so each piece is a single block face.

        A road feature drawn as one polyline can turn the block corner; if
        stations ran around the corner the cuts would rotate with it and
        slash the corner lots diagonally. Gentle bends below the threshold
        are kept as a single frontage.
        """
        points = line.asPolyline()
        if len(points) < 3:
            return [line]

        max_angle = math.radians(max_angle_deg)
        faces = []
        current = [points[0], points[1]]
        for i in range(1, len(points) - 1):
            prev_pt, pt, next_pt = points[i - 1], points[i], points[i + 1]
            angle_in = math.atan2(pt.y() - prev_pt.y(), pt.x() - prev_pt.x())
            angle_out = math.atan2(next_pt.y() - pt.y(), next_pt.x() - pt.x())
            turn = abs(angle_out - angle_in)
            if turn > math.pi:
                turn = 2 * math.pi - turn
            if turn > max_angle:
                faces.append(current)
                current = [pt, next_pt]
            else:
                current.append(next_pt)
        faces.append(current)

        if len(faces) == 1:
            return [line]
        return [QgsGeometry.fromPolylineXY(face) for face in faces if len(face) >= 2]

    def _trim_to_part(self, line, part):
        """Trim a frontage line to the stretch that actually fronts the parcel.

        The buffer clipping leaves overshoots past the block corners (and
        around cross streets); stations must be distributed corner to corner,
        so the line is cut back to the extent of the parcel projected onto it.
        """
        length = line.length()
        if length <= 0:
            return line

        min_station = length
        max_station = 0.0
        for vertex in part.vertices():
            station = line.lineLocatePoint(QgsGeometry.fromPointXY(QgsPointXY(vertex.x(), vertex.y())))
            if station < 0:
                continue
            min_station = min(min_station, station)
            max_station = max(max_station, station)

        if max_station - min_station <= 0:
            return line
        try:
            sub_curve = line.constGet().curveSubstring(min_station, max_station)
        except (AttributeError, TypeError):
            return line
        if sub_curve is None or sub_curve.isEmpty():
            return line
        return QgsGeometry(sub_curve)

    def _frontage_axis(self, part, ref_index, ref_geoms, lot_width):
        """Return the longest stretch of reference line near the parcel, to cut perpendicular to it."""
        candidates = self._frontage_candidates(part, ref_index, ref_geoms, lot_width)
        if not candidates:
            return None, 0.0

        bbox = part.boundingBox()
        half_len = math.hypot(bbox.width(), bbox.height()) + 2 * lot_width
        return candidates[0], half_len

    def _omb_axis(self, part):
        """Return the long axis of the oriented minimum bounding box as a line through its center."""
        try:
            omb_geom = part.orientedMinimumBoundingBox()[0]
        except Exception:
            return None, 0.0
        if omb_geom is None or omb_geom.isEmpty():
            return None, 0.0

        rings = omb_geom.asPolygon()
        if not rings or len(rings[0]) < 5:
            return None, 0.0
        ring = rings[0]

        edge1 = math.hypot(ring[1].x() - ring[0].x(), ring[1].y() - ring[0].y())
        edge2 = math.hypot(ring[2].x() - ring[1].x(), ring[2].y() - ring[1].y())
        if edge1 <= 0 or edge2 <= 0:
            return None, 0.0

        if edge1 >= edge2:
            long_len, short_len = edge1, edge2
            ux = (ring[1].x() - ring[0].x()) / edge1
            uy = (ring[1].y() - ring[0].y()) / edge1
        else:
            long_len, short_len = edge2, edge1
            ux = (ring[2].x() - ring[1].x()) / edge2
            uy = (ring[2].y() - ring[1].y()) / edge2

        cx = sum(p.x() for p in ring[:4]) / 4.0
        cy = sum(p.y() for p in ring[:4]) / 4.0
        axis = QgsGeometry.fromPolylineXY([
            QgsPointXY(cx - ux * long_len / 2, cy - uy * long_len / 2),
            QgsPointXY(cx + ux * long_len / 2, cy + uy * long_len / 2)])
        half_len = short_len * 0.55 + 0.1
        return axis, half_len

    def _cut_stations(self, length, lot_width, remainder_mode, corner_width):
        """Compute the distances along the axis where cuts are placed.

        Wider corner lots are reserved first; the interior lots are then
        distributed according to the remainder mode: 0 = equal widths as close
        as possible to the desired width, 1 = exact widths with the leftover
        split between the two end lots, 2 = exact widths with the leftover in
        the last lot.
        """
        if corner_width > 0:
            interior = length - 2.0 * corner_width
            if interior >= lot_width:
                inner = self._cut_stations(interior, lot_width, remainder_mode, 0.0)
                return ([corner_width]
                        + [corner_width + d for d in inner]
                        + [length - corner_width])
            if interior > 0:
                # Only the two corner lots and one middle lot fit
                return [corner_width, length - corner_width]
            # Too short for two corner lots: fall back to standard division

        if remainder_mode == 0:
            num_lots = int(round(length / lot_width))
            if num_lots < 2:
                return []
            width = length / num_lots
            return [i * width for i in range(1, num_lots)]

        num_lots = int(math.floor(length / lot_width))
        if num_lots < 2:
            return []
        remainder = length - num_lots * lot_width
        start_offset = remainder / 2 if remainder_mode == 1 else 0.0
        return [start_offset + i * lot_width for i in range(1, num_lots)]

    def _perpendicular_at(self, axis, distance, half_len, delta):
        """Cut line perpendicular to the local direction of the axis at a station."""
        length = axis.length()
        center = axis.interpolate(distance)
        if center is None or center.isEmpty():
            return None
        point = center.asPoint()
        before = axis.interpolate(max(0.0, distance - delta)).asPoint()
        after = axis.interpolate(min(length, distance + delta)).asPoint()
        angle = math.atan2(after.y() - before.y(), after.x() - before.x()) + math.pi / 2
        dx = math.cos(angle) * half_len
        dy = math.sin(angle) * half_len
        return [QgsPointXY(point.x() + dx, point.y() + dy),
                QgsPointXY(point.x() - dx, point.y() - dy)]

    def _build_cuts(self, axis, row, lot_width, target_area, remainder_mode, corner_width, half_len):
        """Build cut lines perpendicular to the axis at the computed stations.

        The perpendicular direction is computed locally at each station so the
        cuts follow curved frontage lines.
        """
        length = axis.length()
        if target_area > 0:
            stations = self._area_stations(axis, row, target_area, remainder_mode, corner_width, half_len)
        else:
            stations = self._cut_stations(length, lot_width, remainder_mode, corner_width)
        if not stations:
            return []

        delta = max(min(lot_width * 0.25, 0.5), 1e-6)

        cuts = []
        for distance in stations:
            cut = self._perpendicular_at(axis, distance, half_len, delta)
            if cut is not None:
                cuts.append(cut)
        return cuts

    def _chord_depth(self, axis, row, distance, half_len, delta):
        """Depth of the block at a station: length of the perpendicular chord inside the polygon."""
        cut = self._perpendicular_at(axis, distance, half_len, delta)
        if cut is None:
            return 0.0
        chord = QgsGeometry.fromPolylineXY(cut).intersection(row)
        if chord is None or chord.isEmpty():
            return 0.0
        return sum(line.length() for line in self._line_parts(chord))

    def _area_stations(self, axis, row, target_area, remainder_mode, corner_width, half_len):
        """Stations that give each lot the target area.

        A cumulative area profile is built by integrating the block depth
        along the frontage (trapezoidal rule on perpendicular chords); each
        cut is placed where the accumulated area reaches the next lot.
        Because the depth varies along the block, lot widths adjust
        automatically so every lot reaches the target area.
        """
        length = axis.length()
        if length <= 0 or target_area <= 0:
            return []

        step = max(min(length / 100.0, 2.0), 0.1)
        delta = max(step / 4.0, 1e-6)

        profile_stations = [0.0]
        profile_areas = [0.0]
        prev_depth = self._chord_depth(axis, row, 0.0, half_len, delta)
        distance = step
        while True:
            distance = min(distance, length)
            depth = self._chord_depth(axis, row, distance, half_len, delta)
            segment = distance - profile_stations[-1]
            profile_areas.append(profile_areas[-1] + (prev_depth + depth) / 2.0 * segment)
            profile_stations.append(distance)
            prev_depth = depth
            if distance >= length:
                break
            distance += step

        total_area = profile_areas[-1]
        if total_area <= target_area:
            return []

        def station_at(area_value):
            idx = 1
            while idx < len(profile_areas) and profile_areas[idx] < area_value:
                idx += 1
            if idx >= len(profile_areas):
                return None
            a0, a1 = profile_areas[idx - 1], profile_areas[idx]
            s0, s1 = profile_stations[idx - 1], profile_stations[idx]
            if a1 <= a0:
                return s0
            return s0 + (area_value - a0) / (a1 - a0) * (s1 - s0)

        def area_at(station_value):
            idx = 1
            while idx < len(profile_stations) and profile_stations[idx] < station_value:
                idx += 1
            if idx >= len(profile_stations):
                return total_area
            s0, s1 = profile_stations[idx - 1], profile_stations[idx]
            a0, a1 = profile_areas[idx - 1], profile_areas[idx]
            if s1 <= s0:
                return a0
            return a0 + (station_value - s0) / (s1 - s0) * (a1 - a0)

        stations = []
        start_area, end_area = 0.0, total_area
        if corner_width > 0 and length > 2.0 * corner_width:
            stations = [corner_width, length - corner_width]
            start_area = area_at(corner_width)
            end_area = area_at(length - corner_width)

        interior_area = end_area - start_area
        if interior_area <= target_area:
            return stations

        if remainder_mode == 0:
            num_lots = int(round(interior_area / target_area))
            if num_lots < 2:
                return stations
            lot_area = interior_area / num_lots
            targets = [start_area + lot_area * i for i in range(1, num_lots)]
        else:
            num_lots = int(math.floor(interior_area / target_area))
            if num_lots < 2:
                return stations
            remainder = interior_area - num_lots * target_area
            offset = remainder / 2.0 if remainder_mode == 1 else 0.0
            targets = [start_area + offset + target_area * i for i in range(1, num_lots)]

        for area_value in targets:
            station = station_at(area_value)
            if station is not None:
                stations.append(station)
        return sorted(stations)

    def _split_polygon(self, part, cuts):
        pieces = [QgsGeometry(part)]
        for cut in cuts:
            next_pieces = []
            for geom in pieces:
                try:
                    result_code, new_parts, _ = geom.splitGeometry(cut, False)
                except Exception:
                    result_code, new_parts = 1, []
                next_pieces.append(geom)
                if result_code == 0 and new_parts:
                    next_pieces.extend(new_parts)
            pieces = next_pieces
        return [g for g in pieces if g is not None and not g.isEmpty() and g.area() > 0]

    def _merge_small_pieces(self, pieces, threshold_ratio):
        """Greedily merge pieces smaller than the threshold into the neighbor sharing the longest boundary."""
        if threshold_ratio <= 0 or len(pieces) < 2:
            return pieces

        pieces = list(pieces)
        unmergeable = set()
        while len(pieces) > 1:
            areas = [g.area() for g in pieces]
            threshold = threshold_ratio * (sum(areas) / len(areas))

            candidate = None
            for i in sorted(range(len(pieces)), key=lambda k: areas[k]):
                if areas[i] < threshold and id(pieces[i]) not in unmergeable:
                    candidate = i
                    break
            if candidate is None:
                break

            best_neighbor = -1
            best_shared = 0.0
            for j in range(len(pieces)):
                if j == candidate:
                    continue
                shared = self._shared_boundary_length(pieces[candidate], pieces[j])
                if shared > best_shared:
                    best_shared = shared
                    best_neighbor = j

            if best_neighbor < 0:
                unmergeable.add(id(pieces[candidate]))
                continue

            merged = pieces[best_neighbor].combine(pieces[candidate])
            if merged is None or merged.isEmpty():
                unmergeable.add(id(pieces[candidate]))
                continue

            pieces = [g for k, g in enumerate(pieces) if k not in (candidate, best_neighbor)]
            pieces.append(merged)
        return pieces

    def _shared_boundary_length(self, geom_a, geom_b):
        intersection = geom_a.intersection(geom_b)
        if intersection is None or intersection.isEmpty():
            return 0.0
        total = 0.0
        for line in self._line_parts(intersection):
            total += line.length()
        return total

    def _line_parts(self, geom):
        """Extract the linear parts of a geometry (handles geometry collections)."""
        parts = []
        for part in geom.constParts():
            if part is None:
                continue
            if int(QgsWkbTypes.geometryType(part.wkbType())) == int(LINE_GEOMETRY):
                parts.append(QgsGeometry(part.clone()))
        return parts

    def name(self):
        return 'optimizedparceldivision'

    def displayName(self):
        return self.tr('Optimized Parcel Division')

    def group(self):
        return self.tr('ArcGeek Calculator')

    def groupId(self):
        return 'arcgeekcalculator'

    def shortHelpString(self):
        return self.tr("""
        This algorithm divides parcels into lots of a specified width.

        Parameters:
        - Input polygon layer: The layer containing the parcels to be divided.
        - Desired lot width: The width you want each resulting lot to have (in CRS units, requires a projected CRS).
        - Target lot area: If greater than 0 (e.g. 200 for 200 m2), lots are sized by area instead of width: each cut is placed where the accumulated block area reaches the next lot, so lot widths adjust automatically where the block depth varies. With "Redistribute evenly" all lots in a row get exactly the same area, as close as possible to the target; corner lots absorb the adjustment in the other modes. The desired lot width is ignored when this is set.
        - Merge threshold: Percentage of average lot area below which small leftover lots are merged into their neighbor.
        - Remainder distribution: What to do when the frontage is not an exact multiple of the lot width. "Redistribute evenly" resizes all lots equally so no slivers are created (recommended, like Civil 3D's redistribute remainder); "Split between the two corner lots" makes both corner lots slightly wider; "Place in the last lot" keeps exact widths and accumulates the leftover at the end.
        - Corner lot width: If greater than 0, the first and last lot of each row are reserved with this width. Corner lots are usually designed wider because they have setbacks on two streets, or to give them frontage on the side street. Use 0 to make corner lots the same width as the rest.
        - Two rows of lots: If checked (default), the block is split along its centerline (rear lot line) so the lots in each row face their own street, instead of running from street to street. When frontage reference lines are provided, the split is automatic per parcel: it only happens if each row has its own street to front - parcels served by a single street keep one row of full-depth lots. Without reference lines, uncheck this for single-frontage parcels.
        - Frontage reference lines (optional): A line layer (e.g. street fronts or street centerlines). When provided, lots are measured along the nearest frontage line and cut perpendicular to it - this is the recommended option for irregular or curved parcels.

        How it works:
        1. If "Two rows of lots" is checked, the block is split along its rear lot line: the line equidistant between the two street frontages when there are streets on both sides (both rows get the same depth), the frontage offset by half the block depth when there is a single street, or the long axis of the oriented minimum bounding box when no frontage lines are given.
        2. For each row, a division axis is determined: its own nearest frontage line if provided, otherwise the long axis of the row's oriented minimum bounding box.
        3. Cut lines are generated perpendicular to that axis: first the corner lots are reserved (if a corner lot width is given), then the interior lots are distributed according to the remainder mode. Along curved frontage lines, each cut follows the local perpendicular.
        4. Each parcel is split independently with its own cut lines (neighboring parcels never affect each other).
        5. Lots smaller than the merge threshold are merged into the neighbor (within the same row) sharing the longest boundary.

        Note: Without frontage lines the tool works best on rectangular or convex blocks. For L-shaped or strongly curved blocks, provide frontage reference lines.
        """)

    def createInstance(self):
        return OptimizedParcelDivisionAlgorithm()

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)
