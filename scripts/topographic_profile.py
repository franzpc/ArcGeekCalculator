from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource, QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer, QgsProcessingParameterField,
    QgsProcessingParameterNumber, QgsProcessingParameterFileDestination,
    QgsProcessingParameterBoolean,
    QgsProcessingException, QgsProject, QgsRasterLayer, QgsVectorLayer,
    QgsPointXY, QgsMessageLog, Qgis, QgsGeometry
)
import os
import webbrowser
import urllib.parse
import plotly.graph_objects as go

class TopographicProfileAlgorithm(QgsProcessingAlgorithm):
    """
    Generates an interactive topographic profile from a line and a DEM, with an optional points layer.
    The profile area is filled with a semi-transparent vibrant color and the main profile line is drawn in a vivid tone.
    A summary paragraph is displayed below the chart.

    Parameters:
      - Input Line Layer: Vector line used to sample the DEM.
      - Digital Elevation Model (DEM): Raster layer with elevation data.
      - Points Layer (optional): Vector points to label on the profile.
      - Name Field for Points (optional): Field containing point names.
      - Maximum Distance for Points (meters): Only points within this distance from the line are considered.
      - Apply smoothing?: If enabled (default), a moving average is applied to smooth the profile:
            • No smoothing if total distance < 1000 m.
            • Moderate smoothing (window = 3) if total distance is between 1000 and 10000 m.
            • Heavy smoothing (window = 7) if total distance > 10 km.
      - Invert profile direction?: If enabled, the profile is drawn from end to start.
            (Note: The DEM-derived Z-values for points are always sampled directly from the DEM.)
      - Output HTML File: Path to save the interactive HTML output.
      
    Note: All layers must share the same coordinate system (CRS).
    """

    INPUT_LINE = 'INPUT_LINE'
    INPUT_DEM = 'INPUT_DEM'
    INPUT_POINTS = 'INPUT_POINTS'
    NAME_FIELD = 'NAME_FIELD'
    POINT_THRESHOLD = 'POINT_THRESHOLD'
    SMOOTH = 'SMOOTH'
    INVERT_PROFILE = 'INVERT_PROFILE'
    OUTPUT_HTML = 'OUTPUT_HTML'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return TopographicProfileAlgorithm()

    def name(self):
        return 'topographicprofile_singlefill'

    def displayName(self):
        return self.tr('Topographic Profile')

    def group(self):
        return self.tr('Terrain Analysis')

    def groupId(self):
        return 'terrainanalysis'

    def shortHelpString(self):
        return self.tr(
            "Generates an interactive topographic profile from a line and a DEM, with optional points.\n"
            "Parameters:\n"
            " - Input Line Layer: Vector line used to sample the DEM.\n"
            " - Digital Elevation Model (DEM): Raster layer with elevation data.\n"
            " - Points Layer (optional): Vector points to label on the profile.\n"
            " - Name Field for Points (optional): Field containing point names.\n"
            " - Maximum Distance for Points (meters): Only points within this distance from the line are considered.\n"
            " - Apply smoothing?: If enabled (default), a moving average is applied:\n"
            "      • No smoothing if total distance < 1000 m.\n"
            "      • Moderate smoothing (window = 3) if distance is between 1000 and 10000 m.\n"
            "      • Heavy smoothing (window = 7) if distance > 10 km.\n"
            " - Invert profile direction?: If enabled, the profile is drawn from end to start. (The DEM-derived\n"
            "      Z-values for points are always obtained directly from the DEM.)\n"
            " - Output HTML File: Path to save the interactive HTML output.\n"
            "All layers must share the same CRS."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_LINE,
                self.tr('Input Line Layer'),
                [QgsProcessing.TypeVectorLine]
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DEM,
                self.tr('Digital Elevation Model (DEM)')
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_POINTS,
                self.tr('Points Layer'),
                [QgsProcessing.TypeVectorPoint],
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.NAME_FIELD,
                self.tr('Name Field for Points'),
                parentLayerParameterName=self.INPUT_POINTS,
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.POINT_THRESHOLD,
                self.tr('Maximum Distance for Points (meters)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=300.0,
                minValue=0.0
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SMOOTH,
                self.tr('Apply smoothing?'),
                defaultValue=True
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.INVERT_PROFILE,
                self.tr('Invert profile direction?'),
                defaultValue=False
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_HTML,
                self.tr('Output HTML File'),
                'HTML files (*.html)'
            )
        )

    def smooth(self, data, window):
        """Apply a simple moving average with the given window size."""
        if window <= 1:
            return data
        n = len(data)
        smoothed = []
        half_window = window // 2
        for i in range(n):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            avg = sum(data[start:end]) / (end - start)
            smoothed.append(avg)
        return smoothed

    def processAlgorithm(self, parameters, context, feedback):
        # Retrieve input parameters.
        line_layer = self.parameterAsSource(parameters, self.INPUT_LINE, context)
        dem_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
        points_layer = self.parameterAsVectorLayer(parameters, self.INPUT_POINTS, context)
        name_field = self.parameterAsString(parameters, self.NAME_FIELD, context)
        threshold = self.parameterAsDouble(parameters, self.POINT_THRESHOLD, context)
        smooth_enabled = self.parameterAsBool(parameters, self.SMOOTH, context)
        invert_profile = self.parameterAsBool(parameters, self.INVERT_PROFILE, context)
        output_html = self.parameterAsFileOutput(parameters, self.OUTPUT_HTML, context)

        if not line_layer:
            raise QgsProcessingException("No input line layer specified")
        if not dem_layer.isValid():
            raise QgsProcessingException("Invalid DEM layer")

        # Ensure that line layer and DEM share the same CRS.
        if dem_layer.crs() != line_layer.sourceCrs():
            feedback.pushWarning("Line layer CRS does not match DEM CRS. Reloading line layer with DEM CRS.")
            line_layer = QgsVectorLayer(line_layer.dataProvider().dataSourceUri(), 
                                        line_layer.name(), 
                                        line_layer.dataProvider().name())
            line_layer.setCrs(dem_layer.crs())

        # Check points layer CRS if provided.
        if points_layer and points_layer.isValid():
            if points_layer.sourceCrs() != dem_layer.crs():
                feedback.reportError("Not all layers share the same CRS. Please reproject them.")
                raise QgsProcessingException("CRS mismatch among layers.")

        features = list(line_layer.getFeatures())
        if not features:
            raise QgsProcessingException("No features in line layer")
        line_geometry = features[0].geometry()
        if not dem_layer.extent().contains(line_geometry.boundingBox()):
            raise QgsProcessingException("Line is outside DEM bounds")

        # Check that the line forms a single continuous line.
        if line_geometry.isMultipart():
            multi = line_geometry.asMultiPolyline()
            if len(multi) > 1:
                tol = 0.001
                for i in range(1, len(multi)):
                    if multi[i-1][-1].distance(multi[i][0]) > tol:
                        raise QgsProcessingException("The input line is not continuous (contains branches or discontinuities).")
                # Merge continuous segments.
                points = []
                for poly in multi:
                    points.extend(poly)
            else:
                points = [QgsPointXY(p) for p in multi[0]]
        else:
            points = [QgsPointXY(p) for p in line_geometry.asPolyline()]
        if not points:
            raise QgsProcessingException("No points found in line geometry")

        # Set a default for sample_interval.
        sample_interval = 10

        # Sample raw DEM elevations along the line.
        raw_distances = []
        raw_elevations = []
        current_distance = 0
        no_data_value = dem_layer.dataProvider().sourceNoDataValue(1)
        for i in range(len(points) - 1):
            if feedback.isCanceled():
                break
            start_point = points[i]
            end_point = points[i+1]
            segment_distance = start_point.distance(end_point)
            total_length = sum(points[j].distance(points[j+1]) for j in range(len(points) - 1))
            if total_length > 10000:
                sample_interval = 100
            elif total_length > 5000:
                sample_interval = 50
            elif total_length > 1000:
                sample_interval = 20
            else:
                sample_interval = 10
            num_samples = max(1, int(segment_distance / sample_interval))
            for j in range(num_samples):
                t = j / num_samples
                x = start_point.x() + t * (end_point.x() - start_point.x())
                y = start_point.y() + t * (end_point.y() - start_point.y())
                pt = QgsPointXY(x, y)
                elevation, ok = dem_layer.dataProvider().sample(pt, 1)
                if ok and elevation != no_data_value and elevation is not None:
                    current_distance += segment_distance / num_samples
                    raw_distances.append(current_distance)
                    raw_elevations.append(elevation)
                else:
                    feedback.pushInfo(f"Warning: Invalid elevation at ({x:.3f}, {y:.3f})")
        if not raw_elevations:
            raise QgsProcessingException("No valid elevations obtained")

        # Optionally invert the profile.
        if invert_profile:
            max_distance = max(raw_distances)
            graph_distances = [max_distance - d for d in reversed(raw_distances)]
            graph_elevations = list(reversed(raw_elevations))
        else:
            graph_distances = raw_distances
            graph_elevations = raw_elevations

        # Convert distances if needed (to km if > 5000 m).
        if max(graph_distances) > 5000:
            distances_converted = [d / 1000 for d in graph_distances]
            distance_unit = 'km'
        else:
            distances_converted = graph_distances
            distance_unit = 'm'

        def get_elevation_at_distance(target, dist_list, elev_list):
            """Interpolate elevation at a given target distance."""
            if target <= dist_list[0]:
                return elev_list[0]
            elif target >= dist_list[-1]:
                return elev_list[-1]
            else:
                for i in range(1, len(dist_list)):
                    if dist_list[i] >= target:
                        d0 = dist_list[i-1]
                        d1 = dist_list[i]
                        e0 = elev_list[i-1]
                        e1 = elev_list[i]
                        frac = (target - d0) / (d1 - d0)
                        return e0 + frac * (e1 - e0)
            return elev_list[-1]

        # Process optional points layer: use raw DEM values for Z.
        labeled_points = []
        if points_layer and points_layer.isValid():
            for feat in points_layer.getFeatures():
                if feedback.isCanceled():
                    break
                geom = feat.geometry()
                if geom is None or geom.isEmpty() or geom.type() != 0:
                    continue
                pt = geom.asPoint()
                pt_geom = QgsGeometry.fromPointXY(pt)
                dist_to_line = line_geometry.distance(pt_geom)
                if dist_to_line <= self.parameterAsDouble(parameters, self.POINT_THRESHOLD, context):
                    proj_distance = line_geometry.lineLocatePoint(pt_geom)
                    if proj_distance is None:
                        continue
                    label_distance_raw = proj_distance
                    if invert_profile:
                        max_distance_raw = max(raw_distances)
                        label_distance = max_distance_raw - label_distance_raw
                    else:
                        label_distance = label_distance_raw
                    if distance_unit == 'km':
                        label_distance_conv = label_distance / 1000
                    else:
                        label_distance_conv = label_distance
                    if name_field:
                        name = feat[name_field]
                    else:
                        name = f"Point {feat.id()}"
                    elevation, ok = dem_layer.dataProvider().sample(pt, 1)
                    if not (ok and elevation != no_data_value and elevation is not None):
                        feedback.pushInfo(f"Warning: Could not sample DEM at point {pt}")
                        continue
                    labeled_points.append({
                        "distance": label_distance_conv,
                        "elevation": elevation,
                        "name": name
                    })

        # Apply smoothing to the graph data if enabled.
        if smooth_enabled:
            total_distance_m = max(distances_converted)*1000 if distance_unit == 'km' else max(distances_converted)
            if total_distance_m < 1000:
                smooth_window = 1
            elif total_distance_m <= 10000:
                smooth_window = 3
            else:
                smooth_window = 7
        else:
            smooth_window = 1

        smoothed_elevations = self.smooth(graph_elevations, smooth_window)

        # Generate final HTML using the smoothed graph elevations, but raw DEM values for the table and summary.
        html_content = self._generate_html(distances_converted, smoothed_elevations, raw_elevations, labeled_points, distance_unit, sample_interval)

        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html_content)

        webbrowser.open("file://" + output_html)
        return {self.OUTPUT_HTML: output_html}

    def _generate_html(self, distances, smoothed_elevations, raw_elevations, labeled_points, distance_unit, sample_interval):
        """
        Builds the Plotly figure using the smoothed elevations for the chart and forces
        the X-axis to start at 0.
        """
        elev_min = min(smoothed_elevations)
        elev_max = max(smoothed_elevations)

        x_min = 0
        x_max = max(distances)

        fig = go.Figure()

        # Main line with vibrant color and thicker line.
        fig.add_trace(go.Scatter(
            x=distances,
            y=smoothed_elevations,
            mode='lines',
            name='Topographic Profile',
            line=dict(color='#FF5722', width=3)
        ))

        # Filled area with semi-transparent modern color.
        fig.add_trace(go.Scatter(
            x=distances,
            y=smoothed_elevations,
            mode='none',
            fill='tozeroy',
            fillcolor='rgba(255, 235, 59,0.7)',
            name='Profile Area'
        ))

        # Labeled points with larger markers and improved text.
        if labeled_points:
            fig.add_trace(go.Scatter(
                x=[pt["distance"] for pt in labeled_points],
                y=[pt["elevation"] for pt in labeled_points],
                mode='markers+text',
                name='Points',
                text=[pt["name"] for pt in labeled_points],
                textposition='top center',
                textfont=dict(size=14, color='#212121'),
                marker=dict(color='#4CAF50', size=12)
            ))

        fig.update_layout(
            title="Topographic Profile",
            xaxis_title=f"Distance ({distance_unit})",
            yaxis_title="Elevation (m a.s.l.)",
            template="plotly_white",
            hovermode="x unified",
            autosize=True,
            height=600,
            margin=dict(l=50, r=50, t=80, b=50)
        )

        # Force X-axis range to [0, x_max].
        fig.update_xaxes(
            range=[x_min, x_max],
            showgrid=False,
            showline=True,
            linecolor='black',
            ticks="outside",
            ticklen=5,
            tickcolor='black'
        )
        # Force Y-axis range to [elev_min, elev_max].
        fig.update_yaxes(
            range=[elev_min, elev_max],
            showgrid=False,
            showline=True,
            linecolor='black',
            ticks="outside",
            ticklen=5,
            tickcolor='black'
        )

        chart_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
        return self._create_html_template(chart_html, distances, raw_elevations, labeled_points, distance_unit, sample_interval)

    def _create_html_template(self, chart_html, distances, raw_elevations, labeled_points, distance_unit, sample_interval):
        """
        Creates the final HTML output embedding the Plotly chart and a merged table.
        The table uses the raw DEM values so that each point's Z-value exactly matches the DEM.
        """
        # Build merged sample data from all sampling points using raw DEM values.
        sample_data = [{"distance": d, "elevation": e, "name": ""} for d, e in zip(distances, raw_elevations)]
        tol = (sample_interval / 1000) * 0.5 if distance_unit == 'km' else sample_interval * 0.5
        for lp in labeled_points:
            found = False
            for sd in sample_data:
                if abs(sd["distance"] - lp["distance"]) < tol:
                    # Replace the sample's elevation with the labeled point's DEM value.
                    sd["elevation"] = lp["elevation"]
                    sd["name"] = (sd["name"] + ", " + lp["name"] if sd["name"] else lp["name"])
                    found = True
                    break
            if not found:
                sample_data.append(lp)
        sample_data.sort(key=lambda x: x["distance"])

        merged_csv = "Distance,Elevation,Name\n" + "\n".join(
            f"{sd['distance']:.3f},{sd['elevation']:.2f},{sd['name']}"
            for sd in sample_data
        )
        encoded_csv = urllib.parse.quote(merged_csv)
        csv_data_uri = f"data:text/csv;charset=utf-8,{encoded_csv}"

        count = len(raw_elevations)
        total_distance = max(distances)
        elev_min = min(raw_elevations)
        elev_max = max(raw_elevations)
        elev_diff = elev_max - elev_min

        summary_text = (
            f"This topographic profile has {count} sampled points, "
            f"with a total distance of {total_distance:.2f} {distance_unit}, "
            f"a minimum elevation of {elev_min:.2f} m, "
            f"a maximum elevation of {elev_max:.2f} m, "
            f"and an elevation difference of {elev_diff:.2f} m."
        )

        table_rows = "\n".join(
            f"<tr><td>{sd['distance']:.3f}</td><td>{sd['elevation']:.2f}</td><td>{sd['name']}</td></tr>"
            for sd in sample_data
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Topographic Profile</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 20px;
    }}
    .container {{
      max-width: 1200px;
      margin: auto;
      width: 100%;
    }}
    .chart-container {{
      margin-bottom: 20px;
    }}
    .summary-info {{
      margin-top: 20px;
      font-size: 1rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 20px;
    }}
    th, td {{
      border: 1px solid #ccc;
      padding: 10px;
      text-align: center;
    }}
    th {{
      background: #f0f0f0;
    }}
    .button {{
      display: inline-block;
      padding: 10px 20px;
      margin: 5px;
      background-color: #4CAF50;
      color: white;
      text-decoration: none;
      border-radius: 4px;
      cursor: pointer;
    }}
    .button.info {{
      background-color: #03A9F4;
    }}
    .button.download {{
      background-color: #8BC34A;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Topographic Profile by ArcGeek</h1>
    <div class="chart-container">
      {chart_html}
    </div>
    <div class="summary-info">
      <p>{summary_text}</p>
    </div>
    <div>
      <button class="button info" id="toggleTableBtn">Show/Hide Table</button>
      <a class="button download" id="downloadCSVBtn" download="profile_points.csv" href="{csv_data_uri}">Download CSV</a>
    </div>
    <div id="dataTable" style="display: none;">
      <table>
        <thead>
          <tr>
            <th>Distance ({distance_unit})</th>
            <th>Elevation (m)</th>
            <th>Name</th>
          </tr>
        </thead>
        <tbody>
          {table_rows}
        </tbody>
      </table>
    </div>
  </div>
  <script>
    document.getElementById('toggleTableBtn').addEventListener('click', function() {{
      var table = document.getElementById('dataTable');
      if (table.style.display === 'none') {{
        table.style.display = 'block';
      }} else {{
        table.style.display = 'none';
      }}
    }});
  </script>
</body>
</html>
"""

# End of plugin code
