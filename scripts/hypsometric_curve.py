from qgis.core import *
from qgis.analysis import QgsZonalStatistics

try:
    from qgis.PyQt.QtGui import QColor, QPainter, QFont, QImage, QPen, QBrush, QPolygonF
    from qgis.PyQt.QtCore import QSize, Qt, QPointF, QRectF
    QT_VERSION = "PyQt_via_QGIS"
except ImportError:
    try:
        from PyQt6.QtGui import QColor, QPainter, QFont, QImage, QPen, QBrush, QPolygonF
        from PyQt6.QtCore import QSize, Qt, QPointF, QRectF
        QT_VERSION = "PyQt6"
    except ImportError:
        try:
            from PyQt5.QtGui import QColor, QPainter, QFont, QImage, QPen, QBrush, QPolygonF
            from PyQt5.QtCore import QSize, Qt, QPointF, QRectF
            QT_VERSION = "PyQt5"
        except ImportError:
            raise ImportError("No compatible Qt version found. Please install PyQt5 or PyQt6.")

import processing
import os
import csv
import platform
import math

# ===== LAZY LOADING SYSTEM =====
# Plotly will be imported only when the tool is executed
# This prevents QGIS from freezing during plugin initialization

PLOTLY_AVAILABLE = None

def _check_plotly_dependency():
    """Check if plotly is available. Called only when tool is used."""
    global PLOTLY_AVAILABLE
    
    if PLOTLY_AVAILABLE is None:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            PLOTLY_AVAILABLE = True
        except ImportError:
            PLOTLY_AVAILABLE = False
    
    return PLOTLY_AVAILABLE

def _get_plotly_missing_message():
    """Returns a user-friendly message if plotly is missing."""
    if not _check_plotly_dependency():
        return (
            "This tool requires the 'plotly' Python library which is not installed.\n\n"
            "To install it, open the OSGeo4W Shell or your Python environment and run:\n"
            "pip install plotly\n\n"
            "After installation, restart QGIS."
        )
    return None
# ===== END LAZY LOADING SYSTEM =====


def get_basin_stage(hi):
    if hi >= 0.60:
        return "Young stage"
    elif 0.35 <= hi < 0.60:
        return "Mature stage"
    else:
        return "Old stage"

def calculate_hypsometric_integral(elevation, area):
    sorted_pairs = sorted(zip(elevation, area), reverse=True)
    elevation_sorted, area_sorted = zip(*sorted_pairs)
    
    min_elev = elevation_sorted[-1]
    max_elev = elevation_sorted[0]
    
    relative_height = [(e - min_elev) / (max_elev - min_elev) for e in elevation_sorted]
    
    total_area = area_sorted[0]
    relative_area = [a / total_area for a in area_sorted]
    
    area_accum = 0
    for i in range(len(relative_area) - 1):
        a1 = relative_area[i]
        a2 = relative_area[i + 1]
        h1 = relative_height[i]
        h2 = relative_height[i + 1]
        area_accum += abs(a2 - a1) * ((h1 + h2) / 2)
    
    return area_accum

def calculate_hypsometric_integral_pixel_method(elevation_values):
    if len(elevation_values) <= 10:
        return None
    
    elevation_values = sorted(elevation_values, reverse=True)
    
    min_elev = elevation_values[-1]
    max_elev = elevation_values[0]
    
    if max_elev <= min_elev:
        return None
    
    hh = [(elev - min_elev) / (max_elev - min_elev) for elev in elevation_values]
    aa = [(i + 1) / len(elevation_values) for i in range(len(elevation_values))]
    
    x = [i / 100 for i in range(101)]
    y = []
    for xi in x:
        idx = int(xi * (len(aa) - 1))
        if idx >= len(hh):
            idx = len(hh) - 1
        y.append(hh[idx])
    
    area_accum = 0
    for n in range(len(x) - 1):
        a1 = x[n]
        a2 = x[n + 1]
        h1 = y[n]
        h2 = y[n + 1]
        area_accum += (a2 - a1) * h2 + ((a2 - a1) * (h1 - h2)) / 2
    
    return area_accum

def simplify_data_for_visualization(elevation_values_sorted, target_points=100):
    if len(elevation_values_sorted) <= target_points:
        return elevation_values_sorted
    
    step = len(elevation_values_sorted) // target_points
    simplified = []
    for i in range(0, len(elevation_values_sorted), step):
        simplified.append(elevation_values_sorted[i])
    
    if simplified[-1] != elevation_values_sorted[-1]:
        simplified.append(elevation_values_sorted[-1])
    
    return simplified

def extract_pixel_values_from_basin(dem_layer, basin_layer, feedback):
    try:
        dem_provider = dem_layer.dataProvider()
        dem_extent = dem_layer.extent()
        dem_width = dem_layer.width()
        dem_height = dem_layer.height()
        
        x_res = dem_layer.rasterUnitsPerPixelX()
        y_res = dem_layer.rasterUnitsPerPixelY()
        
        feedback.pushInfo(f"DEM dimensions: {dem_width} x {dem_height}")
        feedback.pushInfo(f"Pixel resolution: {x_res:.3f} x {y_res:.3f}")
        
        basin_geom = None
        for feature in basin_layer.getFeatures():
            if basin_geom is None:
                basin_geom = feature.geometry()
            else:
                basin_geom = basin_geom.combine(feature.geometry())
        
        if basin_geom is None:
            feedback.reportError("No basin geometry found")
            return None
        
        elevation_values = []
        total_pixels = 0
        pixels_in_basin = 0
        
        dem_block = dem_provider.block(1, dem_extent, dem_width, dem_height)
        
        for row in range(dem_height):
            for col in range(dem_width):
                x = dem_extent.xMinimum() + (col + 0.5) * x_res
                y = dem_extent.yMaximum() - (row + 0.5) * y_res
                
                point = QgsPointXY(x, y)
                
                if basin_geom.contains(point):
                    pixels_in_basin += 1
                    
                    elevation = dem_block.value(row, col)
                    
                    if not dem_block.isNoData(row, col) and elevation is not None:
                        elevation_values.append(elevation)
                
                total_pixels += 1
        
        feedback.pushInfo(f"Total pixels processed: {total_pixels}")
        feedback.pushInfo(f"Pixels in basin: {pixels_in_basin}")
        feedback.pushInfo(f"Valid elevation values: {len(elevation_values)}")
        
        if len(elevation_values) < 10:
            feedback.reportError("Insufficient elevation data for hypsometric analysis")
            return None
        
        return elevation_values
    
    except Exception as e:
        feedback.reportError(f"Error extracting pixel values: {str(e)}")
        return None

def generate_hypsometric_curve_qgis(dem_layer, basin_layer, output_folder, feedback):
    try:
        result = processing.run(
            "qgis:hypsometriccurves", 
            {
                'INPUT_DEM': dem_layer,
                'BOUNDARY_LAYER': basin_layer,
                'STEP': 100,
                'USE_PERCENTAGE': False,
                'OUTPUT_DIRECTORY': output_folder
            },
            feedback=feedback
        )

        csv_files = [f for f in os.listdir(output_folder) if f.startswith('histogram_') and f.endswith('.csv')]
        if not csv_files:
            feedback.reportError("No histogram CSV file found in the output directory.")
            return None

        csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_folder, x)), reverse=True)
        csv_file = os.path.join(output_folder, csv_files[0])
        feedback.pushInfo(f"Using most recent histogram file: {csv_file}")

        try:
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                data = list(reader)

            area = [float(row[0]) for row in data]
            elevation = [float(row[1]) for row in data]

            total_area = max(area)
            total_area_km2 = total_area / 1e6

            hi = calculate_hypsometric_integral(elevation, area)
            stage = get_basin_stage(hi)
            feedback.pushInfo(f"Calculated Hypsometric Integral: {hi:.3f} ({stage})")

            min_elev, max_elev = min(elevation), max(elevation)
            relative_height = [(e - min_elev) / (max_elev - min_elev) for e in elevation]
            relative_area = [a / total_area for a in area]

            relative_height = relative_height[::-1]
            relative_area = [1 - a for a in relative_area[::-1]]

            processed_csv = os.path.join(output_folder, 'hypsometric_processed.csv')
            with open(processed_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Elevation (m)', 'Area (m²)', 'Relative Height (h/H)', 'Relative Area (a/A)'])
                for i in range(len(elevation)):
                    writer.writerow([
                        elevation[i],
                        area[i],
                        relative_height[len(elevation)-1-i],
                        relative_area[len(elevation)-1-i]
                    ])

            curve_color = QColor(19, 138, 249)
            point_color = QColor(203, 67, 53)

            create_static_image(relative_area, relative_height, curve_color, point_color, output_folder, hi, total_area_km2, stage)

            if _check_plotly_dependency():
                create_interactive_html(relative_area, relative_height, area, elevation, curve_color, point_color, output_folder, hi, total_area_km2, stage, 0)
                feedback.pushInfo("Interactive hypsometric curve (HTML) generated.")
            else:
                feedback.pushInfo("Plotly is not installed; interactive version not generated.")

            feedback.pushInfo(f"Processed data saved to: {processed_csv}")
            feedback.pushInfo(f"Hypsometric curve analysis completed. Results in: {output_folder}")
            
            return {
                'CSV': processed_csv,
                'PNG': os.path.join(output_folder, 'hypsometric_curve.png'),
                'HTML': os.path.join(output_folder, 'hypsometric_curve_interactive.html') if _check_plotly_dependency() else None,
                'HI': hi,
                'TOTAL_AREA': total_area_km2,
                'STAGE': stage
            }

        except Exception as e:
            feedback.reportError(f"Error processing CSV file: {str(e)}")
            return None

    except Exception as e:
        feedback.reportError(f"Error in generate_hypsometric_curve: {str(e)}")
        return None

def generate_hypsometric_curve_pixel(dem_layer, basin_layer, output_folder, feedback):
    try:
        feedback.pushInfo("Starting pixel-by-pixel hypsometric analysis...")
        
        elevation_values = extract_pixel_values_from_basin(dem_layer, basin_layer, feedback)
        if elevation_values is None:
            return None
        
        hi = calculate_hypsometric_integral_pixel_method(elevation_values)
        if hi is None:
            feedback.reportError("Failed to calculate hypsometric integral")
            return None
        
        stage = get_basin_stage(hi)
        feedback.pushInfo(f"Calculated Hypsometric Integral: {hi:.3f} ({stage})")
        
        basin_area_m2 = sum([f.geometry().area() for f in basin_layer.getFeatures()])
        total_area_km2 = basin_area_m2 / 1e6
        
        elevation_values_sorted = sorted(elevation_values, reverse=True)
        min_elev = min(elevation_values)
        max_elev = max(elevation_values)
        
        elevation_simplified = simplify_data_for_visualization(elevation_values_sorted, target_points=300)
        
        relative_height = [(e - min_elev) / (max_elev - min_elev) for e in elevation_simplified]
        relative_area = [(i + 1) / len(elevation_simplified) for i in range(len(elevation_simplified))]
        
        area_m2 = [basin_area_m2 * rel_area for rel_area in relative_area]
        
        processed_csv = os.path.join(output_folder, 'hypsometric_processed.csv')
        with open(processed_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Elevation (m)', 'Area (m²)', 'Relative Height (h/H)', 'Relative Area (a/A)'])
            for i in range(len(elevation_simplified)):
                writer.writerow([
                    elevation_simplified[i],
                    area_m2[i],
                    relative_height[i],
                    relative_area[i]
                ])
        
        curve_color = QColor(19, 138, 249)
        point_color = QColor(203, 67, 53)
        
        create_static_image(relative_area, relative_height, curve_color, point_color, output_folder, hi, total_area_km2, stage)
        
        if _check_plotly_dependency():
            create_interactive_html(relative_area, relative_height, area_m2, elevation_simplified, curve_color, point_color, output_folder, hi, total_area_km2, stage, 1)
            feedback.pushInfo("Interactive hypsometric curve (HTML) generated.")
        else:
            feedback.pushInfo("Plotly is not installed; interactive version not generated.")
        
        feedback.pushInfo(f"Processed data saved to: {processed_csv}")
        feedback.pushInfo(f"Hypsometric curve analysis completed. Results in: {output_folder}")
        
        return {
            'CSV': processed_csv,
            'PNG': os.path.join(output_folder, 'hypsometric_curve.png'),
            'HTML': os.path.join(output_folder, 'hypsometric_curve_interactive.html') if _check_plotly_dependency() else None,
            'HI': hi,
            'TOTAL_AREA': total_area_km2,
            'STAGE': stage
        }
    
    except Exception as e:
        feedback.reportError(f"Error in generate_hypsometric_curve: {str(e)}")
        return None

def generate_hypsometric_curve(dem_layer, basin_layer, output_folder, feedback, method=0):
    if method == 0:
        return generate_hypsometric_curve_qgis(dem_layer, basin_layer, output_folder, feedback)
    else:
        return generate_hypsometric_curve_pixel(dem_layer, basin_layer, output_folder, feedback)

def create_static_image(relative_area, relative_height, curve_color, point_color, output_folder, hi, total_area_km2, stage):
    width, height = 700, 700
    
    try:
        if hasattr(QImage, 'Format'):
            image_format = QImage.Format.Format_ARGB32_Premultiplied
        else:
            image_format = QImage.Format_ARGB32_Premultiplied
    except AttributeError:
        try:
            image_format = QImage.Format_ARGB32_Premultiplied
        except:
            image_format = 5
    
    image = QImage(QSize(width, height), image_format)
    
    try:
        if hasattr(Qt, 'GlobalColor'):
            white_color = Qt.GlobalColor.white
            black_color = Qt.GlobalColor.black
            solid_line = Qt.PenStyle.SolidLine
            dash_line = Qt.PenStyle.DashLine
            dot_line = Qt.PenStyle.DotLine
            dash_dot_line = Qt.PenStyle.DashDotLine
            align_center = Qt.AlignmentFlag.AlignCenter
            align_right = Qt.AlignmentFlag.AlignRight
            align_vcenter = Qt.AlignmentFlag.AlignVCenter
        else:
            white_color = Qt.white
            black_color = Qt.black
            solid_line = Qt.SolidLine
            dash_line = Qt.DashLine
            dot_line = Qt.DotLine
            dash_dot_line = Qt.DashDotLine
            align_center = Qt.AlignCenter
            align_right = Qt.AlignRight
            align_vcenter = Qt.AlignVCenter
    except:
        white_color = Qt.white
        black_color = Qt.black
        solid_line = Qt.SolidLine
        dash_line = Qt.DashLine
        dot_line = Qt.DotLine
        dash_dot_line = Qt.DashDotLine
        align_center = Qt.AlignCenter
        align_right = Qt.AlignRight
        align_vcenter = Qt.AlignVCenter
    
    image.fill(white_color)
    
    painter = QPainter()
    painter.begin(image)
    
    try:
        if hasattr(QPainter, 'RenderHint'):
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        else:
            painter.setRenderHint(QPainter.Antialiasing)
    except:
        painter.setRenderHint(QPainter.Antialiasing)
    
    margin_left = 80
    margin_right = 80
    margin_top = 80
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    
    painter.setPen(QPen(black_color, 2))
    painter.drawLine(margin_left, height - margin_bottom, width - margin_right, height - margin_bottom)
    painter.drawLine(margin_left, height - margin_bottom, margin_left, margin_top)
    
    painter.setFont(QFont('Arial', 9))
    for i in range(11):
        x = int(margin_left + (i/10) * plot_width)
        y = height - margin_bottom
        painter.drawLine(x, y, x, y + 5)
        painter.drawText(x - 15, y + 15, 30, 20, align_center, f"{i/10:.1f}")
    
    for i in range(11):
        x = margin_left
        y = int(height - margin_bottom - i * plot_height / 10)
        painter.drawLine(x - 5, y, x, y)
        painter.drawText(x - 60, y - 10, 50, 20, align_right | align_vcenter, f"{i/10:.1f}")
    
    def draw_curve(func, color, style=solid_line):
        painter.setPen(QPen(color, 1, style))
        polygon = QPolygonF()
        for i in range(101):
            x_val = i / 100
            y_val = func(x_val)
            px = margin_left + x_val * plot_width
            py = height - margin_bottom - y_val * plot_height
            polygon.append(QPointF(px, py))
        painter.drawPolyline(polygon)
    
    draw_curve(lambda x: 1 - x**2, QColor('#FF6B6B'), dash_line)
    draw_curve(lambda x: 1 - x, QColor('#4ECDC4'), dot_line)
    draw_curve(lambda x: (1 - x)**2, QColor('#2ECC71'), dash_dot_line)
    
    painter.setPen(QPen(curve_color, 3))
    polygon_curve = QPolygonF()
    for i in range(len(relative_area)):
        x_val = margin_left + relative_area[i] * plot_width
        y_val = height - margin_bottom - relative_height[i] * plot_height
        polygon_curve.append(QPointF(x_val, y_val))
    painter.drawPolyline(polygon_curve)
    
    painter.setPen(QPen(black_color))
    
    try:
        if hasattr(QFont, 'Weight'):
            painter.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        else:
            painter.setFont(QFont('Arial', 12, QFont.Bold))
    except:
        painter.setFont(QFont('Arial', 12, QFont.Bold))
    
    painter.drawText(
        QRectF(width/2 - 300, 10, 600, 30), 
        align_center,
        f"Hypsometric Curve (HI = {hi:.3f} - {stage}, Area = {total_area_km2:.2f} km²)"
    )
    
    painter.setFont(QFont('Arial', 10))
    painter.drawText(
        QRectF(width/2 - 100, height - 25, 200, 20), 
        align_center,
        "Relative area (a/A)"
    )
    
    painter.save()
    painter.translate(20, height/2)
    painter.rotate(-90)
    painter.drawText(
        QRectF(-100, -30, 200, 60), 
        align_center,
        "Relative height (h/H)"
    )
    painter.restore()
    
    painter.end()
    
    output_path = os.path.join(output_folder, 'hypsometric_curve.png')
    image.save(output_path)

def create_interactive_html(relative_area, relative_height, area, elevation, curve_color, point_color, output_folder, hi, total_area_km2, stage, method):
    # Check if plotly is available (lazy check)
    if not _check_plotly_dependency():
        return
    
    # Import plotly now that we know it's available
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3])
    
    x_ref = [i/100 for i in range(101)]
    y_young = [1 - x**2 for x in x_ref]
    y_mature = [1 - x for x in x_ref]
    y_old = [(1 - x)**2 for x in x_ref]
    
    fig.add_trace(go.Scatter(
        x=x_ref, y=y_young, 
        mode='lines', 
        name='Young stage', 
        line=dict(color='#FF6B6B', width=1, dash='dash')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x_ref, y=y_mature, 
        mode='lines', 
        name='Mature stage', 
        line=dict(color='#4ECDC4', width=1, dash='dot')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x_ref, y=y_old, 
        mode='lines', 
        name='Old stage', 
        line=dict(color='#2ECC71', width=1, dash='dashdot')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=relative_area, 
        y=relative_height, 
        mode='lines', 
        name='Hypsometric Curve',
        line=dict(color=f'rgb{curve_color.getRgb()[:3]}', width=3),
        hovertemplate='a/A: %{x:.3f}<br>h/H: %{y:.3f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=relative_area, 
        y=relative_height, 
        mode='markers', 
        name='Data points',
        marker=dict(color=f'rgb{point_color.getRgb()[:3]}', size=8),
        hovertemplate='a/A: %{x:.3f}<br>h/H: %{y:.3f}<extra></extra>',
        visible='legendonly'
    ), row=1, col=1) if method == 0 else fig.add_trace(go.Scatter(
        x=[relative_area[i] for i in range(0, len(relative_area), len(relative_area)//20)],
        y=[relative_height[i] for i in range(0, len(relative_height), len(relative_height)//20)],
        mode='markers', 
        name='Data points',
        marker=dict(color=f'rgb{point_color.getRgb()[:3]}', size=8),
        hovertemplate='a/A: %{x:.3f}<br>h/H: %{y:.3f}<extra></extra>',
        visible='legendonly'
    ), row=1, col=1)
    
    if method == 0:
        areas_diff = []
        percentages = []
        total_area_val = max(area)
        for i in range(len(area)):
            if i == 0:
                diff = area[i]
            else:
                diff = area[i] - area[i-1]
            areas_diff.append(diff)
            percentages.append((diff/total_area_val) * 100)

        fig.add_trace(go.Bar(
            y=elevation,
            x=[a/1e6 for a in areas_diff],
            orientation='h',
            name='Elevation Distribution',
            marker=dict(color=f'rgb{curve_color.getRgb()[:3]}'),
            hovertemplate='Area: %{x:.2f} km² (%{customdata:.1f}%)<br>Elevation: %{y:.1f} m<extra></extra>',
            customdata=percentages
        ), row=1, col=2)
    else:
        areas_diff = []
        percentages = []
        total_area_val = max(area)
        
        min_elevation = min(elevation)
        max_elevation = max(elevation)
        elevation_bins = 15
        
        bin_width = (max_elevation - min_elevation) / elevation_bins
        
        binned_elevation = []
        binned_area = []
        binned_percentages = []
        
        for i in range(elevation_bins):
            bin_min = min_elevation + i * bin_width
            bin_max = min_elevation + (i + 1) * bin_width
            bin_center = (bin_min + bin_max) / 2
            
            bin_area = 0
            count = 0
            
            for j, elev in enumerate(elevation):
                if bin_min <= elev < bin_max or (i == elevation_bins - 1 and elev == bin_max):
                    if j == 0:
                        bin_area += area[j]
                    else:
                        bin_area += area[j] - area[j-1]
                    count += 1
            
            if count > 0:
                binned_elevation.append(bin_center)
                binned_area.append(bin_area)
                binned_percentages.append((bin_area / total_area_val) * 100)
        
        fig.add_trace(go.Bar(
            y=binned_elevation,
            x=[a/1e6 for a in binned_area],
            orientation='h',
            name='Elevation Distribution',
            marker=dict(color=f'rgb{curve_color.getRgb()[:3]}'),
            hovertemplate='Area: %{x:.2f} km² (%{customdata:.1f}%)<br>Elevation: %{y:.1f} m<extra></extra>',
            customdata=binned_percentages
        ), row=1, col=2)
    
    fig.update_layout(
        title_text=f"Hypsometric Curve (HI = {hi:.3f} - {stage}, Area = {total_area_km2:.2f} km²)",
        showlegend=True,
        plot_bgcolor='white'
    )
    
    fig.update_xaxes(
        title_text="Relative area (a/A)",
        range=[0, 1],
        row=1, col=1,
        showgrid=False,
        zeroline=True,
        showline=True,
        linewidth=1,
        linecolor='black'
    )
    
    fig.update_yaxes(
        title_text="Relative height (h/H)",
        range=[0, 1],
        row=1, col=1,
        showgrid=False,
        zeroline=True,
        showline=True,
        linewidth=1,
        linecolor='black'
    )
    
    fig.update_xaxes(
        title_text="Area (km²)",
        row=1, col=2,
        showgrid=False,
        zeroline=True,
        showline=True,
        linewidth=1,
        linecolor='black'
    )
    
    fig.update_yaxes(
        title_text="Elevation (m)",
        row=1, col=2,
        showgrid=False,
        zeroline=True,
        showline=True,
        linewidth=1,
        linecolor='black'
    )
    
    html_content = fig.to_html(include_plotlyjs='cdn')
    
    output_path = os.path.join(output_folder, 'hypsometric_curve_interactive.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)