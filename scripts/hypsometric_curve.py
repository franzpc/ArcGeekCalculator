"""
Hypsometric Curve Generator
---------------------------
This script generates hypsometric curves and calculates morphometric indices for watershed analysis.

References:
-----------
- Strahler, A.N. (1952). Hypsometric (area-altitude) analysis of erosional topography.
- Harlin, J.M. (1978). Statistical moments of the hypsometric curve and its density function.

Classification Criteria based on HI:
- HI ≥ 0.60: Young stage
- 0.35 ≤ HI < 0.60: Mature stage
- HI < 0.35: Old stage
"""

from qgis.core import *
from qgis.analysis import QgsZonalStatistics
from qgis.PyQt.QtGui import QColor, QPainter, QFont, QImage, QPen, QBrush, QPolygonF
from qgis.PyQt.QtCore import QSize, Qt, QPointF, QRectF
import processing
import os
import csv
import platform
import math

# Try to import plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

def get_basin_stage(hi):
    """
    Determine the watershed development stage based on the HI value.
    """
    if hi >= 0.60:
        return "Young stage"
    elif 0.35 <= hi < 0.60:
        return "Mature stage"
    else:
        return "Old stage"

def calculate_hypsometric_integral(elevation, area):
    """
    Calculate the hypsometric integral using the trapezoidal method (Harlin, 1978).
    """
    # Sort data by elevation in descending order
    sorted_pairs = sorted(zip(elevation, area), reverse=True)
    elevation_sorted, area_sorted = zip(*sorted_pairs)
    
    min_elev = elevation_sorted[-1]
    max_elev = elevation_sorted[0]
    
    # Calculate relative height (h/H)
    relative_height = [(e - min_elev) / (max_elev - min_elev) for e in elevation_sorted]
    
    # Calculate relative area (a/A)
    total_area = area_sorted[0]
    relative_area = [a / total_area for a in area_sorted]
    
    # Integration using the trapezoidal rule
    area_accum = 0
    for i in range(len(relative_area) - 1):
        a1 = relative_area[i]
        a2 = relative_area[i + 1]
        h1 = relative_height[i]
        h2 = relative_height[i + 1]
        area_accum += abs(a2 - a1) * ((h1 + h2) / 2)
    
    return area_accum

def generate_hypsometric_curve(dem_layer, basin_layer, output_folder, feedback):
    """
    Generate the hypsometric curve from a DEM and a basin layer.
    """
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

        # Use the most recent CSV file
        csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_folder, x)), reverse=True)
        csv_file = os.path.join(output_folder, csv_files[0])
        feedback.pushInfo(f"Using most recent histogram file: {csv_file}")

        try:
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                data = list(reader)

            area = [float(row[0]) for row in data]
            elevation = [float(row[1]) for row in data]

            # Total area in km² (convert from m²)
            total_area = max(area)
            total_area_km2 = total_area / 1e6

            # Calculate HI and determine stage
            hi = calculate_hypsometric_integral(elevation, area)
            stage = get_basin_stage(hi)
            feedback.pushInfo(f"Calculated Hypsometric Integral: {hi:.3f} ({stage})")

            # Calculate relative height and area
            min_elev, max_elev = min(elevation), max(elevation)
            relative_height = [(e - min_elev) / (max_elev - min_elev) for e in elevation]
            relative_area = [a / total_area for a in area]

            # Reverse the order of data for the hypsometric curve
            relative_height = relative_height[::-1]
            relative_area = [1 - a for a in relative_area[::-1]]

            # Save processed data to CSV
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

            # Define colors for the curve
            curve_color = QColor(19, 138, 249)  # Blue
            point_color = QColor(203, 67, 53)   # Red

            # Create static image
            create_static_image(relative_area, relative_height, curve_color, point_color, output_folder, hi, total_area_km2, stage)

            # Create interactive HTML if Plotly is available
            if PLOTLY_AVAILABLE:
                create_interactive_html(relative_area, relative_height, area, elevation, curve_color, point_color, output_folder, hi, total_area_km2, stage)
                feedback.pushInfo("Interactive hypsometric curve (HTML) generated.")
            else:
                feedback.pushInfo("Plotly is not installed; interactive version not generated.")

            feedback.pushInfo(f"Processed data saved to: {processed_csv}")
            feedback.pushInfo(f"Hypsometric curve analysis completed. Results in: {output_folder}")
            
            return {
                'CSV': processed_csv,
                'PNG': os.path.join(output_folder, 'hypsometric_curve.png'),
                'HTML': os.path.join(output_folder, 'hypsometric_curve_interactive.html') if PLOTLY_AVAILABLE else None,
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

def create_static_image(relative_area, relative_height, curve_color, point_color, output_folder, hi, total_area_km2, stage):
    """
    Create a static image of the hypsometric curve.
    """
    width, height = 800, 600
    image = QImage(QSize(width, height), QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.white)

    painter = QPainter()
    painter.begin(image)
    painter.setRenderHint(QPainter.Antialiasing)

    # Define margins
    margin_left = 80
    margin_right = 50
    margin_top = 50
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    # Draw axes
    painter.setPen(QPen(Qt.black, 2))
    painter.drawLine(margin_left, height - margin_bottom, width - margin_right, height - margin_bottom)  # X-axis
    painter.drawLine(margin_left, height - margin_bottom, margin_left, margin_top)  # Y-axis

    # Draw axis labels
    painter.setFont(QFont('Arial', 9))
    # X-axis labels (0 to 1)
    for i in range(11):
        x = int(margin_left + (i/10) * plot_width)
        y = height - margin_bottom
        painter.drawLine(x, y, x, y + 5)
        painter.drawText(x - 15, y + 15, 30, 20, Qt.AlignCenter, f"{i/10:.1f}")
    
    # Y-axis labels (0 to 1)
    for i in range(11):
        x = margin_left
        y = int(height - margin_bottom - i * plot_height / 10)
        painter.drawLine(x - 5, y, x, y)
        painter.drawText(x - 60, y - 10, 50, 20, Qt.AlignRight | Qt.AlignVCenter, f"{i/10:.1f}")

    # Helper function to draw reference curves
    def draw_curve(func, color, style=Qt.SolidLine):
        painter.setPen(QPen(color, 1, style))
        polygon = QPolygonF()
        for i in range(101):
            x_val = i / 100
            y_val = func(x_val)
            px = margin_left + x_val * plot_width
            py = height - margin_bottom - y_val * plot_height
            polygon.append(QPointF(px, py))
        painter.drawPolyline(polygon)

    # Draw reference curves (Young, Mature, Old)
    draw_curve(lambda x: 1 - x**2, QColor('#FF6B6B'), Qt.DashLine)    # Young stage
    draw_curve(lambda x: 1 - x, QColor('#4ECDC4'), Qt.DotLine)        # Mature stage
    draw_curve(lambda x: (1 - x)**2, QColor('#2ECC71'), Qt.DashDotLine)  # Old stage

    # Draw hypsometric curve
    painter.setPen(QPen(curve_color, 3))
    polygon_curve = QPolygonF()
    for i in range(len(relative_area)):
        x_val = margin_left + relative_area[i] * plot_width
        y_val = height - margin_bottom - relative_height[i] * plot_height
        polygon_curve.append(QPointF(x_val, y_val))
    painter.drawPolyline(polygon_curve)

    # Draw data points
    painter.setPen(QPen(Qt.black))
    painter.setBrush(QBrush(point_color))
    for i in range(len(relative_area)):
        painter.drawEllipse(QPointF(
            margin_left + relative_area[i] * plot_width,
            height - margin_bottom - relative_height[i] * plot_height
        ), 4, 4)

    # Add title with HI, stage, and total area
    painter.setPen(QPen(Qt.black))
    painter.setFont(QFont('Arial', 12, QFont.Bold))
    painter.drawText(
        QRectF(width/2 - 300, 10, 600, 30), 
        Qt.AlignCenter,
        f"Hypsometric Curve (HI = {hi:.3f} - {stage}, Area = {total_area_km2:.2f} km²)"
    )

    # Add X-axis label
    painter.setFont(QFont('Arial', 10))
    painter.drawText(
        QRectF(width/2 - 100, height - 25, 200, 20), 
        Qt.AlignCenter,
        "Relative area (a/A)"
    )
    
    # Add Y-axis label
    painter.save()
    painter.translate(20, height/2)
    painter.rotate(-90)
    painter.drawText(
        QRectF(-100, -30, 200, 60), 
        Qt.AlignCenter,
        "Relative height (h/H)"
    )
    painter.restore()

    painter.end()

    output_path = os.path.join(output_folder, 'hypsometric_curve.png')
    image.save(output_path)

def create_interactive_html(relative_area, relative_height, area, elevation, curve_color, point_color, output_folder, hi, total_area_km2, stage):
    """
    Create an interactive HTML version of the hypsometric curve using Plotly.
    """
    if not PLOTLY_AVAILABLE:
        return

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3])

    # Reference curves
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

    # Hypsometric curve
    fig.add_trace(go.Scatter(
        x=relative_area, 
        y=relative_height, 
        mode='lines', 
        name='Hypsometric Curve',
        line=dict(color=f'rgb{curve_color.getRgb()[:3]}', width=3),
        hovertemplate='a/A: %{x:.3f}<br>h/H: %{y:.3f}<extra></extra>'
    ), row=1, col=1)

    # Data points
    fig.add_trace(go.Scatter(
        x=relative_area, 
        y=relative_height, 
        mode='markers', 
        name='Data points',
        marker=dict(color=f'rgb{point_color.getRgb()[:3]}', size=8),
        hovertemplate='a/A: %{x:.3f}<br>h/H: %{y:.3f}<extra></extra>'
    ), row=1, col=1)

    # Calculate area differences and percentages for histogram
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
        x=[a/1e6 for a in areas_diff],  # Convert from m² to km²
        orientation='h',
        name='Elevation Distribution',
        marker=dict(color=f'rgb{curve_color.getRgb()[:3]}'),
        hovertemplate='Area: %{x:.2f} km² (%{customdata:.1f}%)<br>Elevation: %{y:.1f} m<extra></extra>',
        customdata=percentages
    ), row=1, col=2)

    # Update layout
    fig.update_layout(
        title_text=f"Hypsometric Curve (HI = {hi:.3f} - {stage}, Area = {total_area_km2:.2f} km²)",
        showlegend=True,
        plot_bgcolor='white'
    )

    # Update axes for the hypsometric curve
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

    # Update axes for the elevation distribution histogram
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

    # Save interactive HTML
    output_path = os.path.join(output_folder, 'hypsometric_curve_interactive.html')
    fig.write_html(output_path)
