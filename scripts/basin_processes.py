import math
from qgis.core import QgsGeometry, QgsPointXY, QgsRasterBandStats, QgsFeature, QgsField, QgsVectorLayer
from qgis.analysis import QgsRasterCalculatorEntry, QgsRasterCalculator
from qgis.PyQt.QtCore import QVariant

def neighbor_average_interpolation(dem_layer, point):
    x_res = dem_layer.rasterUnitsPerPixelX()
    y_res = dem_layer.rasterUnitsPerPixelY()

    col = int((point.x() - dem_layer.extent().xMinimum()) / x_res)
    row = int((dem_layer.extent().yMaximum() - point.y()) / y_res)

    values = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            value = dem_layer.dataProvider().sample(QgsPointXY(dem_layer.extent().xMinimum() + (col+i)*x_res, 
                                                               dem_layer.extent().yMaximum() - (row+j)*y_res), 1)[0]
            if not math.isnan(value):
                values.append(value)

    if values:
        return sum(values) / len(values)
    else:
        return None

def calculate_parameters(basin_source, streams_source, dem_layer, pour_point, stream_order_field, mean_slope_degrees, feedback):
    # Check if DEM layer is valid
    if not dem_layer or not dem_layer.isValid():
        feedback.reportError("Invalid DEM layer. Cannot proceed with calculations.")
        return None

    basin_area = sum([f.geometry().area() for f in basin_source.getFeatures()]) / 1e6  # m² to km²
    perimeter = sum([f.geometry().length() for f in basin_source.getFeatures()]) / 1e3  # m to km

    basin_length = calculate_basin_length(basin_source, QgsPointXY(pour_point))
    basin_width = basin_area / basin_length if basin_length != 0 else 0

    total_stream_length = sum([f.geometry().length() for f in streams_source.getFeatures()]) / 1e3  # m to km
    main_channel_length = sum([f.geometry().length() for f in streams_source.getFeatures() if f[stream_order_field] == max([f[stream_order_field] for f in streams_source.getFeatures()])]) / 1e3

    stream_order = calculate_stream_order(streams_source, stream_order_field)
    num_streams_first_order = stream_order.count(1)
    total_stream_number = len(stream_order)

    drainage_density = total_stream_length / basin_area if basin_area > 0 else 0

    if num_streams_first_order != 0:
        bifurcation_ratio = total_stream_number / num_streams_first_order
    else:
        bifurcation_ratio = None

    try:
        dem_stats = dem_layer.dataProvider().bandStatistics(1, QgsRasterBandStats.All)
        max_elevation = dem_stats.maximumValue
        min_elevation = dem_stats.minimumValue
        relief = max_elevation - min_elevation
        mean_elevation = dem_stats.mean 
    except Exception as e:
        feedback.reportError(f"Error calculating DEM statistics: {str(e)}")
        max_elevation = min_elevation = relief = None

    mean_stream_length = total_stream_length / total_stream_number if total_stream_number != 0 else None
    stream_frequency = total_stream_number / basin_area if basin_area > 0 else 0
    drainage_intensity = stream_frequency / drainage_density if drainage_density != 0 else None
    length_of_overland_flow = 1 / (2 * drainage_density) if drainage_density != 0 else None

    # mean_elevation = (max_elevation + min_elevation) / 2 if max_elevation is not None and min_elevation is not None else None
    mean_elevation = dem_stats.mean
    mean_slope_radians = math.radians(mean_slope_degrees)
    mean_slope_m_per_m = math.tan(mean_slope_radians)
    mean_slope_percent = math.tan(math.radians(mean_slope_degrees)) * 100

    # Get all segments of the main channel
    main_channel_segments = [f.geometry() for f in streams_source.getFeatures() if f[stream_order_field] == max([f[stream_order_field] for f in streams_source.getFeatures()])]

    # Merge all segments into a single line
    main_channel = QgsGeometry.unaryUnion(main_channel_segments)

    # Ensure the result is a single line
    if main_channel.isMultipart():
        main_channel = main_channel.mergeLines()

    # Get the start and end points
    vertices = main_channel.asPolyline()
    upstream_point = vertices[0]
    downstream_point = vertices[-1]

    # Print debug information
    # feedback.pushInfo(f"Upstream Point: {upstream_point.x()}, {upstream_point.y()}")
    # feedback.pushInfo(f"Downstream Point: {downstream_point.x()}, {downstream_point.y()}")

    # Check if points are within DEM extent
    dem_extent = dem_layer.extent()
    # feedback.pushInfo(f"DEM Extent: {dem_extent.toString()}")

    # Calculate elevations using neighbor interpolation
    upstream_elevation = neighbor_average_interpolation(dem_layer, upstream_point)
    downstream_elevation = neighbor_average_interpolation(dem_layer, downstream_point)

    # Determine which is actually the start point (highest) and end point (lowest)
    if upstream_elevation is not None and downstream_elevation is not None:
        if upstream_elevation > downstream_elevation:
            start_point = upstream_point
            end_point = downstream_point
            start_elevation = upstream_elevation
            end_elevation = downstream_elevation
        else:
            start_point = downstream_point
            end_point = upstream_point
            start_elevation = downstream_elevation
            end_elevation = upstream_elevation

        # Print elevation values for verification
        feedback.pushInfo(f"Start Elevation (highest point): {start_elevation}")
        feedback.pushInfo(f"End Elevation (lowest point): {end_elevation}")

        # Calculate slope only if both elevations are valid
        slope_s = (start_elevation - end_elevation) / (main_channel_length * 1000)
        slope_percent = slope_s * 100
        feedback.pushInfo(f"Slope: {slope_percent}%")
    else:
        feedback.pushInfo("Warning: Unable to calculate slope due to invalid elevation values")
        start_elevation = end_elevation = slope_s = slope_percent = None
    middle_distance = main_channel_length / (basin_area ** 0.5)


    channel_compensated_slope_results = calculate_channel_compensated_slope(main_channel, dem_layer)


    # Time of concentration calculations

    # Kerby method needs to define a roughness coefficient 'n' (now 0.3)
    time_of_concentration_kerby = (0.606 * ((basin_length * 1000) * 0.3 / math.sqrt(slope_s)) ** 0.467) / 60 if slope_s and slope_s > 0 else None

    time_of_concentration_kirpich = (0.0195 * ((main_channel_length * 1000) ** 0.77) / (slope_s ** 0.385)) / 60 if slope_s and slope_s > 0 else None
    # time_of_concentration_kerby = (0.828 * (basin_length * 1000) ** 0.467 / (slope_s ** 0.235)) / 60 if slope_s and slope_s > 0 else None
    time_of_concentration_giandotti = ((4 * math.sqrt(basin_area) + 1.5 * main_channel_length) / (0.8 * math.sqrt(relief))) if relief > 0 else None
    time_of_concentration_temez = 0.3 * (main_channel_length * (slope_s ** 0.25)) ** 0.76 if slope_s and slope_s > 0 else None
    time_of_concentration_usda = (3.3 * basin_length) / math.sqrt(mean_slope_percent) if mean_slope_percent > 0 else None
    time_of_concentration_ventura_heras = middle_distance * (basin_area ** 0.5 / slope_percent) if slope_percent and slope_percent > 0 else None
    time_of_concentration_passini = middle_distance * ((basin_area * main_channel_length) ** (1/3)) / (slope_percent ** 0.5) if slope_percent and slope_percent > 0 else None

    time_of_concentration_california_culverts = 0.0195 * (main_channel_length ** 3 / relief) ** 0.385 if relief > 0 else None
    time_of_concentration_bransby_williams = 0.243 * (main_channel_length / (basin_area ** 0.1 * (slope_s * 1000) ** 0.2)) if slope_s and slope_s > 0 else None
    time_of_concentration_johnstone_cross = 2.6 * (main_channel_length / (slope_s * 1000) ** 0.5) ** 0.5 if slope_s and slope_s > 0 else None
    time_of_concentration_clark = 0.335 * (basin_area / (slope_s * 1000) ** 0.5) ** 0.593 if slope_s and slope_s > 0 else None


    form_factor = basin_area / (basin_length ** 2)
    elongation_ratio = (2 * math.sqrt(basin_area / math.pi)) / basin_length
    circularity_ratio = (4 * math.pi * basin_area) / (perimeter ** 2)
    compactness_coefficient = 0.2821 * perimeter / math.sqrt(basin_area)
    ruggedness_number = drainage_density * relief / 1000  # Convert relief to km
    infiltration_number = drainage_density * stream_frequency
    drainage_texture = total_stream_number / perimeter
    fitness_ratio = main_channel_length / perimeter
    asymmetry_factor = calculate_asymmetry_factor(basin_source, QgsPointXY(pour_point))
    orographic_coefficient = calculate_orographic_coefficient(relief, basin_area)

    # New parameters
    relief_ratio = relief / basin_length
    hortons_form_factor = basin_area / (basin_length ** 2)
    schumms_elongation_ratio = (2 * math.sqrt(basin_area / math.pi)) / basin_length
    main_channel_gradient = relief / main_channel_length
    main_channel_sinuosity = main_channel_length / basin_length
    massivity_index = mean_elevation / basin_area
    texture_ratio = total_stream_number / perimeter
    junction_density = total_stream_number / basin_area
    storage_coefficient = 0.3025 * (basin_length ** 2) / relief  # This is a simplified formula, might need adjustment

    return {
        "Basin Area (A)": {"value": basin_area, "unit": "km²", "interpretation": get_basin_area_interpretation(basin_area)},
        "Perimeter (P)": {"value": perimeter, "unit": "km", "interpretation": "Basin perimeter"},
        "Basin Length (Lb)": {"value": basin_length, "unit": "km", "interpretation": "Basin length"},
        "Basin Width (B)": {"value": basin_width, "unit": "km", "interpretation": "Basin width"},
        "Relief (H)": {"value": relief, "unit": "m", "interpretation": get_relief_interpretation(relief)},
        "Mean Elevation": {"value": mean_elevation, "unit": "m a.s.l.", "interpretation": "Average elevation of the basin"},
        "Minimum Elevation": {"value": min_elevation, "unit": "m a.s.l.", "interpretation": "Minimum elevation of the basin"},
        "Maximum Elevation": {"value": max_elevation, "unit": "m a.s.l.", "interpretation": "Maximum elevation of the basin"},
        "Start Elevation (Main Channel)": {"value": start_elevation, "unit": "m a.s.l.", "interpretation": "Elevation at the start of the main channel"},
        "End Elevation (Main Channel)": {"value": end_elevation, "unit": "m a.s.l.", "interpretation": "Elevation at the end of the main channel"},
        "Mean slope of the Basin (degrees)": {"value": mean_slope_degrees, "unit": "degrees", "interpretation": get_mean_slope_interpretation(mean_slope_degrees)},
        "Mean slope of the Basin (percent)": {"value": mean_slope_m_per_m * 100, "unit": "%", "interpretation": get_mean_slope_interpretation(mean_slope_m_per_m * 100, percent=True)},
        "Main Channel Slope (Endpoints)": {"value": slope_percent, "unit": "%", "interpretation": get_main_channel_slope_interpretation(slope_percent)},


        "Compensated Channel Slope": channel_compensated_slope_results.get("Compensated Channel Slope", {"value": None, "unit": "m/m", "interpretation": "Unable to calculate"}),
        "Compensated Channel Slope (%)": channel_compensated_slope_results.get("Compensated Channel Slope (%)", {"value": None, "unit": "%", "interpretation": "Unable to calculate"}),

        "Drainage Density (Dd)": {"value": drainage_density, "unit": "km/km²", "interpretation": get_drainage_density_interpretation(drainage_density)},
        "Stream Frequency (Fs)": {"value": stream_frequency, "unit": "streams/km²", "interpretation": get_stream_frequency_interpretation(stream_frequency)},
        "Elongation Ratio (Re)": {"value": elongation_ratio, "unit": "", "interpretation": get_elongation_ratio_interpretation(elongation_ratio)},
        "Circularity Ratio (Rc)": {"value": circularity_ratio, "unit": "", "interpretation": get_circularity_ratio_interpretation(circularity_ratio)},
        "Compactness Coefficient of Gravelius (Kc)": {"value": compactness_coefficient, "unit": "", "interpretation": get_compactness_coefficient_interpretation(compactness_coefficient)},
        "Form Factor (Ff)": {"value": form_factor, "unit": "", "interpretation": get_form_factor_interpretation(form_factor)},
        "Horton's Form Factor": {"value": hortons_form_factor, "unit": "", "interpretation": get_hortons_form_factor_interpretation(hortons_form_factor)},
        "Schumm's Elongation Ratio": {"value": schumms_elongation_ratio, "unit": "", "interpretation": get_schumms_elongation_ratio_interpretation(schumms_elongation_ratio)},
        "Length of Overland Flow (Lo)": {"value": length_of_overland_flow, "unit": "km", "interpretation": get_length_of_overland_flow_interpretation(length_of_overland_flow)},
        "Constant of Channel Maintenance (C)": {"value": 1/drainage_density if drainage_density != 0 else None, "unit": "km²/km", "interpretation": get_constant_channel_maintenance_interpretation(1/drainage_density if drainage_density != 0 else None)},
        "Ruggedness Number (Rn)": {"value": ruggedness_number, "unit": "", "interpretation": get_ruggedness_number_interpretation(ruggedness_number)},

        "Time of Concentration - Kirpich (Tc)": {"value": time_of_concentration_kirpich, "unit": "hours", "interpretation": get_time_of_concentration_interpretation(time_of_concentration_kirpich)},
        "Time of Concentration - Kerby (Tc)": {"value": time_of_concentration_kerby, "unit": "hours", "interpretation": get_time_of_concentration_interpretation(time_of_concentration_kerby)},
        "Time of Concentration - Giandotti (Tc)": {"value": time_of_concentration_giandotti, "unit": "hours", "interpretation": get_time_of_concentration_interpretation(time_of_concentration_giandotti)},
        "Time of Concentration - Témez (Tc)": {"value": time_of_concentration_temez, "unit": "hours", "interpretation": get_time_of_concentration_interpretation(time_of_concentration_temez)},
        "Time of Concentration - USDA (Tc)": {"value": time_of_concentration_usda, "unit": "hours", "interpretation": get_time_of_concentration_interpretation(time_of_concentration_usda)},
        "Time of Concentration - Passini (Tc)": {"value": time_of_concentration_passini, "unit": "hours", "interpretation": get_time_of_concentration_interpretation(time_of_concentration_passini)},
        "Time of Concentration - Ventura-Heras (Tc)": {"value": time_of_concentration_ventura_heras, "unit": "hours", "interpretation": get_time_of_concentration_interpretation(time_of_concentration_ventura_heras)},

         "Time of Concentration - Kirpich (Tc)": {"value": time_of_concentration_kirpich, "unit": "hours", "interpretation": get_time_of_concentration_interpretation(time_of_concentration_kirpich)},
        "Time of Concentration - Témez (Tc)": {"value": time_of_concentration_temez, "unit": "hours", "interpretation": get_time_of_concentration_interpretation(time_of_concentration_temez)},
        "Time of Concentration - Giandotti (Tc)": {"value": time_of_concentration_giandotti, "unit": "hours", "interpretation": get_time_of_concentration_interpretation(time_of_concentration_giandotti)},
        # "Time of Concentration - California Culverts Practice (Tc)": {"value": time_of_concentration_california_culverts, "unit": "hours", "interpretation": get_time_of_concentration_interpretation(time_of_concentration_california_culverts)},
        "Time of Concentration - Bransby-Williams (Tc)": {"value": time_of_concentration_bransby_williams, "unit": "hours", "interpretation": get_time_of_concentration_interpretation(time_of_concentration_bransby_williams)},
        "Time of Concentration - Johnstone-Cross (Tc)": {"value": time_of_concentration_johnstone_cross, "unit": "hours", "interpretation": get_time_of_concentration_interpretation(time_of_concentration_johnstone_cross)},
        "Time of Concentration - Clark (Tc)": {"value": time_of_concentration_clark, "unit": "hours", "interpretation": get_time_of_concentration_interpretation(time_of_concentration_clark)},
        "Time of Concentration - Kerby (Tc)": {"value": time_of_concentration_kerby, "unit": "hours", "interpretation": get_time_of_concentration_interpretation(time_of_concentration_kerby)},

        "Bifurcation Ratio (Rb)": {"value": bifurcation_ratio, "unit": "", "interpretation": get_bifurcation_ratio_interpretation(bifurcation_ratio)},
        "Stream Order": {"value": max(stream_order), "unit": "", "interpretation": f"Highest stream order (Strahler): {max(stream_order)}"},
        "Mean Stream Length (Lm)": {"value": mean_stream_length, "unit": "km", "interpretation": "Average length of streams"},
        "Drainage Intensity (Id)": {"value": drainage_intensity, "unit": "", "interpretation": get_drainage_intensity_interpretation(drainage_intensity)},
        "Main Channel Gradient": {"value": main_channel_gradient, "unit": "m/km", "interpretation": get_main_channel_gradient_interpretation(main_channel_gradient)},
        "Main Channel Sinuosity": {"value": main_channel_sinuosity, "unit": "", "interpretation": get_main_channel_sinuosity_interpretation(main_channel_sinuosity)},
        "Main Channel Length (Lc)": {"value": main_channel_length, "unit": "km", "interpretation": "Length of the main channel"},
        "Total Length of Channels (Lt)": {"value": total_stream_length, "unit": "km", "interpretation": "Total length of all channels"},
        "Number of Streams (Nu)": {"value": total_stream_number, "unit": "", "interpretation": "Total number of streams"},
        "Drainage Texture (Dt)": {"value": drainage_texture, "unit": "", "interpretation": get_drainage_texture_interpretation(drainage_texture)},
        "Infiltration Number (If)": {"value": infiltration_number, "unit": "", "interpretation": get_infiltration_number_interpretation(infiltration_number)},
        "Fitness Ratio (Rf)": {"value": fitness_ratio, "unit": "", "interpretation": get_fitness_ratio_interpretation(fitness_ratio)},
        "Asymmetry Factor (Af)": {"value": asymmetry_factor, "unit": "", "interpretation": get_asymmetry_factor_interpretation(asymmetry_factor)},
        "Orographic Coefficient (Oc)": {"value": orographic_coefficient, "unit": "", "interpretation": get_orographic_coefficient_interpretation(orographic_coefficient)},
        "Massivity Index": {"value": massivity_index, "unit": "m/km²", "interpretation": get_massivity_index_interpretation(massivity_index)},
        "Junction Density": {"value": junction_density, "unit": "junctions/km²", "interpretation": get_junction_density_interpretation(junction_density)},
        "Storage Coefficient": {"value": storage_coefficient, "unit": "km", "interpretation": get_storage_coefficient_interpretation(storage_coefficient)}
    }


def calculate_basin_length(basin_source, pour_point):
    basin_geom = [f.geometry() for f in basin_source.getFeatures()][0]
    furthest_point = basin_geom.vertexAt(0)
    max_distance = 0
    for vertex in basin_geom.vertices():
        distance = QgsGeometry.fromPointXY(QgsPointXY(vertex)).distance(QgsGeometry.fromPointXY(pour_point))
        if distance > max_distance:
            max_distance = distance
            furthest_point = vertex
    basin_length = QgsGeometry.fromPointXY(QgsPointXY(furthest_point)).distance(QgsGeometry.fromPointXY(pour_point))
    return basin_length / 1e3  # m to km

def calculate_stream_order(streams_source, stream_order_field):
    stream_order = []
    for f in streams_source.getFeatures():
        if stream_order_field and stream_order_field in f.fields().names():
            stream_order.append(f[stream_order_field])
        else:
            stream_order.append(1)  # Assume all streams are first order if no order field exists
    return stream_order

def calculate_asymmetry_factor(basin_source, pour_point):
    # Implement the calculation for the asymmetry factor
    return 0.5  # Placeholder value

def calculate_orographic_coefficient(relief, basin_area):
    return (relief * basin_area) / 1000  # Dividing by 1000 to get a more manageable number

def linear_regression(x, y):
    """
    Calcular regresión lineal usando mínimos cuadrados de forma nativa
    Retorna la pendiente (slope) de la línea de mejor ajuste
    """
    n = len(x)
    
    # Calcular sumas necesarias
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x_squared = sum(xi * xi for xi in x)
    
    # Fórmula de mínimos cuadrados para la pendiente
    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_x_squared - sum_x * sum_x
    
    # Manejar caso de división por cero
    if denominator == 0:
        return 0
    
    slope = numerator / denominator
    
    return slope

def calculate_channel_compensated_slope(main_channel_geom, dem_layer):
    """
    Calculate the compensated slope of the main channel using linear regression
    
    Parameters:
    main_channel_geom (QgsGeometry): Main channel geometry
    dem_layer (QgsRasterLayer): Digital Elevation Model layer
    
    Returns:
    dict: Compensated channel slope information
    """
    # Obtain points of the main channel
    vertices = main_channel_geom.asPolyline()
    
    # Initialize lists for distances and elevations
    distances = [0]  # Start with 0 distance
    elevations = []
    
    # Interpolate elevations for each point of the channel
    for i, vertex in enumerate(vertices):
        # Get elevation using neighbor interpolation from DEM
        elevation = neighbor_average_interpolation(dem_layer, QgsPointXY(vertex))
        
        # Calculate cumulative distance
        if i > 0:
            # Calculate distance from previous point
            distance = distances[-1] + QgsPointXY(vertex).distance(QgsPointXY(vertices[i-1]))
            distances.append(distance)
        
        # Save elevation if valid
        if elevation is not None:
            elevations.append(elevation)
    
    # Truncate distances to match elevations length for safety
    distances = distances[:len(elevations)]
    
    # Check if we have sufficient data for calculation
    if len(distances) < 2 or len(elevations) < 2:
        return {
            "Compensated Channel Slope": {"value": None, "unit": "m/m", "interpretation": "Insufficient data to calculate slope"},
            "Compensated Channel Slope (%)": {"value": None, "unit": "%", "interpretation": "Insufficient data to calculate slope"}
        }
    
    # Calculate the slope using linear regression
    # We use negative slope because elevation decreases as distance increases
    compensated_slope = -linear_regression(distances, elevations)
    
    # Prepare return dictionary with multiple representations of the slope
    return {
        "Compensated Channel Slope": {
            "value": compensated_slope, 
            "unit": "m/m", 
            "interpretation": "Calculated compensated channel slope using linear regression"
        },
        "Compensated Channel Slope (%)": {
            "value": compensated_slope * 100, 
            "unit": "%", 
            "interpretation": get_compensated_channel_slope_interpretation(compensated_slope * 100)
        }
    }



# Interpretations
def get_basin_area_interpretation(area):
    if area < 100:
        return "Small basin"
    elif 100 <= area < 1000:
        return "Medium-sized basin"
    else:
        return "Large basin"

def get_mean_slope_interpretation(mean_slope, percent=False):
    if percent:
        if mean_slope < 5:
            return "Gently sloping"
        elif 5 <= mean_slope < 10:
            return "Moderately steep"
        else:
            return "Steep"
    else:
        if mean_slope < 2.86:
            return "Gently sloping"
        elif 2.86 <= mean_slope < 5.71:
            return "Moderately steep"
        else:
            return "Steep"

def get_form_factor_interpretation(form_factor):
    if form_factor < 0.5:
        return "Elongated shape, low susceptibility to flash floods"
    elif 0.5 <= form_factor < 0.75:
        return "Intermediate shape"
    else:
        return "Circular shape, high susceptibility to flash floods"

def get_elongation_ratio_interpretation(elongation_ratio):
    if elongation_ratio < 0.5:
        return "Elongated shape"
    elif 0.5 <= elongation_ratio < 0.75:
        return "Oval shape"
    else:
        return "Circular shape"

def get_circularity_ratio_interpretation(circularity_ratio):
    if circularity_ratio < 0.4:
        return "Strongly elongated shape"
    elif 0.4 <= circularity_ratio < 0.6:
        return "Elongated shape"
    elif 0.6 <= circularity_ratio < 0.8:
        return "Oval shape"
    else:
        return "Circular shape"

def get_drainage_density_interpretation(drainage_density):
    if drainage_density < 0.5:
        return "Very coarse drainage texture"
    elif 0.5 <= drainage_density < 1.0:
        return "Coarse drainage texture"
    elif 1.0 <= drainage_density < 2.0:
        return "Moderate drainage texture"
    elif 2.0 <= drainage_density < 3.5:
        return "Fine drainage texture"
    else:
        return "Very fine drainage texture"

def get_stream_frequency_interpretation(stream_frequency):
    if stream_frequency < 1:
        return "Low stream frequency"
    elif 1 <= stream_frequency < 3:
        return "Moderate stream frequency"
    elif 3 <= stream_frequency < 5:
        return "High stream frequency"
    else:
        return "Very high stream frequency"

def get_compactness_coefficient_interpretation(compactness_coefficient):
    if compactness_coefficient < 1.25:
        return "Almost circular shape"
    elif 1.25 <= compactness_coefficient < 1.5:
        return "Oval-circular to oval-oblong shape"
    elif 1.5 <= compactness_coefficient < 1.75:
        return "Oval-oblong shape"
    else:
        return "Rectangular-oblong shape"

def get_length_of_overland_flow_interpretation(length_of_overland_flow):
    if length_of_overland_flow < 0.25:
        return "Short overland flow length, indicating high drainage density"
    elif 0.25 <= length_of_overland_flow < 0.5:
        return "Moderate overland flow length"
    else:
        return "Long overland flow length, indicating low drainage density"

def get_constant_channel_maintenance_interpretation(constant_channel_maintenance):
    if constant_channel_maintenance < 0.5:
        return "Low constant of channel maintenance, indicating high drainage density"
    elif 0.5 <= constant_channel_maintenance < 1:
        return "Moderate constant of channel maintenance"
    else:
        return "High constant of channel maintenance, indicating low drainage density"

def get_ruggedness_number_interpretation(ruggedness_number):
    if ruggedness_number < 0.1:
        return "Extremely low ruggedness"
    elif 0.1 <= ruggedness_number < 0.5:
        return "Low ruggedness"
    elif 0.5 <= ruggedness_number < 1:
        return "Moderate ruggedness"
    elif 1 <= ruggedness_number < 2:
        return "High ruggedness"
    else:
        return "Extremely high ruggedness"

def get_time_of_concentration_interpretation(time_of_concentration):
    if time_of_concentration is None:
        return "Unable to calculate time of concentration"
    elif time_of_concentration < 1:
        return "Very short time of concentration, indicating rapid response to rainfall"
    elif 1 <= time_of_concentration < 3:
        return "Short time of concentration"
    elif 3 <= time_of_concentration < 6:
        return "Moderate time of concentration"
    else:
        return "Long time of concentration, indicating slow response to rainfall"


def get_bifurcation_ratio_interpretation(bifurcation_ratio):
    if bifurcation_ratio is None:
        return "Unable to calculate bifurcation ratio"
    elif bifurcation_ratio < 3:
        return "Low bifurcation ratio, indicating uniform lithology and gentle slopes"
    elif 3 <= bifurcation_ratio <= 5:
        return "Normal bifurcation ratio for natural drainage systems"
    else:
        return "High bifurcation ratio, indicating steep slopes and structural control"

def get_drainage_intensity_interpretation(drainage_intensity):
    if drainage_intensity is None:
        return "Unable to calculate drainage intensity"
    elif drainage_intensity < 1:
        return "Low drainage intensity"
    elif 1 <= drainage_intensity < 2:
        return "Moderate drainage intensity"
    elif 2 <= drainage_intensity < 3:
        return "High drainage intensity"
    else:
        return "Very high drainage intensity"

def get_relief_interpretation(relief):
    if relief < 100:
        return "Low relief, indicating flat terrain"
    elif 100 <= relief < 300:
        return "Moderate relief"
    elif 300 <= relief < 600:
        return "High relief"
    else:
        return "Very high relief, indicating mountainous terrain"

def get_drainage_texture_interpretation(drainage_texture):
    if drainage_texture < 2:
        return "Coarse drainage texture"
    elif 2 <= drainage_texture < 4:
        return "Moderate drainage texture"
    elif 4 <= drainage_texture < 6:
        return "Fine drainage texture"
    else:
        return "Very fine drainage texture"

def get_infiltration_number_interpretation(infiltration_number):
    if infiltration_number < 1:
        return "Low infiltration number, indicating high infiltration"
    elif 1 <= infiltration_number < 3:
        return "Moderate infiltration number"
    elif 3 <= infiltration_number < 5:
        return "High infiltration number"
    else:
        return "Very high infiltration number"

def get_fitness_ratio_interpretation(fitness_ratio):
    if fitness_ratio < 0.2:
        return "Low fitness ratio, indicating inefficient drainage network"
    elif 0.2 <= fitness_ratio < 0.4:
        return "Moderate fitness ratio"
    else:
        return "High fitness ratio, indicating efficient drainage network"

def get_asymmetry_factor_interpretation(asymmetry_factor):
    if asymmetry_factor < 45:
        return "Significant tilt to the right (looking downstream)"
    elif 45 <= asymmetry_factor < 55:
        return "Relatively symmetric basin"
    else:
        return "Significant tilt to the left (looking downstream)"

def get_orographic_coefficient_interpretation(orographic_coefficient):
    if orographic_coefficient < 6:
        return "Low orographic influence"
    elif 6 <= orographic_coefficient < 18:
        return "Moderate orographic influence"
    else:
        return "High orographic influence"

# New interpretation functions for the added parameters
def get_relief_ratio_interpretation(relief_ratio):
    if relief_ratio < 0.1:
        return "Low relief ratio, indicating relatively flat terrain"
    elif 0.1 <= relief_ratio < 0.3:
        return "Moderate relief ratio"
    else:
        return "High relief ratio, indicating steep terrain"

def get_hortons_form_factor_interpretation(form_factor):
    if form_factor < 0.3:
        return "Elongated basin shape"
    elif 0.3 <= form_factor < 0.5:
        return "Slightly elongated basin shape"
    elif 0.5 <= form_factor < 0.75:
        return "Normal basin shape"
    else:
        return "Circular basin shape"

def get_schumms_elongation_ratio_interpretation(elongation_ratio):
    if elongation_ratio < 0.6:
        return "Elongated basin"
    elif 0.6 <= elongation_ratio < 0.8:
        return "Less elongated basin"
    elif 0.8 <= elongation_ratio < 0.9:
        return "Oval shaped basin"
    else:
        return "Circular basin"

def get_main_channel_gradient_interpretation(gradient):
    if gradient < 0.005:
        return "Very low gradient, indicative of a flat channel"
    elif 0.005 <= gradient < 0.02:
        return "Low gradient channel"
    elif 0.02 <= gradient < 0.04:
        return "Moderate gradient channel"
    else:
        return "High gradient channel"

def get_main_channel_sinuosity_interpretation(sinuosity):
    if sinuosity < 1.05:
        return "Almost straight channel"
    elif 1.05 <= sinuosity < 1.25:
        return "Winding channel"
    elif 1.25 <= sinuosity < 1.5:
        return "Twisty channel"
    else:
        return "Meandering channel"

def get_massivity_index_interpretation(massivity_index):
    if massivity_index < 50:
        return "Low massivity, indicating relatively flat terrain"
    elif 50 <= massivity_index < 100:
        return "Moderate massivity"
    else:
        return "High massivity, indicating mountainous terrain"

def get_texture_ratio_interpretation(texture_ratio):
    if texture_ratio < 4:
        return "Coarse texture"
    elif 4 <= texture_ratio < 10:
        return "Intermediate texture"
    else:
        return "Fine texture"

def get_junction_density_interpretation(junction_density):
    if junction_density < 1:
        return "Low junction density"
    elif 1 <= junction_density < 3:
        return "Moderate junction density"
    else:
        return "High junction density"

def get_storage_coefficient_interpretation(storage_coefficient):
    if storage_coefficient < 10:
        return "Low storage capacity"
    elif 10 <= storage_coefficient < 30:
        return "Moderate storage capacity"
    else:
        return "High storage capacity"

def get_main_channel_slope_interpretation(slope_percent):
    if slope_percent is None:
        return "Unable to calculate main channel slope"
    elif slope_percent < 1:
        return "Very gentle slope"
    elif 1 <= slope_percent < 3:
        return "Gentle slope"
    elif 3 <= slope_percent < 5:
        return "Moderate slope"
    elif 5 <= slope_percent < 10:
        return "Steep slope"
    else:
        return "Very steep slope"

def get_compensated_channel_slope_interpretation(slope_percent):
    if slope_percent is None:
        return "Unable to calculate compensated channel slope"
    elif slope_percent < 1:
        return "Very gentle compensated slope"
    elif 1 <= slope_percent < 3:
        return "Gentle compensated slope"
    elif 3 <= slope_percent < 5:
        return "Moderate compensated slope"
    elif 5 <= slope_percent < 10:
        return "Steep compensated slope"
    else:
        return "Very steep compensated slope"

# Source: https://www.sciencedirect.com/science/article/pii/S258947142300030X