"""Spanish translations for ArcGeek Calculator (English text -> Spanish).

Used by translation.ag_tr() when QGIS runs in Spanish. Keys must match the
English text in the code exactly; a missing key just shows English.
"""

ES = {
    'Calculate Point Coordinates':
        'Calcular coordenadas de puntos',
    'Calculate Line Geometry':
        'Calcular geometría de líneas',
    'Calculate Polygon Geometry':
        'Calcular geometría de polígonos',
    'Calculate Angles':
        'Calcular ángulos',
    'Extract Ordered Points from Polygons':
        'Extraer puntos ordenados de polígonos',
    'Lines to Ordered Points':
        'Líneas a puntos ordenados',
    'Azimuth and Distance from Coordinates and Table':
        'Azimut y distancia desde coordenadas y tabla',
    'Export to CSV (Excel compatible)':
        'Exportar a CSV (compatible con Excel)',
    'Tree Planting Pattern Generator':
        'Generador de patrones de plantación',
    'Stream Network with Order':
        'Red de drenaje con orden',
    'Watershed Basin Delineation':
        'Delimitación de cuenca hidrográfica',
    'Watershed Morphometric Analysis':
        'Análisis morfométrico de cuencas',
    'Multiple Basin Delineation by Points':
        'Delimitación de múltiples cuencas por puntos',
    'Global Curve Number':
        'Número de curva global',
    'Land Use Change Detection':
        'Detección de cambios de uso del suelo',
    'Weighted Sum':
        'Suma ponderada',
    'Least Cost Path Finder':
        'Ruta de menor costo',
    'Dam Flood Simulation':
        'Simulación de inundación por presa',
    'Kriging Analysis':
        'Análisis Kriging',
    'Export Contours to 3D CAD':
        'Exportar curvas de nivel a CAD 3D',
    'Optimized Parcel Division':
        'División optimizada de parcelas',
    'Topographic Profile':
        'Perfil topográfico',
    'Manage Basemaps (Google, Bing, Esri)':
        'Gestionar mapas base (Google, Bing, Esri)',
    'Screen Capture':
        'Captura de pantalla',
    'Satellite Index Calculator':
        'Calculadora de índices satelitales',
    'Enhanced Image Classification':
        'Clasificación avanzada de imágenes',
    'MCP Server':
        'Servidor MCP',
    'Go to XY':
        'Ir a XY',
    'Calculate XY Coordinates':
        'Calcular coordenadas XY',
    'Calculate Length':
        'Calcular longitud',
    'Calculate Area and Perimeter':
        'Calcular área y perímetro',
    'Manage Basemaps':
        'Gestionar mapas base',
    'Apply':
        'Aplicar',
    'Restart Required':
        'Reinicio necesario',
    'Basemaps have been added successfully. Please restart QGIS to see the changes.':
        'Los mapas base se agregaron correctamente. Reinicie QGIS para ver los cambios.',
    'No Changes':
        'Sin cambios',
    'No new basemaps were added.':
        'No se agregaron mapas base nuevos.',
    '\n        This algorithm performs a comprehensive analysis of a hydrological basin.\n        It calculates various morphometric parameters and provides interpretations.\n\n        Parameters:\n            Basin layer: A polygon layer representing the basin boundary\n            Stream network: A line layer representing the stream network within the basin\n            Stream Order Field: Field containing stream order (Strahler)\n            Digital Elevation Model: A raster layer representing the terrain elevation\n            Decimal precision: Number of decimal places for the results (default: 4)\n            Hypsometric calculation method: Choose between QGIS Algorithm (fast) or Pixel-by-Pixel (precise)\n\n        Outputs:\n            Output Report: A table with calculated morphometric parameters and their interpretations\n            Hypsometric Curve Output: HTML file containing the hypsometric curve visualization\n\n        Note: All input layers must have the same Coordinate Reference System (CRS).\n        ':
        '\n        Este algoritmo realiza un análisis completo de una cuenca hidrográfica.\n        Calcula diversos parámetros morfométricos y ofrece su interpretación.\n\n        Parámetros:\n            Capa de cuenca: capa de polígonos con el límite de la cuenca\n            Red de drenaje: capa de líneas con la red de cauces dentro de la cuenca\n            Campo de orden de cauce: campo con el orden de cauce (Strahler)\n            Modelo digital de elevación: ráster con la elevación del terreno\n            Precisión decimal: número de decimales de los resultados (por defecto: 4)\n            Método de cálculo hipsométrico: elija entre algoritmo de QGIS (rápido) o píxel por píxel (preciso)\n\n        Salidas:\n            Reporte: tabla con los parámetros morfométricos calculados y su interpretación\n            Curva hipsométrica: archivo HTML con la visualización de la curva hipsométrica\n\n        Nota: todas las capas de entrada deben tener el mismo sistema de referencia de coordenadas (SRC).\n        ',
    'Basin layer':
        'Capa de cuenca',
    'Stream network':
        'Red de drenaje',
    'Field containing stream order (Strahler)':
        'Campo con el orden de cauce (Strahler)',
    'Digital Elevation Model':
        'Modelo digital de elevación',
    'Decimal precision':
        'Precisión decimal',
    'Hypsometric calculation method':
        'Método de cálculo hipsométrico',
    'QGIS Algorithm (Fast)':
        'Algoritmo de QGIS (rápido)',
    'Pixel-by-Pixel (Precise)':
        'Píxel por píxel (preciso)',
    'Morphometric Report':
        'Reporte morfométrico',
    'Hypsometric Curve Output':
        'Salida de la curva hipsométrica',
    'Processing morphometric analysis...':
        'Procesando el análisis morfométrico...',
    'Analysis completed successfully':
        'Análisis completado correctamente',
    'One or more input layers are invalid':
        'Una o más capas de entrada no son válidas',
    'Input layers have different Coordinate Reference Systems (CRS)':
        'Las capas de entrada tienen diferentes sistemas de referencia de coordenadas (SRC)',
    'The CRS "{0} - {1}" uses geographic coordinates (degrees). All input layers must be in a projected coordinate system (meters) for morphometric calculations to be correct. Please reproject your layers to a suitable projected CRS (e.g. UTM) before running this tool.':
        'El SRC "{0} - {1}" usa coordenadas geográficas (grados). Todas las capas de entrada deben estar en un sistema de coordenadas proyectado (metros) para que los cálculos morfométricos sean correctos. Reproyecte sus capas a un SRC proyectado adecuado (p. ej. UTM) antes de ejecutar esta herramienta.',
    'Warning: The CRS "{0}" map units are not meters. Results involving area, length, and slope may be in unexpected units.':
        'Advertencia: las unidades del SRC "{0}" no son metros. Los resultados de área, longitud y pendiente pueden quedar en unidades inesperadas.',
    'Failed to calculate basin parameters':
        'No se pudieron calcular los parámetros de la cuenca',
    'Hypsometric Integral (HI)':
        'Integral hipsométrica (HI)',
    'dimensionless':
        'adimensional',
    'An error occurred: {0}':
        'Ocurrió un error: {0}',
    'Could not open HTML report automatically: {0}':
        'No se pudo abrir automáticamente el reporte HTML: {0}',
    'Small basin':
        'Cuenca pequeña',
    'Medium-sized basin':
        'Cuenca mediana',
    'Large basin':
        'Cuenca grande',
    'Gently sloping':
        'Pendiente suave',
    'Moderately steep':
        'Moderadamente escarpada',
    'Steep':
        'Escarpada',
    'Elongated shape, low susceptibility to flash floods':
        'Forma alargada, baja susceptibilidad a crecidas súbitas',
    'Intermediate shape':
        'Forma intermedia',
    'Circular shape, high susceptibility to flash floods':
        'Forma circular, alta susceptibilidad a crecidas súbitas',
    'Elongated shape':
        'Forma alargada',
    'Oval shape':
        'Forma oval',
    'Circular shape':
        'Forma circular',
    'Strongly elongated shape':
        'Forma muy alargada',
    'Very coarse drainage texture':
        'Textura de drenaje muy gruesa',
    'Coarse drainage texture':
        'Textura de drenaje gruesa',
    'Moderate drainage texture':
        'Textura de drenaje moderada',
    'Fine drainage texture':
        'Textura de drenaje fina',
    'Very fine drainage texture':
        'Textura de drenaje muy fina',
    'Low stream frequency':
        'Frecuencia de cauces baja',
    'Moderate stream frequency':
        'Frecuencia de cauces moderada',
    'High stream frequency':
        'Frecuencia de cauces alta',
    'Very high stream frequency':
        'Frecuencia de cauces muy alta',
    'Almost circular shape':
        'Forma casi circular',
    'Oval-circular to oval-oblong shape':
        'Forma oval redonda a oval oblonga',
    'Oval-oblong shape':
        'Forma oval oblonga',
    'Rectangular-oblong shape':
        'Forma rectangular oblonga',
    'Unable to calculate length of overland flow':
        'No se pudo calcular la longitud del flujo superficial',
    'Short overland flow length, indicating high drainage density':
        'Longitud del flujo superficial corta, indica alta densidad de drenaje',
    'Moderate overland flow length':
        'Longitud del flujo superficial moderada',
    'Long overland flow length, indicating low drainage density':
        'Longitud del flujo superficial larga, indica baja densidad de drenaje',
    'Unable to calculate constant of channel maintenance':
        'No se pudo calcular la constante de mantenimiento del cauce',
    'Low constant of channel maintenance, indicating high drainage density':
        'Constante de mantenimiento del cauce baja, indica alta densidad de drenaje',
    'Moderate constant of channel maintenance':
        'Constante de mantenimiento del cauce moderada',
    'High constant of channel maintenance, indicating low drainage density':
        'Constante de mantenimiento del cauce alta, indica baja densidad de drenaje',
    'Unable to calculate ruggedness number':
        'No se pudo calcular el número de rugosidad',
    'Extremely low ruggedness':
        'Rugosidad extremadamente baja',
    'Low ruggedness':
        'Rugosidad baja',
    'Moderate ruggedness':
        'Rugosidad moderada',
    'High ruggedness':
        'Rugosidad alta',
    'Extremely high ruggedness':
        'Rugosidad extremadamente alta',
    'Unable to calculate time of concentration':
        'No se pudo calcular el tiempo de concentración',
    'Very short time of concentration, indicating rapid response to rainfall':
        'Tiempo de concentración muy corto, indica respuesta rápida a la lluvia',
    'Short time of concentration':
        'Tiempo de concentración corto',
    'Moderate time of concentration':
        'Tiempo de concentración moderado',
    'Long time of concentration, indicating slow response to rainfall':
        'Tiempo de concentración largo, indica respuesta lenta a la lluvia',
    'Unable to calculate bifurcation ratio':
        'No se pudo calcular la relación de bifurcación',
    'Low bifurcation ratio, indicating uniform lithology and gentle slopes':
        'Relación de bifurcación baja, indica litología uniforme y pendientes suaves',
    'Normal bifurcation ratio for natural drainage systems':
        'Relación de bifurcación normal para sistemas de drenaje naturales',
    'High bifurcation ratio, indicating steep slopes and structural control':
        'Relación de bifurcación alta, indica pendientes fuertes y control estructural',
    'Unable to calculate drainage intensity':
        'No se pudo calcular la intensidad de drenaje',
    'Low drainage intensity':
        'Intensidad de drenaje baja',
    'Moderate drainage intensity':
        'Intensidad de drenaje moderada',
    'High drainage intensity':
        'Intensidad de drenaje alta',
    'Very high drainage intensity':
        'Intensidad de drenaje muy alta',
    'Unable to calculate relief':
        'No se pudo calcular el relieve',
    'Low relief, indicating flat terrain':
        'Relieve bajo, indica terreno plano',
    'Moderate relief':
        'Relieve moderado',
    'High relief':
        'Relieve alto',
    'Very high relief, indicating mountainous terrain':
        'Relieve muy alto, indica terreno montañoso',
    'Low infiltration number, indicating high infiltration':
        'Número de infiltración bajo, indica alta infiltración',
    'Moderate infiltration number':
        'Número de infiltración moderado',
    'High infiltration number':
        'Número de infiltración alto',
    'Very high infiltration number':
        'Número de infiltración muy alto',
    'Low fitness ratio, indicating inefficient drainage network':
        'Relación de ajuste baja, indica una red de drenaje poco eficiente',
    'Moderate fitness ratio':
        'Relación de ajuste moderada',
    'High fitness ratio, indicating efficient drainage network':
        'Relación de ajuste alta, indica una red de drenaje eficiente',
    'Unable to calculate asymmetry factor':
        'No se pudo calcular el factor de asimetría',
    'Significant tilt to the right (looking downstream)':
        'Inclinación significativa hacia la derecha (mirando aguas abajo)',
    'Relatively symmetric basin':
        'Cuenca relativamente simétrica',
    'Significant tilt to the left (looking downstream)':
        'Inclinación significativa hacia la izquierda (mirando aguas abajo)',
    'Unable to calculate orographic coefficient':
        'No se pudo calcular el coeficiente orográfico',
    'Low orographic influence':
        'Influencia orográfica baja',
    'Moderate orographic influence':
        'Influencia orográfica moderada',
    'High orographic influence':
        'Influencia orográfica alta',
    'Low relief ratio, indicating relatively flat terrain':
        'Relación de relieve baja, indica terreno relativamente plano',
    'Moderate relief ratio':
        'Relación de relieve moderada',
    'High relief ratio, indicating steep terrain':
        'Relación de relieve alta, indica terreno escarpado',
    'Elongated basin shape':
        'Cuenca de forma alargada',
    'Slightly elongated basin shape':
        'Cuenca de forma ligeramente alargada',
    'Normal basin shape':
        'Cuenca de forma normal',
    'Circular basin shape':
        'Cuenca de forma circular',
    'Elongated basin':
        'Cuenca alargada',
    'Less elongated basin':
        'Cuenca poco alargada',
    'Oval shaped basin':
        'Cuenca de forma oval',
    'Circular basin':
        'Cuenca circular',
    'Unable to calculate main channel gradient':
        'No se pudo calcular el gradiente del cauce principal',
    'Very low gradient, indicative of a flat channel':
        'Gradiente muy bajo, propio de un cauce plano',
    'Low gradient channel':
        'Cauce de gradiente bajo',
    'Moderate gradient channel':
        'Cauce de gradiente moderado',
    'High gradient channel':
        'Cauce de gradiente alto',
    'Unable to calculate main channel sinuosity':
        'No se pudo calcular la sinuosidad del cauce principal',
    'Almost straight channel':
        'Cauce casi recto',
    'Winding channel':
        'Cauce sinuoso',
    'Twisty channel':
        'Cauce muy sinuoso',
    'Meandering channel':
        'Cauce meándrico',
    'Unable to calculate massivity index':
        'No se pudo calcular el coeficiente de masividad',
    'Low massivity, indicating relatively flat terrain':
        'Masividad baja, indica terreno relativamente plano',
    'Moderate massivity':
        'Masividad moderada',
    'High massivity, indicating mountainous terrain':
        'Masividad alta, indica terreno montañoso',
    'Coarse texture':
        'Textura gruesa',
    'Intermediate texture':
        'Textura intermedia',
    'Fine texture':
        'Textura fina',
    'Low junction density':
        'Densidad de confluencias baja',
    'Moderate junction density':
        'Densidad de confluencias moderada',
    'High junction density':
        'Densidad de confluencias alta',
    'Unable to calculate storage coefficient':
        'No se pudo calcular el coeficiente de almacenamiento',
    'Low storage capacity':
        'Capacidad de almacenamiento baja',
    'Moderate storage capacity':
        'Capacidad de almacenamiento moderada',
    'High storage capacity':
        'Capacidad de almacenamiento alta',
    'Unable to calculate main channel slope':
        'No se pudo calcular la pendiente del cauce principal',
    'Very gentle slope':
        'Pendiente muy suave',
    'Gentle slope':
        'Pendiente suave',
    'Moderate slope':
        'Pendiente moderada',
    'Steep slope':
        'Pendiente fuerte',
    'Very steep slope':
        'Pendiente muy fuerte',
    'Unable to calculate compensated channel slope':
        'No se pudo calcular la pendiente compensada del cauce',
    'Very gentle compensated slope':
        'Pendiente compensada muy suave',
    'Gentle compensated slope':
        'Pendiente compensada suave',
    'Moderate compensated slope':
        'Pendiente compensada moderada',
    'Steep compensated slope':
        'Pendiente compensada fuerte',
    'Very steep compensated slope':
        'Pendiente compensada muy fuerte',
    'Basin Area (A)':
        'Área de la cuenca (A)',
    'Perimeter (P)':
        'Perímetro (P)',
    'Basin Length (Lb)':
        'Longitud de la cuenca (Lb)',
    'Basin Width (B)':
        'Ancho de la cuenca (B)',
    'Maximum Elevation':
        'Elevación máxima',
    'Minimum Elevation':
        'Elevación mínima',
    'Mean Elevation':
        'Elevación media',
    'Relief (H)':
        'Relieve (H)',
    'Mean Slope of the Basin (degrees)':
        'Pendiente media de la cuenca (grados)',
    'Mean Slope of the Basin (percent)':
        'Pendiente media de la cuenca (porcentaje)',
    'Main Channel Length (Lc)':
        'Longitud del cauce principal (Lc)',
    'Start Elevation (Main Channel)':
        'Elevación inicial (cauce principal)',
    'End Elevation (Main Channel)':
        'Elevación final (cauce principal)',
    'Main Channel Gradient':
        'Gradiente del cauce principal',
    'Main Channel Sinuosity':
        'Sinuosidad del cauce principal',
    'Main Channel Slope - Endpoints (S)':
        'Pendiente del cauce principal - extremos (S)',
    'Compensated Channel Slope (Taylor-Schwartz)':
        'Pendiente compensada del cauce (Taylor-Schwartz)',
    'Compensated Channel Slope - TS (%)':
        'Pendiente compensada del cauce - TS (%)',
    'Total Length of Channels (Lt)':
        'Longitud total de cauces (Lt)',
    'Drainage Density (Dd)':
        'Densidad de drenaje (Dd)',
    'Stream Frequency (Fs)':
        'Frecuencia de cauces (Fs)',
    'Drainage Texture (Dt)':
        'Textura de drenaje (Dt)',
    'Drainage Intensity (Id)':
        'Intensidad de drenaje (Id)',
    'Length of Overland Flow (Lo)':
        'Longitud del flujo superficial (Lo)',
    'Constant of Channel Maintenance (C)':
        'Constante de mantenimiento del cauce (C)',
    'Infiltration Number (If)':
        'Número de infiltración (If)',
    'Stream Order (Strahler)':
        'Orden de cauce (Strahler)',
    'Number of Streams (Nu)':
        'Número de cauces (Nu)',
    'Mean Stream Length (Lm)':
        'Longitud media de cauces (Lm)',
    'Bifurcation Ratio (Rb)':
        'Relación de bifurcación (Rb)',
    'Junction Density':
        'Densidad de confluencias',
    'Form Factor (Ff) - Horton (1932)':
        'Factor de forma (Ff) - Horton (1932)',
    'Elongation Ratio (Re) - Schumm (1956)':
        'Relación de elongación (Re) - Schumm (1956)',
    'Circularity Ratio (Rc)':
        'Relación de circularidad (Rc)',
    'Compactness Coefficient of Gravelius (Kc)':
        'Coeficiente de compacidad de Gravelius (Kc)',
    'Asymmetry Factor (Af)':
        'Factor de asimetría (Af)',
    'Fitness Ratio (Rf)':
        'Relación de ajuste (Rf)',
    'Ruggedness Number (Rn)':
        'Número de rugosidad (Rn)',
    'Orographic Coefficient (Oc)':
        'Coeficiente orográfico (Oc)',
    'Massivity Index':
        'Coeficiente de masividad',
    'Storage Coefficient':
        'Coeficiente de almacenamiento',
    'Time of Concentration - Kirpich (Tc)':
        'Tiempo de concentración - Kirpich (Tc)',
    'Time of Concentration - Kerby (Tc)':
        'Tiempo de concentración - Kerby (Tc)',
    'Time of Concentration - Giandotti (Tc)':
        'Tiempo de concentración - Giandotti (Tc)',
    'Time of Concentration - Témez (Tc)':
        'Tiempo de concentración - Témez (Tc)',
    'Time of Concentration - USDA (Tc)':
        'Tiempo de concentración - USDA (Tc)',
    'Time of Concentration - Passini (Tc)':
        'Tiempo de concentración - Passini (Tc)',
    'Time of Concentration - Ventura-Heras (Tc)':
        'Tiempo de concentración - Ventura-Heras (Tc)',
    'Time of Concentration - Bransby-Williams (Tc)':
        'Tiempo de concentración - Bransby-Williams (Tc)',
    'Time of Concentration - Johnstone-Cross (Tc)':
        'Tiempo de concentración - Johnstone-Cross (Tc)',
    'Time of Concentration - Clark (Tc)':
        'Tiempo de concentración - Clark (Tc)',
    'Time of Concentration - California Culverts (Tc)':
        'Tiempo de concentración - California Culverts (Tc)',
    'Compensated Channel Slope':
        'Pendiente compensada del cauce',
    'Compensated Channel Slope (%)':
        'Pendiente compensada del cauce (%)',
    'Invalid DEM layer. Cannot proceed with calculations.':
        'Capa DEM no válida. No se pueden realizar los cálculos.',
    'Start Elevation (highest point): {0}':
        'Elevación inicial (punto más alto): {0}',
    'End Elevation (lowest point): {0}':
        'Elevación final (punto más bajo): {0}',
    'Slope: {0}%':
        'Pendiente: {0}%',
    'Warning: Unable to calculate slope due to invalid elevation values':
        'Advertencia: no se pudo calcular la pendiente por valores de elevación no válidos',
    'Basin perimeter':
        'Perímetro de la cuenca',
    'Basin length':
        'Longitud de la cuenca',
    'Basin width':
        'Ancho de la cuenca',
    'm a.s.l.':
        'm s.n.m.',
    'Maximum elevation of the basin':
        'Elevación máxima de la cuenca',
    'Minimum elevation of the basin':
        'Elevación mínima de la cuenca',
    'Average elevation of the basin':
        'Elevación promedio de la cuenca',
    'degrees':
        'grados',
    'Length of the main channel':
        'Longitud del cauce principal',
    'Elevation at the upstream end of the main channel':
        'Elevación en el extremo aguas arriba del cauce principal',
    'Elevation at the downstream end of the main channel':
        'Elevación en el extremo aguas abajo del cauce principal',
    'Total length of all channels':
        'Longitud total de todos los cauces',
    'streams/km²':
        'cauces/km²',
    'Highest stream order (Strahler): {0}':
        'Orden de cauce más alto (Strahler): {0}',
    'Total number of stream segments':
        'Número total de tramos de cauce',
    'Average length of streams':
        'Longitud promedio de los cauces',
    'junctions/km²':
        'confluencias/km²',
    'hours':
        'horas',
    'Calculated compensated channel slope using linear regression':
        'Pendiente compensada del cauce calculada por regresión lineal',
    'Error calculating DEM statistics: {0}':
        'Error al calcular las estadísticas del DEM: {0}',
    'Unable to calculate':
        'No se pudo calcular',
    'Insufficient data to calculate slope':
        'Datos insuficientes para calcular la pendiente',
    '\n        This algorithm calculates internal and external angles at vertices of lines or polygons.\n        \n        Parameters:\n        - Input layer: Line or polygon layer to process\n        - Output: Point layer with calculated angles\n        \n        Requirements:\n        - Line features must have at least 3 vertices\n        - Polygon features must have at least 4 vertices (closed)\n        \n        Output fields:\n        - internal_angle: Interior angle at the vertex\n        - external_angle: Exterior angle at the vertex\n        - feature_id: ID of the original feature\n        ':
        '\n        Este algoritmo calcula los ángulos internos y externos en los vértices de líneas o polígonos.\n\n        Parámetros:\n        - Capa de entrada: capa de líneas o polígonos a procesar\n        - Salida: capa de puntos con los ángulos calculados\n\n        Requisitos:\n        - Las líneas deben tener al menos 3 vértices\n        - Los polígonos deben tener al menos 4 vértices (cerrados)\n\n        Campos de salida:\n        - internal_angle: ángulo interior en el vértice\n        - external_angle: ángulo exterior en el vértice\n        - feature_id: ID de la entidad original\n        ',
    'Processed {0} features and created {1} angle points':
        'Se procesaron {0} entidades y se crearon {1} puntos con ángulos',
    'Input layer (line or polygon)':
        'Capa de entrada (líneas o polígonos)',
    'Output vertices with angles':
        'Vértices de salida con ángulos',
    'Azimuth and distance from Coordinates and Table':
        'Azimut y distancia desde coordenadas y tabla',
    '\n        This algorithm calculates a line and points from starting coordinates and a table of distances and angles.\n\n        Parameters:\n        - Starting X and Y coordinates: The initial point of the line.\n        - Input table: A vector layer or table (csv, txt, etc) containing distance and angle data.\n        - Distance field: The field in the input table containing distance values.\n        - Angle field: The field in the input table containing angle values.\n        - Angle type in input data: Specify whether the input angles are Azimuth or Polar.\n        - Observations field (optional): A field for additional information about each point.\n        - Output CRS: The coordinate reference system for the output layers.\n\n        Outputs:\n        - A line layer with segments connecting the calculated points.\n        - A point layer with all calculated points.\n\n        The algorithm reads the input table row by row, calculating new points based on the distance and angle from the previous point. It starts from the given X and Y coordinates and creates a new point for each row in the table.\n\n        Angle types:\n        - Azimuth: Measured clockwise from north (0-360 degrees)\n        - Polar: Measured counterclockwise from east (0-360 degrees)\n\n        Use this tool when you need to:\n        - Convert tabular distance and angle data into spatial features.\n        - Generate a line path from a series of movements.\n        - Create point locations from relative positioning data.\n\n        Note: Ensure that you select the correct angle type that matches your input data, and that you select the appropriate CRS for your data to avoid potential inaccuracies in the results.\n        ':
        '\n        Este algoritmo calcula una línea y puntos a partir de coordenadas iniciales y una tabla de distancias y ángulos.\n\n        Parámetros:\n        - Coordenadas X e Y iniciales: punto de partida de la línea.\n        - Tabla de entrada: capa vectorial o tabla (csv, txt, etc.) con los datos de distancia y ángulo.\n        - Campo de distancia: campo de la tabla con los valores de distancia.\n        - Campo de ángulo: campo de la tabla con los valores de ángulo.\n        - Tipo de ángulo en los datos: indique si los ángulos son azimut o polares.\n        - Campo de observaciones (opcional): campo con información adicional de cada punto.\n        - SRC de salida: sistema de referencia de coordenadas de las capas de salida.\n\n        Salidas:\n        - Una capa de líneas con los segmentos que unen los puntos calculados.\n        - Una capa de puntos con todos los puntos calculados.\n\n        El algoritmo lee la tabla fila por fila y calcula cada punto nuevo a partir de la distancia y el ángulo desde el punto anterior. Parte de las coordenadas X e Y indicadas y crea un punto por cada fila de la tabla.\n\n        Tipos de ángulo:\n        - Azimut: medido en sentido horario desde el norte (0-360 grados)\n        - Polar: medido en sentido antihorario desde el este (0-360 grados)\n\n        Use esta herramienta cuando necesite:\n        - Convertir datos tabulares de distancia y ángulo en entidades espaciales.\n        - Generar un recorrido de línea a partir de una serie de desplazamientos.\n        - Crear ubicaciones de puntos a partir de datos de posicionamiento relativo.\n\n        Nota: asegúrese de elegir el tipo de ángulo que corresponde a sus datos y el SRC adecuado, para evitar imprecisiones en los resultados.\n        ',
    'Starting X coordinate':
        'Coordenada X inicial',
    'Starting Y coordinate':
        'Coordenada Y inicial',
    'Input table':
        'Tabla de entrada',
    'Distance field':
        'Campo de distancia',
    'Angle field':
        'Campo de ángulo',
    'Angle type in input data':
        'Tipo de ángulo en los datos',
    'Azimuth':
        'Azimut',
    'Observations field':
        'Campo de observaciones',
    'Output CRS':
        'SRC de salida',
    'Output line layer':
        'Capa de líneas de salida',
    'Output points layer':
        'Capa de puntos de salida',
    'Could not create output layers':
        'No se pudieron crear las capas de salida',
    'Error in row {0}: {1}':
        'Error en la fila {0}: {1}',
    'Calculates and adds a length field to line layers.\n\n        Parameters:\n        - Input line layer: Select the layer to process.\n        - Length units: Choose the desired unit for length calculation.\n        - Precision: Set the number of decimal places for results.\n        - Calculation method: Choose between Cartesian (planar) or Ellipsoidal (curved surface) calculations.\n        - Selected features only: Process only selected features if checked.\n        - Update existing layer: Modify the input layer instead of creating a new one.\n        - CRS for calculations: Optionally specify a CRS for the calculations.\n        - Output layer: Specify the output layer (if not updating existing).\n\n        Supports various units and calculation methods (Cartesian/Ellipsoidal).':
        'Calcula y agrega un campo de longitud a capas de líneas.\n\n        Parámetros:\n        - Capa de líneas de entrada: seleccione la capa a procesar.\n        - Unidades de longitud: elija la unidad para el cálculo de longitud.\n        - Precisión: número de decimales de los resultados.\n        - Método de cálculo: elija entre cálculo cartesiano (plano) o elipsoidal (superficie curva).\n        - Solo entidades seleccionadas: procesa solo las entidades seleccionadas si está marcado.\n        - Actualizar la capa existente: modifica la capa de entrada en lugar de crear una nueva.\n        - SRC para los cálculos: opcionalmente, indique un SRC para los cálculos.\n        - Capa de salida: capa de salida (si no se actualiza la existente).\n\n        Admite varias unidades y métodos de cálculo (cartesiano/elipsoidal).',
    'Input line layer':
        'Capa de líneas de entrada',
    'Length units':
        'Unidades de longitud',
    'Precision':
        'Precisión',
    'Calculation method':
        'Método de cálculo',
    'Cartesian':
        'Cartesiano',
    'Ellipsoidal':
        'Elipsoidal',
    'Selected features only':
        'Solo entidades seleccionadas',
    'Update existing layer':
        'Actualizar la capa existente',
    'CRS for calculations':
        'SRC para los cálculos',
    'Output layer':
        'Capa de salida',
    'Calculates and adds area and perimeter fields to polygon layers.\n\n        Parameters:\n        - Input polygon layer: Select the layer to process.\n        - Area units: Choose the desired unit for area calculation.\n        - Perimeter units: Choose the desired unit for perimeter calculation.\n        - Precision: Set the number of decimal places for results.\n        - Calculation method: Choose between Cartesian (planar) or Ellipsoidal (curved surface) calculations.\n        - Selected features only: Process only selected features if checked.\n        - Update existing layer: Modify the input layer instead of creating a new one.\n        - CRS for calculations: Optionally specify a CRS for the calculations.\n        - Output layer: Specify the output layer (if not updating existing).\n\n        Supports various units and calculation methods (Cartesian/Ellipsoidal).':
        'Calcula y agrega campos de área y perímetro a capas de polígonos.\n\n        Parámetros:\n        - Capa de polígonos de entrada: seleccione la capa a procesar.\n        - Unidades de área: elija la unidad para el cálculo de área.\n        - Unidades de perímetro: elija la unidad para el cálculo de perímetro.\n        - Precisión: número de decimales de los resultados.\n        - Método de cálculo: elija entre cálculo cartesiano (plano) o elipsoidal (superficie curva).\n        - Solo entidades seleccionadas: procesa solo las entidades seleccionadas si está marcado.\n        - Actualizar la capa existente: modifica la capa de entrada en lugar de crear una nueva.\n        - SRC para los cálculos: opcionalmente, indique un SRC para los cálculos.\n        - Capa de salida: capa de salida (si no se actualiza la existente).\n\n        Admite varias unidades y métodos de cálculo (cartesiano/elipsoidal).',
    'Input polygon layer':
        'Capa de polígonos de entrada',
    'Area units':
        'Unidades de área',
    'Perimeter units':
        'Unidades de perímetro',
    '\n        This algorithm exports contour lines to a 3D CAD file (DXF format) that preserves elevation values,\n        making it compatible with CAD software like AutoCAD, Civil 3D, BricsCAD, and other CAD applications\n        that support 3D DXF files.\n\n        Parameters:\n            Input contour layer: Vector line layer containing contours\n            Elevation field: Numeric field containing elevation values\n            Target CRS: Coordinate Reference System for the CAD export (optional)\n            CAD Version: AutoCAD version compatibility\n            Output CAD file: Destination path for the exported DXF\n\n        The algorithm automatically detects if your input is already 3D and:\n        - If 3D: Uses it directly\n        - If 2D: Converts it to 3D using the elevation field\n\n        The output DXF file will maintain Z-coordinates from your elevation values,\n        making it ready to use in CAD software.\n\n        Note: For best compatibility with older CAD software, use AutoCAD 2000 format.\n        ':
        '\n        Este algoritmo exporta curvas de nivel a un archivo CAD 3D (formato DXF) que conserva los valores de elevación,\n        compatible con programas CAD como AutoCAD, Civil 3D, BricsCAD y otras aplicaciones\n        que admiten archivos DXF 3D.\n\n        Parámetros:\n            Capa de curvas de nivel: capa de líneas con las curvas de nivel\n            Campo de elevación: campo numérico con los valores de elevación\n            SRC de destino: sistema de referencia de coordenadas para la exportación CAD (opcional)\n            Versión CAD: compatibilidad con la versión de AutoCAD\n            Archivo CAD de salida: ruta de destino del DXF exportado\n\n        El algoritmo detecta automáticamente si la entrada ya es 3D:\n        - Si es 3D: la usa directamente\n        - Si es 2D: la convierte a 3D usando el campo de elevación\n\n        El DXF de salida conserva las coordenadas Z a partir de los valores de elevación,\n        listo para usarse en programas CAD.\n\n        Nota: para máxima compatibilidad con programas CAD antiguos, use el formato AutoCAD 2000.\n        ',
    'Processing contours...':
        'Procesando curvas de nivel...',
    'Exporting to CAD format...':
        'Exportando a formato CAD...',
    'Input contour layer':
        'Capa de curvas de nivel',
    'Elevation field':
        'Campo de elevación',
    'Target CRS for CAD export':
        'SRC de destino para la exportación CAD',
    'CAD Version':
        'Versión CAD',
    'Output CAD file':
        'Archivo CAD de salida',
    'No input layer specified':
        'No se indicó la capa de entrada',
    'No elevation field specified':
        'No se indicó el campo de elevación',
    'Elevation field not found in layer':
        'El campo de elevación no existe en la capa',
    'Elevation field must contain numeric values':
        'El campo de elevación debe contener valores numéricos',
    'Input layer is already 3D, using directly...':
        'La capa de entrada ya es 3D, se usará directamente...',
    'Converting to 3D using elevation field...':
        'Convirtiendo a 3D con el campo de elevación...',
    'Output file was not created':
        'No se creó el archivo de salida',
    'Error during DXF export: {0}':
        'Error durante la exportación DXF: {0}',
    'Calculate Coordinates':
        'Calcular coordenadas',
    '\n        This algorithm calculates and adds coordinate information to a point layer in various formats.\n\n        Parameters:\n        - Input layer: The point layer to process.\n        - Modify the current layer: If checked, modifies the input layer. Otherwise, creates a new layer.\n        - Calculate XY: Adds X and Y fields with coordinates in the specified CRS.\n        - Decimal Degrees: Adds latitude and longitude fields in decimal degrees.\n        - Degrees Minutes Seconds (DDD° MM\' SSS.ss" <N|S|E|W>): Adds fields with coordinates in DMS format.\n        - Degrees Minutes Seconds (<N|S|E|W> DDD° MM\' SSS.ss"): Adds fields with coordinates in alternative DMS format.\n        - Precision: Set the number of decimal places for results.\n        - CRS for calculations: Specify a CRS for the calculations. If not specified, the layer\'s CRS will be used.\n\n        Note: Ensure that your input layer has a defined coordinate reference system (CRS) for accurate results.\n        ':
        '\n        Este algoritmo calcula y agrega información de coordenadas a una capa de puntos en varios formatos.\n\n        Parámetros:\n        - Capa de entrada: capa de puntos a procesar.\n        - Modificar la capa actual: si está marcado, modifica la capa de entrada; si no, crea una capa nueva.\n        - Calcular XY: agrega campos X e Y con las coordenadas en el SRC indicado.\n        - Grados decimales: agrega campos de latitud y longitud en grados decimales.\n        - Grados minutos segundos (DDD° MM\' SSS.ss" <N|S|E|W>): agrega campos con las coordenadas en formato GMS.\n        - Grados minutos segundos (<N|S|E|W> DDD° MM\' SSS.ss"): agrega campos con las coordenadas en un formato GMS alternativo.\n        - Precisión: número de decimales de los resultados.\n        - SRC para los cálculos: indique un SRC para los cálculos. Si no se indica, se usa el SRC de la capa.\n\n        Nota: asegúrese de que la capa de entrada tenga un sistema de referencia de coordenadas (SRC) definido para obtener resultados precisos.',
    'Input layer':
        'Capa de entrada',
    'Modify the current layer':
        'Modificar la capa actual',
    'Calculate XY':
        'Calcular XY',
    'Decimal Degrees':
        'Grados decimales',
    'Degrees Minutes Seconds (DDD° MM\' SSS.ss" <N|S|E|W>)':
        'Grados minutos segundos (DDD° MM\' SSS.ss" <N|S|E|W>)',
    'Degrees Minutes Seconds (<N|S|E|W> DDD° MM\' SSS.ss")':
        'Grados minutos segundos (<N|S|E|W> DDD° MM\' SSS.ss")',
    'Input CRS: {0}':
        'SRC de entrada: {0}',
    'Input layer feature count: {0}':
        'Número de entidades de la capa de entrada: {0}',
    'Fields to be added or updated: {0}':
        'Campos que se agregarán o actualizarán: {0}',
    'Input layer could not be loaded':
        'No se pudo cargar la capa de entrada',
    'Modified input layer: {0}':
        'Capa de entrada modificada: {0}',
    'Created new layer: {0}':
        'Capa nueva creada: {0}',
    'Error in processAlgorithm: {0}':
        'Error en processAlgorithm: {0}',
    'Could not retrieve the input layer for modification':
        'No se pudo obtener la capa de entrada para modificarla',
    'This algorithm simulates flooding by creating a water depth raster based on a DEM and specified water level.\n\n        Parameters:\n        - Input DEM: The Digital Elevation Model of the area.\n        - Water Level (m): The water level above the minimum elevation of the DEM.\n        - Study Area: The extent to process the DEM (optional). If not provided, the entire DEM will be used.\n        - Output Water Depth Raster: The resulting raster showing water depth.\n\n        The algorithm calculates the flood level by adding the specified water level to the minimum \n        DEM elevation within the study area (if specified) or the entire DEM. It then calculates \n        water depth where the elevation is below the flood level. Areas above the flood level are \n        set to NoData in the output raster.\n\n        Note: For very mountainous terrain, the algorithm will provide a warning as it may be \n        challenging to find suitable areas for dam construction in such landscapes.':
        'Este algoritmo simula una inundación generando un ráster de profundidad de agua a partir de un DEM y un nivel de agua indicado.\n\n        Parámetros:\n        - DEM de entrada: modelo digital de elevación del área.\n        - Nivel de agua (m): nivel de agua sobre la elevación mínima del DEM.\n        - Área de estudio: extensión del DEM a procesar (opcional). Si no se indica, se usa todo el DEM.\n        - Ráster de profundidad de agua: ráster resultante con la profundidad del agua.\n\n        El algoritmo calcula el nivel de inundación sumando el nivel de agua indicado a la elevación\n        mínima del DEM dentro del área de estudio (si se indica) o de todo el DEM. Luego calcula la\n        profundidad del agua donde la elevación está por debajo del nivel de inundación. Las zonas por\n        encima de ese nivel quedan como NoData en el ráster de salida.\n\n        Nota: en terrenos muy montañosos el algoritmo muestra una advertencia, porque en esos paisajes\n        puede ser difícil encontrar zonas adecuadas para construir una presa.',
    'DEM Min: {0}, Max: {1}':
        'DEM mínimo: {0}, máximo: {1}',
    'Flood level: {0}':
        'Nivel de inundación: {0}',
    'Calculating water depth...':
        'Calculando la profundidad del agua...',
    'The total volume of stored water is: {0:.2f} cubic meters':
        'El volumen total de agua almacenada es: {0:.2f} metros cúbicos',
    'The total flooded area is: {0:.2f} square meters':
        'El área total inundada es: {0:.2f} metros cuadrados',
    'Maximum water depth: {0:.2f} meters':
        'Profundidad máxima del agua: {0:.2f} metros',
    'Cell size used for calculation: {0:.2f} x {1:.2f} meters':
        'Tamaño de celda usado en el cálculo: {0:.2f} x {1:.2f} metros',
    'Input DEM':
        'DEM de entrada',
    'Water Level (m)':
        'Nivel de agua (m)',
    'Study Area':
        'Área de estudio',
    'Output Water Depth Raster':
        'Ráster de profundidad de agua',
    'Invalid DEM layer':
        'Capa DEM no válida',
    'Clipping DEM to specified study area...':
        'Recortando el DEM al área de estudio indicada...',
    'Using the entire DEM (no study area specified).':
        'Usando todo el DEM (no se indicó área de estudio).',
    'The DEM appears to be very mountainous. Due to the nature of the terrain, it may be challenging to find suitable areas for dam construction.':
        'El DEM parece muy montañoso. Por la naturaleza del terreno, puede ser difícil encontrar zonas adecuadas para construir una presa.',
    'Could not create raster output provider.':
        'No se pudo crear el proveedor del ráster de salida.',
    'Failed to clip DEM':
        'No se pudo recortar el DEM',
    '\n            This algorithm performs advanced image classification using various machine learning methods \n            with scientifically rigorous accuracy assessment.\n            \n            Available algorithms:\n            - Random Forest: Robust ensemble learning method with feature importance analysis\n            - Gaussian Mixture Model: Probabilistic model for clustering\n            - Support Vector Machine: Powerful discriminative classifier\n            - K-Nearest Neighbors: Simple distance-based classifier\n            \n            The tool requires:\n            - A point vector layer with class labels for training\n            - The classification field containing class labels\n            - A multiband raster image to classify\n            - Algorithm selection and parameters\n            \n            Outputs:\n            - Classified image with a raster attribute table\n            - Optional comprehensive HTML report following remote sensing standards\n            \n          \n            Note: This algorithm requires scikit-learn and numpy to be installed.\n            For full report functionality, matplotlib and seaborn are also recommended.\n            ':
        '\n            Este algoritmo realiza una clasificación avanzada de imágenes con varios métodos de aprendizaje automático\n            y una evaluación de exactitud con rigor científico.\n\n            Algoritmos disponibles:\n            - Random Forest: método de ensamble robusto con análisis de importancia de variables\n            - Modelo de mezcla gaussiana: modelo probabilístico de agrupamiento\n            - Máquina de vectores de soporte: clasificador discriminativo potente\n            - K vecinos más cercanos: clasificador simple basado en distancias\n\n            La herramienta requiere:\n            - Una capa de puntos con las etiquetas de clase para el entrenamiento\n            - El campo de clasificación con las etiquetas de clase\n            - Una imagen multibanda a clasificar\n            - La selección del algoritmo y sus parámetros\n\n            Salidas:\n            - Imagen clasificada con tabla de atributos ráster\n            - Reporte HTML completo opcional, según estándares de teledetección\n\n            Nota: este algoritmo requiere tener instalados scikit-learn y numpy.\n            Para el reporte completo se recomiendan también matplotlib y seaborn.\n            ',
    '🔍 Extracting image properties...':
        '🔍 Extrayendo las propiedades de la imagen...',
    '📊 Image info: {0}x{1} pixels, {2} bands':
        '📊 Información de la imagen: {0}x{1} píxeles, {2} bandas',
    '📍 Collecting training data...':
        '📍 Recolectando los datos de entrenamiento...',
    '✅ Collected {0} training samples in {1} classes.':
        '✅ Se recolectaron {0} muestras de entrenamiento en {1} clases.',
    '🔧 Preparing data for analysis...':
        '🔧 Preparando los datos para el análisis...',
    '📖 Reading {0} raster bands...':
        '📖 Leyendo {0} bandas ráster...',
    '🔄 Reshaping image data for analysis...':
        '🔄 Reorganizando los datos de la imagen para el análisis...',
    '📊 Preparing training and test data...':
        '📊 Preparando los datos de entrenamiento y prueba...',
    '📈 Training data: {0}, labels: {1}':
        '📈 Datos de entrenamiento: {0}, etiquetas: {1}',
    '📊 Class distribution: {0}':
        '📊 Distribución de clases: {0}',
    '🔀 Splitting data with test size: {0}':
        '🔀 Dividiendo los datos con tamaño de prueba: {0}',
    '📊 Training set: {0}, Test set: {1}':
        '📊 Conjunto de entrenamiento: {0}, conjunto de prueba: {1}',
    '🤖 Training the classifier...':
        '🤖 Entrenando el clasificador...',
    '📊 Calculating classification accuracy...':
        '📊 Calculando la exactitud de la clasificación...',
    '🎯 Classification performance: Overall Accuracy = {0:.4f} ({1:.2f}%)':
        '🎯 Desempeño de la clasificación: exactitud global = {0:.4f} ({1:.2f}%)',
    '🖼️ Classifying the full image...':
        '🖼️ Clasificando la imagen completa...',
    '💾 Formatting classified image...':
        '💾 Dando formato a la imagen clasificada...',
    '💾 Saving classified image...':
        '💾 Guardando la imagen clasificada...',
    '🎯 Enhanced image classification completed successfully! 🎯':
        '🎯 ¡Clasificación avanzada de la imagen completada correctamente! 🎯',
    '📋 CLASSIFICATION SUMMARY:':
        '📋 RESUMEN DE LA CLASIFICACIÓN:',
    '   🎯 Overall Accuracy: {0:.4f} ({1:.2f}%)':
        '   🎯 Exactitud global: {0:.4f} ({1:.2f}%)',
    '   📊 Classes: {0}':
        '   📊 Clases: {0}',
    '   📈 Training samples: {0}':
        '   📈 Muestras de entrenamiento: {0}',
    '   🎛️ Spectral bands: {0}':
        '   🎛️ Bandas espectrales: {0}',
    'Classification Algorithm':
        'Algoritmo de clasificación',
    'Gaussian Mixture Model':
        'Modelo de mezcla gaussiana',
    'Support Vector Machine':
        'Máquina de vectores de soporte',
    'K-Nearest Neighbors':
        'K vecinos más cercanos',
    'Number of estimators (for Random Forest)':
        'Número de estimadores (para Random Forest)',
    'Test set size (proportion)':
        'Tamaño del conjunto de prueba (proporción)',
    'Perform cross-validation assessment':
        'Realizar validación cruzada',
    'Training data points':
        'Puntos de entrenamiento',
    'Classification field':
        'Campo de clasificación',
    'Multiband image to classify':
        'Imagen multibanda a clasificar',
    'Generate classification report (HTML)':
        'Generar reporte de clasificación (HTML)',
    'Classification report (HTML)':
        'Reporte de clasificación (HTML)',
    'Classified Image':
        'Imagen clasificada',
    'Invalid raster: no bands found':
        'Ráster no válido: no se encontraron bandas',
    '⚠️ Single-band image detected. For optimal classification results, use multiband imagery.':
        '⚠️ Se detectó una imagen de una sola banda. Para mejores resultados de clasificación, use imágenes multibanda.',
    '⚠️ Sample size warning: {0} training points available. Recommended minimum: {1} points for {2} bands.':
        '⚠️ Advertencia de tamaño de muestra: hay {0} puntos de entrenamiento. Mínimo recomendado: {1} puntos para {2} bandas.',
    "No attribute named '{0}' in the training layer":
        "No existe el atributo '{0}' en la capa de entrenamiento",
    'Not enough valid training points found. At least 5 points are required.':
        'No se encontraron suficientes puntos de entrenamiento válidos. Se requieren al menos 5 puntos.',
    '⚠️ Smallest class has only {0} samples. Recommended minimum: {1} per class.':
        '⚠️ La clase más pequeña tiene solo {0} muestras. Mínimo recomendado: {1} por clase.',
    '📊 Raster data shape: {0}':
        '📊 Forma de los datos ráster: {0}',
    'Empty raster data - no pixel values found.':
        'Datos ráster vacíos: no se encontraron valores de píxel.',
    '📊 Multi-band image: {0} bands, {1}x{2} pixels':
        '📊 Imagen multibanda: {0} bandas, {1}x{2} píxeles',
    '📊 Single-band image: {0}x{1} pixels':
        '📊 Imagen de una banda: {0}x{1} píxeles',
    '⚠️ Band count mismatch - metadata says {0}, actual data has {1} bands':
        '⚠️ No coincide el número de bandas: los metadatos indican {0}, los datos tienen {1} bandas',
    '📊 Reshaped data size: {0}':
        '📊 Tamaño de los datos reorganizados: {0}',
    '🌲 Using Random Forest Classifier with {0} estimators':
        '🌲 Usando el clasificador Random Forest con {0} estimadores',
    "🔬 Cohen's Kappa = {0:.4f}":
        '🔬 Kappa de Cohen = {0:.4f}',
    '⚖️ Balanced Accuracy = {0:.4f}':
        '⚖️ Exactitud balanceada = {0:.4f}',
    '📈 Kappa interpretation: {0}':
        '📈 Interpretación del Kappa: {0}',
    '✅ Classification complete. Unique classes in result: {0}':
        '✅ Clasificación completada. Clases únicas en el resultado: {0}',
    '📋 Creating raster attribute table...':
        '📋 Creando la tabla de atributos ráster...',
    '📊 Scientific classification report generation was not requested.':
        '📊 No se solicitó el reporte científico de clasificación.',
    "   🔬 Cohen's Kappa: {0:.4f} ({1})":
        '   🔬 Kappa de Cohen: {0:.4f} ({1})',
    '   ⚖️ Balanced Accuracy: {0:.4f}':
        '   ⚖️ Exactitud balanceada: {0:.4f}',
    '   🔄 Cross-validation: {0:.4f} ± {1:.4f}':
        '   🔄 Validación cruzada: {0:.4f} ± {1:.4f}',
    'Performing {0}-fold cross-validation...':
        'Realizando validación cruzada de {0} particiones...',
    'Cross-validation results: {0:.4f} ± {1:.4f}':
        'Resultados de la validación cruzada: {0:.4f} ± {1:.4f}',
    'Individual fold scores: {0}':
        'Puntuación de cada partición: {0}',
    'Could not open raster dataset with GDAL':
        'No se pudo abrir el ráster con GDAL',
    'Reading band {0} of {1}...':
        'Leyendo la banda {0} de {1}...',
    'Error reading raster data: {0}':
        'Error al leer los datos ráster: {0}',
    'Failed to read raster data. Try using a smaller image or a different format.':
        'No se pudieron leer los datos ráster. Pruebe con una imagen más pequeña o con otro formato.',
    'Error reshaping data: {0}':
        'Error al reorganizar los datos: {0}',
    'Failed to reshape raster data: {0}':
        'No se pudieron reorganizar los datos ráster: {0}',
    '📊 Band {0} importance: {1:.4f}':
        '📊 Importancia de la banda {0}: {1:.4f}',
    '🔄 Using Gaussian Mixture Model':
        '🔄 Usando el modelo de mezcla gaussiana',
    '⚠️ Advanced metrics unavailable - install full scikit-learn':
        '⚠️ Métricas avanzadas no disponibles: instale scikit-learn completo',
    'Error during classification: {0}':
        'Error durante la clasificación: {0}',
    'Classification failed: {0}. The image may be too large or complex.':
        'La clasificación falló: {0}. La imagen puede ser demasiado grande o compleja.',
    'Error reshaping classification result: {0}':
        'Error al reorganizar el resultado de la clasificación: {0}',
    'Failed to format classification result: {0}':
        'No se pudo dar formato al resultado de la clasificación: {0}',
    'Failed to create output raster at {0}':
        'No se pudo crear el ráster de salida en {0}',
    'Error saving output raster: {0}':
        'Error al guardar el ráster de salida: {0}',
    'Failed to save output raster: {0}':
        'No se pudo guardar el ráster de salida: {0}',
    '📊 Generating comprehensive scientific classification report: {0}':
        '📊 Generando el reporte científico completo de clasificación: {0}',
    '📊 Report generation was requested but no output path was specified.':
        '📊 Se solicitó el reporte, pero no se indicó una ruta de salida.',
    '   (skipped optional metrics: {0})':
        '   (se omitieron métricas opcionales: {0})',
    'Cross-validation failed: {0}':
        'La validación cruzada falló: {0}',
    'Could not identify raster values at point {0}':
        'No se pudieron identificar los valores del ráster en el punto {0}',
    'Error processing point {0}: {1}':
        'Error al procesar el punto {0}: {1}',
    'Failed to read band {0} - it may be too large or corrupted.':
        'No se pudo leer la banda {0}: puede ser demasiado grande o estar dañada.',
    '⚡ Using Support Vector Machine':
        '⚡ Usando la máquina de vectores de soporte',
    '📊 Comprehensive scientific classification report created successfully!':
        '📊 ¡Reporte científico completo de clasificación creado correctamente!',
    'Skipping point with NULL classification value at {0}':
        'Se omite un punto con valor de clasificación NULL en {0}',
    '🎯 Using K-Nearest Neighbors':
        '🎯 Usando K vecinos más cercanos',
    '🌐 Report opened in default web browser.':
        '🌐 Reporte abierto en el navegador predeterminado.',
    '⚠️ Error generating classification report: {0}':
        '⚠️ Error al generar el reporte de clasificación: {0}',
    '✅ Classification completed successfully, but the report could not be generated.':
        '✅ La clasificación se completó correctamente, pero no se pudo generar el reporte.',
    'Band {0} not found in raster results at {1}':
        'No se encontró la banda {0} en los resultados del ráster en {1}',
    '📄 Report saved successfully. You can open it manually in your web browser.':
        '📄 Reporte guardado correctamente. Puede abrirlo manualmente en su navegador.',
    'Skipping point with NULL band value at {0}':
        'Se omite un punto con valor de banda NULL en {0}',
    '\n        This algorithm exports the attributes of a vector layer to CSV format.\n        \n        It offers two output options:\n        1. Standard CSV: A regular comma-separated values file.\n        2. Excel compatible CSV: A CSV file formatted to be easily opened in Excel.\n        \n        The tool will export all attributes of the input layer, including the feature ID.\n        Geometry information is not included in the output.\n    \n        Note: This export uses UTF-8 encoding with BOM for better compatibility with Excel.\n        ':
        '\n        Este algoritmo exporta los atributos de una capa vectorial a formato CSV.\n\n        Ofrece dos opciones de salida:\n        1. CSV estándar: archivo normal de valores separados por comas.\n        2. CSV compatible con Excel: archivo CSV con formato para abrirse fácilmente en Excel.\n\n        La herramienta exporta todos los atributos de la capa de entrada, incluido el ID de la entidad.\n        La información de geometría no se incluye en la salida.\n\n        Nota: esta exportación usa codificación UTF-8 con BOM para mejor compatibilidad con Excel.\n        ',
    'Output format':
        'Formato de salida',
    'Excel compatible CSV':
        'CSV compatible con Excel',
    'Output file':
        'Archivo de salida',
    'Successfully exported {0} features to {1}':
        'Se exportaron correctamente {0} entidades a {1}',
    'Error writing CSV file: {0}':
        'Error al escribir el archivo CSV: {0}',
    'Extract Ordered Points with Bi-directional Numbering':
        'Extraer puntos ordenados con numeración bidireccional',
    "\n        This algorithm extracts ordered points from the vertices of input polygons and provides bi-directional numbering.\n\n        Polygon ID Field: A polygon with an identifying field is required.\n        \n        Starting Coordinate (optional): Click on the map to select a coordinate. The algorithm will use the vertex closest to this point as the starting point for numbering. If not provided, the northernmost point is used.\n        \n        Features:\n        1. Extracts unique points from each polygon's vertices.\n        2. Orders the points starting from a specified coordinate or the northernmost point.\n        3. Assigns each point both clockwise (CW) and counter-clockwise (CCW) IDs, both starting from 1.\n        4. Provides the Polygon_ID (from the selected field).\n\n        The Polygon ID Field is crucial when processing multiple polygons, as it allows you to identify which points in the output belong to which input polygon.\n        \n        Use this tool when you need to:\n        - Convert polygon boundaries to point features with bi-directional numbering\n        - Analyze or process polygon vertices as individual points with flexible ordering\n        - Create input for other point-based algorithms requiring ordinal information\n        ":
        '\n        Este algoritmo extrae puntos ordenados de los vértices de los polígonos de entrada y les asigna numeración bidireccional.\n\n        Campo ID del polígono: se requiere un polígono con un campo identificador.\n\n        Coordenada inicial (opcional): haga clic en el mapa para elegir una coordenada. El algoritmo usará el vértice más cercano a ese punto como inicio de la numeración. Si no se indica, se usa el punto más al norte.\n\n        Características:\n        1. Extrae los puntos únicos de los vértices de cada polígono.\n        2. Ordena los puntos desde la coordenada indicada o desde el punto más al norte.\n        3. Asigna a cada punto un ID en sentido horario (CW) y otro en sentido antihorario (CCW), ambos desde 1.\n        4. Incluye el Polygon_ID (del campo seleccionado).\n\n        El campo ID del polígono es clave al procesar varios polígonos, porque permite saber a qué polígono de entrada pertenece cada punto de la salida.\n\n        Use esta herramienta cuando necesite:\n        - Convertir límites de polígonos en puntos con numeración bidireccional\n        - Analizar o procesar los vértices de polígonos como puntos individuales con un orden flexible\n        - Crear la entrada para otros algoritmos de puntos que requieren información de orden\n        ',
    'Polygon ID Field':
        'Campo ID del polígono',
    'Starting coordinate':
        'Coordenada inicial',
    'Output points':
        'Puntos de salida',
    '\n        Calculates Curve Number using global datasets:\n        - ESA WorldCover 2021 for land cover\n        - ORNL HYSOG for hydrologic soil groups (tiled system)\n        \n        Parameters:\n        - Study Area: Polygon of the area to calculate CN for\n        - Hydrologic Condition: Poor, Fair (default), or Good\n        - Antecedent Runoff Condition: I (dry), II (normal, default), or III (wet)\n            \n        Outputs:\n        - ESA Land Cover (optional): Land cover classification raster\n        - Hydrologic Soil Groups (optional): Soil groups raster\n        - Curve Number: Final CN raster\n        \n        The tool will also display statistics including mean, minimum, maximum,\n        and standard deviation of the calculated CN values.\n            \n        Data Sources:\n        - Land Cover: ESA WorldCover 2021 (10m resolution)\n        - Soil Groups: ORNL HYSOG Global Hydrologic Soil Groups (tiled system)\n        \n        Note: Internet connection required to download soil data tiles.\n        Processing time depends on the size of your study area and number of tiles needed.\n\n        This tool is based on the "Curve Number Generator" plugin by Abdul Raheem.\n        ':
        '\n        Calcula el número de curva con datos globales:\n        - ESA WorldCover 2021 para la cobertura del suelo\n        - ORNL HYSOG para los grupos hidrológicos de suelo (sistema de teselas)\n\n        Parámetros:\n        - Área de estudio: polígono del área donde se calcula el CN\n        - Condición hidrológica: mala, regular (por defecto) o buena\n        - Condición de humedad antecedente: I (seca), II (normal, por defecto) o III (húmeda)\n\n        Salidas:\n        - Cobertura del suelo ESA (opcional): ráster de clasificación de cobertura\n        - Grupos hidrológicos de suelo (opcional): ráster de grupos de suelo\n        - Número de curva: ráster final de CN\n\n        La herramienta también muestra estadísticas como media, mínimo, máximo\n        y desviación estándar de los valores de CN calculados.\n\n        Fuentes de datos:\n        - Cobertura del suelo: ESA WorldCover 2021 (resolución de 10 m)\n        - Grupos de suelo: ORNL HYSOG, grupos hidrológicos de suelo globales (sistema de teselas)\n\n        Nota: se requiere conexión a internet para descargar las teselas de suelo.\n        El tiempo de proceso depende del tamaño del área de estudio y del número de teselas necesarias.\n\n        Esta herramienta se basa en el complemento "Curve Number Generator" de Abdul Raheem.\n        ',
    'Starting Curve Number calculation...':
        'Iniciando el cálculo del número de curva...',
    'Hydrologic Condition':
        'Condición hidrológica',
    'Antecedent Runoff Condition':
        'Condición de humedad antecedente',
    'ESA Land Cover':
        'Cobertura del suelo ESA',
    'Hydrologic Soil Groups':
        'Grupos hidrológicos de suelo',
    'Curve Number':
        'Número de curva',
    'Invalid study area':
        'Área de estudio no válida',
    'Reprojecting study area to EPSG:4326...':
        'Reproyectando el área de estudio a EPSG:4326...',
    'ESA WorldCover VRT file not found':
        'No se encontró el archivo VRT de ESA WorldCover',
    'Step 1/6: Processing ESA WorldCover data...':
        'Paso 1/6: procesando los datos de ESA WorldCover...',
    'Step 2/6: Processing ORNL HYSOG data from tiles...':
        'Paso 2/6: procesando los datos ORNL HYSOG desde las teselas...',
    'Step 3/6: Aligning datasets...':
        'Paso 3/6: alineando los conjuntos de datos...',
    'Step 4/6: Calculating initial Curve Number...':
        'Paso 4/6: calculando el número de curva inicial...',
    'Step 5/6: Clipping final Curve Number to study area...':
        'Paso 5/6: recortando el número de curva final al área de estudio...',
    'Step 6/6: Calculating statistics...':
        'Paso 6/6: calculando estadísticas...',
    'Invalid URL scheme: {0}':
        'Esquema de URL no válido: {0}',
    'Downloading tiles information...':
        'Descargando la información de las teselas...',
    'Found {0} required tiles: {1}':
        'Se encontraron {0} teselas necesarias: {1}',
    'Clipping HYSOG data to study area...':
        'Recortando los datos HYSOG al área de estudio...',
    'Processing NoData and invalid soil group values...':
        'Procesando valores NoData y grupos de suelo no válidos...',
    '\n=== Curve Number Statistics ===':
        '\n=== Estadísticas del número de curva ===',
    'Mean CN: {0:.2f}':
        'CN medio: {0:.2f}',
    'S = Maximum potential retention (mm): {0:.2f}':
        'S = Retención potencial máxima (mm): {0:.2f}',
    'Primary server unavailable ({0}), switching to GitHub fallback{1}...':
        'Servidor principal no disponible ({0}), se usará el respaldo de GitHub{1}...',
    'No HYSOG tiles found for the study area':
        'No se encontraron teselas HYSOG para el área de estudio',
    'Downloading tile {0}/{1}: {2}':
        'Descargando tesela {0}/{1}: {2}',
    'Merging multiple tiles...':
        'Uniendo varias teselas...',
    'Error downloading HYSOG tiles (both primary and fallback failed): {0}':
        'Error al descargar las teselas HYSOG (fallaron el servidor principal y el de respaldo): {0}',
    'Error processing HYSOG data: {0}':
        'Error al procesar los datos HYSOG: {0}',
    'No valid CN values found in {0}':
        'No se encontraron valores de CN válidos en {0}',
    'Error reading CN values from {0}: {1}':
        'Error al leer los valores de CN de {0}: {1}',
    'Could not open output raster for statistics':
        'No se pudo abrir el ráster de salida para calcular estadísticas',
    'Error calculating statistics: {0}':
        'Error al calcular estadísticas: {0}',
    'Create point marker':
        'Crear marcador de punto',
    'Go':
        'Ir',
    'Capture Coordinate':
        'Capturar coordenada',
    'Clear History':
        'Borrar historial',
    'Copy Coordinates':
        'Copiar coordenadas',
    'Coordinate Type:':
        'Tipo de coordenada:',
    'Input CRS:':
        'SRC de entrada:',
    'History:':
        'Historial:',
    'Longitude:':
        'Longitud:',
    'Latitude:':
        'Latitud:',
    'Young stage':
        'Etapa de juventud',
    'Mature stage':
        'Etapa de madurez',
    'Old stage':
        'Etapa de vejez',
    'Hypsometric Curve (HI = {0:.3f} - {1}, Area = {2:.2f} km²)':
        'Curva hipsométrica (HI = {0:.3f} - {1}, Área = {2:.2f} km²)',
    'Relative area (a/A)':
        'Área relativa (a/A)',
    'Relative height (h/H)':
        'Altura relativa (h/H)',
    'Area (km²)':
        'Área (km²)',
    'Elevation (m)':
        'Elevación (m)',
    'DEM dimensions: {0} x {1}':
        'Dimensiones del DEM: {0} x {1}',
    'Pixel resolution: {0:.3f} x {1:.3f}':
        'Resolución de píxel: {0:.3f} x {1:.3f}',
    'Total pixels processed: {0}':
        'Total de píxeles procesados: {0}',
    'Pixels in basin: {0}':
        'Píxeles dentro de la cuenca: {0}',
    'Valid elevation values: {0}':
        'Valores de elevación válidos: {0}',
    'Using most recent histogram file: {0}':
        'Usando el archivo de histograma más reciente: {0}',
    'Starting pixel-by-pixel hypsometric analysis...':
        'Iniciando el análisis hipsométrico píxel por píxel...',
    'Calculated Hypsometric Integral: {0:.3f} ({1})':
        'Integral hipsométrica calculada: {0:.3f} ({1})',
    'Processed data saved to: {0}':
        'Datos procesados guardados en: {0}',
    'Hypsometric curve analysis completed. Results in: {0}':
        'Análisis de la curva hipsométrica completado. Resultados en: {0}',
    'Hypsometric Curve':
        'Curva hipsométrica',
    'No basin geometry found':
        'No se encontró la geometría de la cuenca',
    'Insufficient elevation data for hypsometric analysis':
        'Datos de elevación insuficientes para el análisis hipsométrico',
    'Error extracting pixel values: {0}':
        'Error al extraer los valores de píxel: {0}',
    'No histogram CSV file found in the output directory.':
        'No se encontró el archivo CSV del histograma en la carpeta de salida.',
    'Error in generate_hypsometric_curve: {0}':
        'Error en generate_hypsometric_curve: {0}',
    'Failed to calculate hypsometric integral':
        'No se pudo calcular la integral hipsométrica',
    'Interactive hypsometric curve (HTML) generated.':
        'Se generó la curva hipsométrica interactiva (HTML).',
    'Plotly is not installed; interactive version not generated.':
        'Plotly no está instalado; no se generó la versión interactiva.',
    'Data points':
        'Puntos de datos',
    'Elevation Distribution':
        'Distribución de elevaciones',
    'Area: %{x:.2f} km² (%{customdata:.1f}%)<br>Elevation: %{y:.1f} m<extra></extra>':
        'Área: %{x:.2f} km² (%{customdata:.1f}%)<br>Elevación: %{y:.1f} m<extra></extra>',
    'Error processing CSV file: {0}':
        'Error al procesar el archivo CSV: {0}',
    'Performs kriging spatial interpolation on point data.\n        \n        Kriging is a geostatistical technique that interpolates the value of a field at an unobserved location from observations at nearby locations, weighted according to spatial covariance values.\n        \n        Parameters:\n        - Input point layer: Vector layer containing points with values to interpolate.\n        - Value field: Field containing the numeric values to interpolate.\n        - Output cell size: Size of the pixels in the output raster (in layer units).\n        - Output extent: Extent of the output raster layer.\n        - Variogram model: Mathematical function that describes spatial correlation in the data.\n        - Kriging type: Ordinary or Universal kriging. Universal kriging includes drift components.\n        - Drift type: Drift components to include in Universal kriging.\n        - Include error estimation: Generate an additional raster showing kriging variance.\n        - Min value (optional): Minimum value allowed in output (leave empty for no limit).\n        - Max value (optional): Maximum value allowed in output (leave empty for no limit).\n        \n        Outputs:\n        - Output interpolated raster: Result of kriging interpolation in GeoTIFF format.\n        - Output kriging variance (optional): Estimation error/uncertainty in GeoTIFF format.\n        \n        Requirements:\n        - This tool requires the numpy, pykrige and scipy Python packages.\n        - If they are not installed, the tool will provide installation instructions.\n        ':
        'Realiza una interpolación espacial kriging sobre datos de puntos.\n\n        El kriging es una técnica geoestadística que interpola el valor de una variable en un lugar no observado a partir de observaciones cercanas, ponderadas según la covarianza espacial.\n\n        Parámetros:\n        - Capa de puntos de entrada: capa vectorial con los puntos y valores a interpolar.\n        - Campo de valores: campo con los valores numéricos a interpolar.\n        - Tamaño de celda de salida: tamaño de los píxeles del ráster de salida (en unidades de la capa).\n        - Extensión de salida: extensión del ráster de salida.\n        - Modelo de variograma: función matemática que describe la correlación espacial de los datos.\n        - Tipo de kriging: ordinario o universal. El kriging universal incluye componentes de deriva.\n        - Tipo de deriva: componentes de deriva a incluir en el kriging universal.\n        - Incluir estimación del error: genera un ráster adicional con la varianza del kriging.\n        - Valor mínimo (opcional): valor mínimo permitido en la salida (vacío para no limitar).\n        - Valor máximo (opcional): valor máximo permitido en la salida (vacío para no limitar).\n\n        Salidas:\n        - Ráster interpolado: resultado de la interpolación kriging en formato GeoTIFF.\n        - Varianza del kriging (opcional): error/incertidumbre de la estimación en formato GeoTIFF.\n\n        Requisitos:\n        - Esta herramienta requiere los paquetes de Python numpy, pykrige y scipy.\n        - Si no están instalados, la herramienta mostrará las instrucciones de instalación.\n        ',
    'Using CRS: {0}':
        'Usando el SRC: {0}',
    'Output grid: {0} x {1} cells, cell size: {2}':
        'Malla de salida: {0} x {1} celdas, tamaño de celda: {2}',
    'Extent: ({0}, {1}) - ({2}, {3})':
        'Extensión: ({0}, {1}) - ({2}, {3})',
    'Using {0} points for interpolation':
        'Usando {0} puntos para la interpolación',
    'Value range: {0} to {1}':
        'Rango de valores: {0} a {1}',
    'Input point layer':
        'Capa de puntos de entrada',
    'Value field to interpolate':
        'Campo de valores a interpolar',
    'Output cell size (use larger values for projected CRS in meters, e.g. 100-1000)':
        'Tamaño de celda de salida (use valores mayores para SRC proyectados en metros, p. ej. 100-1000)',
    'Output extent (xmin,xmax,ymin,ymax)':
        'Extensión de salida (xmin,xmax,ymin,ymax)',
    'Variogram model':
        'Modelo de variograma',
    'Kriging type':
        'Tipo de kriging',
    'Drift type (only for Universal Kriging)':
        'Tipo de deriva (solo para kriging universal)',
    'Include error estimation':
        'Incluir estimación del error',
    'Minimum value (optional)':
        'Valor mínimo (opcional)',
    'Maximum value (optional)':
        'Valor máximo (opcional)',
    'Output interpolated raster':
        'Ráster interpolado de salida',
    'Output kriging variance':
        'Varianza del kriging de salida',
    'Minimum value limit will be applied: {0}':
        'Se aplicará el límite de valor mínimo: {0}',
    'Maximum value limit will be applied: {0}':
        'Se aplicará el límite de valor máximo: {0}',
    'Error estimation will be saved to: {0}':
        'La estimación del error se guardará en: {0}',
    'Using input layer extent':
        'Usando la extensión de la capa de entrada',
    'Invalid extent. Width and height must be greater than 0.':
        'Extensión no válida. El ancho y el alto deben ser mayores que 0.',
    'Invalid grid dimensions. Please check cell size and extent.':
        'Dimensiones de malla no válidas. Revise el tamaño de celda y la extensión.',
    'The output grid is too large ({0} x {1} = {2:,} cells). This would require several gigabytes of RAM and will crash QGIS.\n\nPlease increase the cell size. For your current extent, the minimum recommended cell size is {3:.1f} (in the same units as your CRS).\n\nTip: For projected CRS (meters), use values like 500, 1000 or larger depending on your study area size.':
        'La malla de salida es demasiado grande ({0} x {1} = {2:,} celdas). Requeriría varios gigabytes de RAM y haría que QGIS se cierre.\n\nAumente el tamaño de celda. Para la extensión actual, el tamaño de celda mínimo recomendado es {3:.1f} (en las mismas unidades del SRC).\n\nSugerencia: para SRC proyectados (metros), use valores como 500, 1000 o mayores según el tamaño del área de estudio.',
    'Not enough valid points for kriging interpolation (minimum 3 required).':
        'No hay suficientes puntos válidos para la interpolación kriging (se requieren al menos 3).',
    'Starting kriging with {0} variogram model...':
        'Iniciando el kriging con el modelo de variograma {0}...',
    'Executing kriging interpolation...':
        'Ejecutando la interpolación kriging...',
    'Writing intermediate ASCII Grid...':
        'Escribiendo la malla ASCII intermedia...',
    'Converting to output raster format...':
        'Convirtiendo al formato ráster de salida...',
    'Using Universal Kriging with drift type: {0}':
        'Usando kriging universal con tipo de deriva: {0}',
    'Using Ordinary Kriging':
        'Usando kriging ordinario',
    'Main raster output saved to: {0}':
        'Ráster principal guardado en: {0}',
    'Processing error estimation raster...':
        'Procesando el ráster de estimación del error...',
    'Successfully cleaned up temporary directory: {0}':
        'Carpeta temporal eliminada correctamente: {0}',
    'Warning: Error during cleanup: {0}':
        'Advertencia: error durante la limpieza: {0}',
    'Applying limits: min={0}, max={1}':
        'Aplicando límites: mín={0}, máx={1}',
    'Values below min: {0}':
        'Valores por debajo del mínimo: {0}',
    'Values above max: {0}':
        'Valores por encima del máximo: {0}',
    'After clipping - Values below min: {0}':
        'Tras el recorte - valores por debajo del mínimo: {0}',
    'After clipping - Values above max: {0}':
        'Tras el recorte - valores por encima del máximo: {0}',
    'Error converting main raster: {0}':
        'Error al convertir el ráster principal: {0}',
    'Error raster output saved to: {0}':
        'Ráster de error guardado en: {0}',
    'Warning: Could not clean up temporary directory: {0}':
        'Advertencia: no se pudo eliminar la carpeta temporal: {0}',
    'Applying minimum limit: {0}':
        'Aplicando el límite mínimo: {0}',
    'After applying minimum - Values below min: {0}':
        'Tras aplicar el mínimo - valores por debajo del mínimo: {0}',
    'Error converting variance raster: {0}':
        'Error al convertir el ráster de varianza: {0}',
    'Applying maximum limit: {0}':
        'Aplicando el límite máximo: {0}',
    'After applying maximum - Values above max: {0}':
        'Tras aplicar el máximo - valores por encima del máximo: {0}',
    '\n        Calculates changes between two raster images representing land use/cover at different times.\n\n        Parameters:\n        - Input raster layer (before): Initial state\n        - Input raster layer (after): Final state\n        - Category to analyze: Specific category for gain/loss analysis (must exist in at least one raster)\n\n        Outputs:\n        1. Detailed change raster: Shows category transitions\n           - For ≤9 categories: value = (current year * 10) + previous year\n           - For >9 categories: value = (current year * 1000) + previous year\n\n        2. Simplified raster: Areas with and without changes\n        3. Gain raster: Where specified category was gained\n        4. Loss raster: Where specified category was lost\n\n        Note: Supports up to 99 categories with automatic symbology.\n        ':
        '\n        Calcula los cambios entre dos imágenes ráster de uso/cobertura del suelo en diferentes fechas.\n\n        Parámetros:\n        - Ráster de entrada (antes): estado inicial\n        - Ráster de entrada (después): estado final\n        - Categoría a analizar: categoría específica para el análisis de ganancias/pérdidas (debe existir en al menos uno de los rásteres)\n\n        Salidas:\n        1. Ráster de cambios detallado: muestra las transiciones entre categorías\n           - Para ≤9 categorías: valor = (año actual * 10) + año anterior\n           - Para >9 categorías: valor = (año actual * 1000) + año anterior\n\n        2. Ráster simplificado: zonas con y sin cambios\n        3. Ráster de ganancia: dónde se ganó la categoría indicada\n        4. Ráster de pérdida: dónde se perdió la categoría indicada\n\n        Nota: admite hasta 99 categorías con simbología automática.\n        ',
    'Actual categories detected: {0}':
        'Categorías detectadas: {0}',
    'Number of categories detected: {0}':
        'Número de categorías detectadas: {0}',
    'Input raster layer (before)':
        'Ráster de entrada (antes)',
    'Input raster layer (after)':
        'Ráster de entrada (después)',
    'Category to analyze':
        'Categoría a analizar',
    'Output detailed change raster':
        'Ráster de cambios detallado',
    'Output simplified change raster':
        'Ráster de cambios simplificado',
    'Output gain raster':
        'Ráster de ganancia',
    'Output loss raster':
        'Ráster de pérdida',
    'Invalid input layers':
        'Capas de entrada no válidas',
    'Selected category ({0}) does not exist in either raster. Available categories are: {1}.':
        'La categoría seleccionada ({0}) no existe en ninguno de los rásteres. Categorías disponibles: {1}.',
    'Processing formula: {0}':
        'Fórmula en proceso: {0}',
    'Successfully created: {0}':
        'Creado correctamente: {0}',
    'Custom symbology applied to {0}':
        'Simbología personalizada aplicada a {0}',
    'Failed to apply custom symbology to {0}':
        'No se pudo aplicar la simbología personalizada a {0}',
    'Custom detailed symbology applied to {0}':
        'Simbología detallada aplicada a {0}',
    'Failed to apply custom detailed symbology to {0}':
        'No se pudo aplicar la simbología detallada a {0}',
    'Error code: {0}':
        'Código de error: {0}',
    'Error calculating raster: {0}':
        'Error al calcular el ráster: {0}',
    'Fast Optimal Path Finder':
        'Búsqueda rápida de ruta óptima',
    "Calculates the optimal path between points based on a cost raster.\n        \n        Parameters:\n        - Cost Raster: Raster layer representing the cost of movement\n        - Starting Point: Origin of the path\n        - Intermediate Point (optional): Optional waypoint to pass through\n        - Ending Point: Destination of the path\n        \n        Advanced Options:\n        - Optimization Level: Pre-configured parameter sets for different scenarios\n          * Balanced: Good for general routing with reasonable balance of speed and accuracy\n          * Speed Focus: Faster calculation but may miss optimal routes in complex terrain\n          * Accuracy Focus: More thorough search for optimal paths in complex terrain\n          * Flat Terrain Focus: Specialized mode for capturing subtle variations in flat areas\n        \n        - Heuristic Weight (0.1-5.0): Controls path finding behavior\n          * Lower values (0.1-0.8): More thorough search, better for complex terrain with barriers\n          * Values around 1.0: Balanced approach for most scenarios\n          * Higher values (1.2-5.0): Faster computation but may miss optimal routes\n        \n        - Slope Influence (0.0-1.0): How much elevation changes affect path selection\n          * 0.0: Ignores terrain slopes completely\n          * 0.1-0.3: Minimal consideration, good for flat areas or where slope isn't critical\n          * 0.4-0.6: Moderate consideration, suitable for most scenarios\n          * 0.7-1.0: High sensitivity to slopes, ideal for hiking trails or drainage channels\n        \n        - Diagonal Movement Cost (0.5-2.0): Controls path shape and preference for diagonal movement\n          * Below 1.0: Encourages diagonal movement, resulting in more direct routes\n          * 1.0: Standard cost (diagonal = sqrt(2) times orthogonal)\n          * Above 1.0: Discourages diagonal movement, creates paths with more right angles\n          \n        - Flat Terrain Sensitivity (0.0-1.0): Controls how the algorithm responds to subtle terrain variations\n          * 0.0: Ignores micro-variations in flat terrain (straight lines through flat areas)\n          * 0.3-0.5: Moderate sensitivity to micro-terrain features (slight deviations in flat areas)\n          * 0.6-1.0: High sensitivity to even tiny variations (follows minor terrain undulations)\n          \n        Practical Applications:\n        - For hiking trails: Use Accuracy Focus with Slope Influence 0.7-0.9\n        - For drainage channels: Use Flat Terrain Focus with Slope Influence 0.8-1.0\n        - For irrigation channels: Use Flat Terrain Focus with Slope Influence 0.9 and Diagonal 1.2+\n        - For electric/utility lines: Use Speed Focus with Slope Influence 0.3-0.5\n        - For general access roads: Use Balanced mode with Slope Influence 0.5-0.7\n        - For detailed hydrological analysis: Use Flat Terrain Focus with Flat Sensitivity 0.9-1.0\n        ":
        'Calcula la ruta óptima entre puntos a partir de un ráster de costo.\n\n        Parámetros:\n        - Ráster de costo: ráster que representa el costo de desplazamiento\n        - Punto de inicio: origen de la ruta\n        - Punto intermedio (opcional): punto de paso opcional\n        - Punto final: destino de la ruta\n\n        Opciones avanzadas:\n        - Nivel de optimización: conjuntos de parámetros predefinidos para distintos escenarios\n          * Equilibrado: adecuado para rutas generales, con buen equilibrio entre velocidad y exactitud\n          * Prioridad velocidad: cálculo más rápido, pero puede perder rutas óptimas en terreno complejo\n          * Prioridad exactitud: búsqueda más exhaustiva de rutas óptimas en terreno complejo\n          * Prioridad terreno plano: modo especializado para captar variaciones sutiles en zonas planas\n\n        - Peso heurístico (0.1-5.0): controla el comportamiento de la búsqueda\n          * Valores bajos (0.1-0.8): búsqueda más exhaustiva, mejor en terreno complejo con barreras\n          * Valores cercanos a 1.0: enfoque equilibrado para la mayoría de casos\n          * Valores altos (1.2-5.0): cálculo más rápido, pero puede perder rutas óptimas\n\n        - Influencia de la pendiente (0.0-1.0): cuánto afectan los cambios de elevación a la elección de la ruta\n          * 0.0: ignora por completo las pendientes\n          * 0.1-0.3: consideración mínima, útil en zonas planas o donde la pendiente no es crítica\n          * 0.4-0.6: consideración moderada, adecuada para la mayoría de casos\n          * 0.7-1.0: alta sensibilidad a la pendiente, ideal para senderos o canales de drenaje\n\n        - Costo del movimiento diagonal (0.5-2.0): controla la forma de la ruta y la preferencia por moverse en diagonal\n          * Menor que 1.0: favorece el movimiento diagonal, con rutas más directas\n          * 1.0: costo estándar (diagonal = sqrt(2) veces el ortogonal)\n          * Mayor que 1.0: desfavorece el movimiento diagonal, con rutas de más ángulos rectos\n\n        - Sensibilidad al terreno plano (0.0-1.0): controla la respuesta a variaciones sutiles del terreno\n          * 0.0: ignora las microvariaciones en terreno plano (líneas rectas en zonas planas)\n          * 0.3-0.5: sensibilidad moderada al microrrelieve (ligeras desviaciones en zonas planas)\n          * 0.6-1.0: alta sensibilidad incluso a variaciones mínimas (sigue las ondulaciones menores)\n\n        Aplicaciones prácticas:\n        - Senderos: use Prioridad exactitud con influencia de la pendiente 0.7-0.9\n        - Canales de drenaje: use Prioridad terreno plano con influencia de la pendiente 0.8-1.0\n        - Canales de riego: use Prioridad terreno plano con influencia de la pendiente 0.9 y diagonal 1.2+\n        - Líneas eléctricas/servicios: use Prioridad velocidad con influencia de la pendiente 0.3-0.5\n        - Caminos de acceso: use el modo Equilibrado con influencia de la pendiente 0.5-0.7\n        - Análisis hidrológico detallado: use Prioridad terreno plano con sensibilidad al terreno plano 0.9-1.0\n        ',
    'Matrix shape: {0}x{1}':
        'Forma de la matriz: {0}x{1}',
    'Valid cells: {0}':
        'Celdas válidas: {0}',
    'Cost range: {0} to {1}':
        'Rango de costo: {0} a {1}',
    'Median cost: {0}, IQR: {1}':
        'Costo mediano: {0}, RIC: {1}',
    'Flat terrain threshold: {0}':
        'Umbral de terreno plano: {0}',
    'Initialization took {0:.2f} seconds':
        'La inicialización tardó {0:.2f} segundos',
    'Direct distance: {0:.1f} cells':
        'Distancia directa: {0:.1f} celdas',
    'Max iterations set to: {0}':
        'Máximo de iteraciones: {0}',
    'Start cost: {0}':
        'Costo inicial: {0}',
    'End cost: {0}':
        'Costo final: {0}',
    'Cells visited: {0}':
        'Celdas visitadas: {0}',
    'Pathfinding took {0:.2f} seconds':
        'La búsqueda de la ruta tardó {0:.2f} segundos',
    'Cost Raster':
        'Ráster de costo',
    'Starting Point':
        'Punto de inicio',
    'Intermediate Point':
        'Punto intermedio',
    'Ending Point':
        'Punto final',
    'Optimization Level':
        'Nivel de optimización',
    'Balanced':
        'Equilibrado',
    'Speed Focus':
        'Prioridad velocidad',
    'Accuracy Focus':
        'Prioridad exactitud',
    'Flat Terrain Focus':
        'Prioridad terreno plano',
    'Heuristic Weight (1.0 = balanced)':
        'Peso heurístico (1.0 = equilibrado)',
    'Slope Influence (0.0 = ignore slopes)':
        'Influencia de la pendiente (0.0 = ignorar pendientes)',
    'Diagonal Movement Cost (1.0 = normal)':
        'Costo del movimiento diagonal (1.0 = normal)',
    'Flat Terrain Sensitivity (0.0 = ignore, 1.0 = high)':
        'Sensibilidad al terreno plano (0.0 = ignorar, 1.0 = alta)',
    'Output Optimal Path':
        'Ruta óptima de salida',
    'Step 1/{0}: Preparing cost matrix...':
        'Paso 1/{0}: preparando la matriz de costo...',
    'Raster loading took {0:.2f} seconds':
        'La carga del ráster tardó {0:.2f} segundos',
    'Step 2/{0}: Finding optimal path...':
        'Paso 2/{0}: buscando la ruta óptima...',
    'Start point: ({0}, {1}) -> cell ({2}, {3})':
        'Punto de inicio: ({0}, {1}) -> celda ({2}, {3})',
    'End point: ({0}, {1}) -> cell ({2}, {3})':
        'Punto final: ({0}, {1}) -> celda ({2}, {3})',
    'Step 3/{0}: Creating output geometry...':
        'Paso 3/{0}: creando la geometría de salida...',
    'Path calculation complete!':
        '¡Cálculo de la ruta completado!',
    'Path length: {0:.2f} meters':
        'Longitud de la ruta: {0:.2f} metros',
    'Total cost: {0:.2f}':
        'Costo total: {0:.2f}',
    'Path segments: {0}':
        'Segmentos de la ruta: {0}',
    'Flat terrain sensitivity: {0:.2f}':
        'Sensibilidad al terreno plano: {0:.2f}',
    'Invalid start or end position':
        'Posición de inicio o final no válida',
    'Search terminated after {0} iterations':
        'Búsqueda detenida tras {0} iteraciones',
    'No path found - all possible routes exhausted':
        'No se encontró ruta: se agotaron todas las rutas posibles',
    'Invalid Cost Raster':
        'Ráster de costo no válido',
    'Intermediate point: ({0}, {1}) -> cell ({2}, {3})':
        'Punto intermedio: ({0}, {1}) -> celda ({2}, {3})',
    'Finding path from start to intermediate point...':
        'Buscando la ruta del inicio al punto intermedio...',
    'Finding path from intermediate to end point...':
        'Buscando la ruta del punto intermedio al final...',
    'Finding direct path from start to end...':
        'Buscando la ruta directa del inicio al final...',
    'Failed to create output layer':
        'No se pudo crear la capa de salida',
    'Path must contain at least 2 points to form a line.':
        'La ruta debe tener al menos 2 puntos para formar una línea.',
    'Error in path calculation: {0}':
        'Error en el cálculo de la ruta: {0}',
    'Path found after {0} iterations':
        'Ruta encontrada tras {0} iteraciones',
    'Path length: {0} cells':
        'Longitud de la ruta: {0} celdas',
    'Total cells visited: {0}':
        'Total de celdas visitadas: {0}',
    'Initial path to intermediate point failed. Trying with different parameters...':
        'Falló la ruta inicial al punto intermedio. Probando con otros parámetros...',
    'Initial path from intermediate to end failed. Trying with different parameters...':
        'Falló la ruta inicial del punto intermedio al final. Probando con otros parámetros...',
    'Initial path finding failed. Trying with different parameters...':
        'Falló la búsqueda inicial de la ruta. Probando con otros parámetros...',
    'Corrected geometry is empty.':
        'La geometría corregida está vacía.',
    'Error calculating length: {0}':
        'Error al calcular la longitud: {0}',
    'No path found between start and intermediate points. Try adjusting parameters or positions.':
        'No se encontró ruta entre el inicio y el punto intermedio. Ajuste los parámetros o las posiciones.',
    'No path found between intermediate and end points. Try adjusting parameters or positions.':
        'No se encontró ruta entre el punto intermedio y el final. Ajuste los parámetros o las posiciones.',
    'Trying alternate parameter set {0}...':
        'Probando el conjunto de parámetros alternativo {0}...',
    'All alternative path finding failed. Trying simple line approach...':
        'Fallaron todas las búsquedas alternativas. Probando con una línea simple...',
    'Alternate parameter set {0} found a path!':
        '¡El conjunto de parámetros alternativo {0} encontró una ruta!',
    '\n        This algorithm orders points extracted from a line network, starting from a specified point.\n        It numbers each point once, following the connectivity of the lines using either DFS (Depth-First Search) or BFS (Breadth-First Search).\n\n        Parameters:\n        - Input line layer: The layer containing the network lines.\n        - Select the Start Point: Click on the map to set the starting point for the numbering.\n        - Method: Choose the numbering method (DFS or BFS).\n\n        Outputs:\n        - A point layer with each vertex from the input lines, including:\n          * Order: Numbering from the start point outwards using the selected method.\n\n        Note: The numbering starts exactly from the selected start point or the closest point if the selected point is not exactly on a vertex.\n        ':
        '\n        Este algoritmo ordena los puntos extraídos de una red de líneas a partir de un punto indicado.\n        Numera cada punto una sola vez siguiendo la conectividad de las líneas, con DFS (búsqueda en profundidad) o BFS (búsqueda en anchura).\n\n        Parámetros:\n        - Capa de líneas de entrada: capa con las líneas de la red.\n        - Seleccionar el punto de inicio: haga clic en el mapa para fijar el punto de inicio de la numeración.\n        - Método: elija el método de numeración (DFS o BFS).\n\n        Salidas:\n        - Una capa de puntos con cada vértice de las líneas de entrada, que incluye:\n          * Order: numeración desde el punto de inicio hacia afuera según el método elegido.\n\n        Nota: la numeración empieza exactamente en el punto de inicio elegido, o en el punto más cercano si el elegido no coincide con un vértice.\n        ',
    'Select the Start Point':
        'Seleccionar el punto de inicio',
    'Select the method':
        'Seleccionar el método',
    'DFS (Depth-First Search)':
        'DFS (búsqueda en profundidad)',
    'BFS (Breadth-First Search)':
        'BFS (búsqueda en anchura)',
    'Ordered points':
        'Puntos ordenados',
    '\n        <h3>Multiple Basin Delineation by Points</h3>\n        \n        <p>This tool performs a complete watershed analysis workflow in one step:</p>\n        \n        <ol>\n        <li><b>Generates stream network</b> from DEM using flow accumulation</li>\n        <li><b>Snaps input points</b> to the nearest stream network locations</li>\n        <li><b>Delineates watersheds</b> using snapped points as pour points</li>\n        <li><b>Calculates drainage areas</b> in square miles for each basin</li>\n        </ol>\n        \n        <h4>Parameters:</h4>\n        <ul>\n        <li><b>Input DEM:</b> Digital Elevation Model (30m resolution recommended)</li>\n        <li><b>Input Point Shapefile:</b> Point features to use as watershed pour points</li>\n        <li><b>Flow Accumulation Threshold:</b> Minimum cells to form a stream (default: 50)</li>\n        <li><b>Smooth Iterations:</b> Number of smoothing passes (default: 1)</li>\n        <li><b>Smooth Offset:</b> Smoothing offset value (default: 0.25)</li>\n        <li><b>Process All Points:</b> Whether to process all points or limit the number</li>\n        <li><b>Maximum Points:</b> If not processing all, maximum number of points to process (default: 5)</li>\n        </ul>\n        \n        <h4>Outputs:</h4>\n        <ul>\n        <li><b>Global Stream Network:</b> Generated stream network with Strahler/Shreve orders</li>\n        <li><b>Snapped Pour Points:</b> Input points snapped to stream network for watershed delineation</li>\n        <li><b>Watershed Basins:</b> Delineated basins with drainage areas in square miles</li>\n        <li><b>Local Stream Network:</b> Stream network clipped to each basin</li>\n        </ul>\n        \n        <p><i>Note: This tool requires GRASS GIS to be properly configured in QGIS.</i></p>\n        \n        <p><b>Developed by the Anthropocene Engineering team (anthroeng.com)</b></p>\n        ':
        '\n        <h3>Delimitación de múltiples cuencas por puntos</h3>\n\n        <p>Esta herramienta realiza en un solo paso un flujo completo de análisis de cuencas:</p>\n\n        <ol>\n        <li><b>Genera la red de drenaje</b> a partir del DEM mediante la acumulación de flujo</li>\n        <li><b>Ajusta los puntos de entrada</b> a la ubicación más cercana de la red de drenaje</li>\n        <li><b>Delimita las cuencas</b> usando los puntos ajustados como puntos de salida</li>\n        <li><b>Calcula las áreas de drenaje</b> en millas cuadradas de cada cuenca</li>\n        </ol>\n\n        <h4>Parámetros:</h4>\n        <ul>\n        <li><b>DEM de entrada:</b> modelo digital de elevación (se recomienda resolución de 30 m)</li>\n        <li><b>Capa de puntos de entrada:</b> puntos que se usarán como puntos de salida de las cuencas</li>\n        <li><b>Umbral de acumulación de flujo:</b> mínimo de celdas para formar un cauce (por defecto: 50)</li>\n        <li><b>Iteraciones de suavizado:</b> número de pasadas de suavizado (por defecto: 1)</li>\n        <li><b>Desplazamiento de suavizado:</b> valor de desplazamiento del suavizado (por defecto: 0.25)</li>\n        <li><b>Procesar todos los puntos:</b> procesar todos los puntos o limitar su número</li>\n        <li><b>Máximo de puntos:</b> si no se procesan todos, número máximo de puntos a procesar (por defecto: 5)</li>\n        </ul>\n\n        <h4>Salidas:</h4>\n        <ul>\n        <li><b>Red de drenaje global:</b> red de drenaje generada con órdenes de Strahler/Shreve</li>\n        <li><b>Puntos de salida ajustados:</b> puntos de entrada ajustados a la red de drenaje para delimitar las cuencas</li>\n        <li><b>Cuencas:</b> cuencas delimitadas con sus áreas de drenaje en millas cuadradas</li>\n        <li><b>Red de drenaje local:</b> red de drenaje recortada a cada cuenca</li>\n        </ul>\n\n        <p><i>Nota: esta herramienta requiere que GRASS GIS esté configurado correctamente en QGIS.</i></p>\n\n        <p><b>Desarrollado por el equipo de Anthropocene Engineering (anthroeng.com)</b></p>\n        ',
    'Input DEM (30m recommended)':
        'DEM de entrada (se recomienda 30 m)',
    'Input Point Shapefile':
        'Capa de puntos de entrada',
    'Flow Accumulation Threshold':
        'Umbral de acumulación de flujo',
    'Smooth Iterations':
        'Iteraciones de suavizado',
    'Smooth Offset':
        'Desplazamiento de suavizado',
    'Process All Points':
        'Procesar todos los puntos',
    'Maximum Points to Process (if not processing all)':
        'Máximo de puntos a procesar (si no se procesan todos)',
    'Global Stream Network':
        'Red de drenaje global',
    'Snapped Pour Points':
        'Puntos de salida ajustados',
    'Watershed Basins (with Drainage Areas)':
        'Cuencas (con áreas de drenaje)',
    'Local Stream Network':
        'Red de drenaje local',
    'Starting complete watershed analysis workflow...':
        'Iniciando el flujo completo de análisis de cuencas...',
    'Using Flow Accumulation Threshold: {0}':
        'Usando el umbral de acumulación de flujo: {0}',
    'Step 1: Generating stream network...':
        'Paso 1: generando la red de drenaje...',
    'Step 2: Snapping input points to stream network...':
        'Paso 2: ajustando los puntos de entrada a la red de drenaje...',
    'Processing {0} pour points...':
        'Procesando {0} puntos de salida...',
    'Step 3: Running watershed basin delineation...':
        'Paso 3: delimitando las cuencas...',
    'Step 4: Calculating drainage areas...':
        'Paso 4: calculando las áreas de drenaje...',
    'Preparing outputs...':
        'Preparando las salidas...',
    'Watershed analysis completed successfully!':
        '¡Análisis de cuencas completado correctamente!',
    'Limited to {0} points for processing':
        'Limitado a {0} puntos para el proceso',
    'Snapping points to stream network...':
        'Ajustando los puntos a la red de drenaje...',
    'Snapped {0} points to stream network':
        'Se ajustaron {0} puntos a la red de drenaje',
    'Calculated drainage areas for {0} basins':
        'Se calcularon las áreas de drenaje de {0} cuencas',
    'Invalid input DEM':
        'DEM de entrada no válido',
    'Invalid points layer':
        'Capa de puntos no válida',
    'No points found or snapping failed':
        'No se encontraron puntos o falló el ajuste',
    'Limiting processing to {0} points...':
        'Limitando el proceso a {0} puntos...',
    'Error in watershed analysis: {0}':
        'Error en el análisis de cuencas: {0}',
    'Input DEM is too large to process efficiently even after resampling.':
        'El DEM de entrada es demasiado grande para procesarse eficientemente, incluso tras remuestrearlo.',
    'Error generating stream network: {0}':
        'Error al generar la red de drenaje: {0}',
    'Invalid stream layer':
        'Capa de drenaje no válida',
    'Error calculating stream orders: {0}':
        'Error al calcular los órdenes de cauce: {0}',
    'Error resampling DEM: {0}':
        'Error al remuestrear el DEM: {0}',
    'Error limiting points: {0}':
        'Error al limitar los puntos: {0}',
    'Error snapping points to stream: {0}':
        'Error al ajustar los puntos al cauce: {0}',
    'Processing pour point ID: {0}':
        'Procesando el punto de salida ID: {0}',
    'Error in watershed delineation: {0}':
        'Error en la delimitación de la cuenca: {0}',
    'Error calculating drainage areas: {0}':
        'Error al calcular las áreas de drenaje: {0}',
    'Invalid stream layer type':
        'Tipo de capa de drenaje no válido',
    'DEM resampled to {0:.2f} units per pixel for processing.':
        'DEM remuestreado a {0:.2f} unidades por píxel para el proceso.',
    'Error clipping streams for basin {0}: {1}':
        'Error al recortar los cauces de la cuenca {0}: {1}',
    '\n        This algorithm divides parcels into lots of a specified width.\n\n        Parameters:\n        - Input polygon layer: The layer containing the parcels to be divided.\n        - Desired lot width: The width you want each resulting lot to have (in CRS units, requires a projected CRS).\n        - Target lot area: If greater than 0 (e.g. 200 for 200 m2), lots are sized by area instead of width: each cut is placed where the accumulated block area reaches the next lot, so lot widths adjust automatically where the block depth varies. With "Redistribute evenly" all lots in a row get exactly the same area, as close as possible to the target; corner lots absorb the adjustment in the other modes. The desired lot width is ignored when this is set.\n        - Merge threshold: Percentage of average lot area below which small leftover lots are merged into their neighbor.\n        - Remainder distribution: What to do when the frontage is not an exact multiple of the lot width. "Redistribute evenly" resizes all lots equally so no slivers are created (recommended, like Civil 3D\'s redistribute remainder); "Split between the two corner lots" makes both corner lots slightly wider; "Place in the last lot" keeps exact widths and accumulates the leftover at the end.\n        - Corner lot width: If greater than 0, the first and last lot of each row are reserved with this width. Corner lots are usually designed wider because they have setbacks on two streets, or to give them frontage on the side street. Use 0 to make corner lots the same width as the rest.\n        - Two rows of lots: If checked (default), the block is split along its centerline (rear lot line) so the lots in each row face their own street, instead of running from street to street. When frontage reference lines are provided, the split is automatic per parcel: it only happens if each row has its own street to front - parcels served by a single street keep one row of full-depth lots. Without reference lines, uncheck this for single-frontage parcels.\n        - Frontage reference lines (optional): A line layer (e.g. street fronts or street centerlines). When provided, lots are measured along the nearest frontage line and cut perpendicular to it - this is the recommended option for irregular or curved parcels.\n\n        How it works:\n        1. If "Two rows of lots" is checked, the block is split along its rear lot line: the line equidistant between the two street frontages when there are streets on both sides (both rows get the same depth), the frontage offset by half the block depth when there is a single street, or the long axis of the oriented minimum bounding box when no frontage lines are given.\n        2. For each row, a division axis is determined: its own nearest frontage line if provided, otherwise the long axis of the row\'s oriented minimum bounding box.\n        3. Cut lines are generated perpendicular to that axis: first the corner lots are reserved (if a corner lot width is given), then the interior lots are distributed according to the remainder mode. Along curved frontage lines, each cut follows the local perpendicular.\n        4. Each parcel is split independently with its own cut lines (neighboring parcels never affect each other).\n        5. Lots smaller than the merge threshold are merged into the neighbor (within the same row) sharing the longest boundary.\n\n        Note: Without frontage lines the tool works best on rectangular or convex blocks. For L-shaped or strongly curved blocks, provide frontage reference lines.\n        ':
        '\n        Este algoritmo divide parcelas en lotes de un ancho indicado.\n\n        Parámetros:\n        - Capa de polígonos de entrada: capa con las parcelas a dividir.\n        - Ancho de lote deseado: ancho que tendrá cada lote resultante (en unidades del SRC; requiere un SRC proyectado).\n        - Área objetivo del lote: si es mayor que 0 (p. ej. 200 para 200 m2), los lotes se dimensionan por área en lugar de por ancho: cada corte se ubica donde el área acumulada de la manzana alcanza el siguiente lote, así el ancho se ajusta solo donde varía el fondo de la manzana. Con "Redistribuir de forma pareja" todos los lotes de una fila tienen exactamente la misma área, lo más cerca posible del objetivo; en los otros modos los lotes esquineros absorben el ajuste. Cuando se usa, se ignora el ancho de lote deseado.\n        - Umbral de fusión: porcentaje del área promedio de lote bajo el cual los lotes sobrantes pequeños se unen a su vecino.\n        - Distribución del sobrante: qué hacer cuando el frente no es un múltiplo exacto del ancho de lote. "Redistribuir de forma pareja" ajusta todos los lotes por igual para no dejar franjas (recomendado, como la redistribución del sobrante de Civil 3D); "Repartir entre los dos lotes esquineros" ensancha un poco ambos lotes de esquina; "Colocar en el último lote" mantiene los anchos exactos y acumula el sobrante al final.\n        - Ancho de lote esquinero: si es mayor que 0, el primer y el último lote de cada fila se reservan con este ancho. Los lotes de esquina suelen diseñarse más anchos porque tienen retiros hacia dos calles, o para darles frente a la calle lateral. Use 0 para que tengan el mismo ancho que el resto.\n        - Dos filas de lotes: si está marcado (por defecto), la manzana se divide por su línea central (línea de fondo de los lotes) para que los lotes de cada fila den frente a su propia calle, en lugar de ir de calle a calle. Con líneas de frente de referencia, la división es automática por parcela: solo ocurre si cada fila tiene su propia calle de frente; las parcelas servidas por una sola calle mantienen una fila de lotes de fondo completo. Sin líneas de referencia, desmárquelo para parcelas con un solo frente.\n        - Líneas de frente de referencia (opcional): capa de líneas (p. ej. frentes o ejes de calles). Si se indica, los lotes se miden a lo largo de la línea de frente más cercana y se cortan perpendiculares a ella; es la opción recomendada para parcelas irregulares o curvas.\n\n        Cómo funciona:\n        1. Si "Dos filas de lotes" está marcado, la manzana se divide por su línea de fondo: la línea equidistante entre los dos frentes cuando hay calles a ambos lados (ambas filas con el mismo fondo), el frente desplazado la mitad del fondo cuando hay una sola calle, o el eje largo del rectángulo mínimo orientado cuando no hay líneas de frente.\n        2. Para cada fila se determina un eje de división: su línea de frente más cercana si se indica, o el eje largo del rectángulo mínimo orientado de la fila.\n        3. Las líneas de corte se generan perpendiculares a ese eje: primero se reservan los lotes esquineros (si se indica su ancho) y luego se distribuyen los lotes interiores según el modo de sobrante. En frentes curvos, cada corte sigue la perpendicular local.\n        4. Cada parcela se divide de forma independiente con sus propias líneas de corte (las parcelas vecinas nunca se afectan entre sí).\n        5. Los lotes menores que el umbral de fusión se unen al vecino (en la misma fila) con el que comparten el borde más largo.\n\n        Nota: sin líneas de frente, la herramienta funciona mejor en manzanas rectangulares o convexas. Para manzanas en L o muy curvas, indique líneas de frente de referencia.\n        ',
    'Input layer loaded. Feature count: {0}':
        'Capa de entrada cargada. Número de entidades: {0}',
    'Lot width: {0}':
        'Ancho de lote: {0}',
    'Target lot area: {0}':
        'Área objetivo del lote: {0}',
    'Merge threshold: {0}% of average area':
        'Umbral de fusión: {0}% del área promedio',
    'Remainder distribution: {0}':
        'Distribución del sobrante: {0}',
    'Corner lot width: {0}':
        'Ancho de lote esquinero: {0}',
    'Two rows of lots: {0}':
        'Dos filas de lotes: {0}',
    'Frontage reference lines: {0}':
        'Líneas de frente de referencia: {0}',
    'Processing completed. {0} parcels created.':
        'Proceso completado. Se crearon {0} lotes.',
    'Desired lot width':
        'Ancho de lote deseado',
    'Target lot area (0 = divide by width only)':
        'Área objetivo del lote (0 = dividir solo por ancho)',
    'Merge threshold (% of average area)':
        'Umbral de fusión (% del área promedio)',
    'Remainder distribution':
        'Distribución del sobrante',
    'Redistribute evenly (all lots equal width)':
        'Redistribuir de forma pareja (todos los lotes del mismo ancho)',
    'Split between the two corner lots':
        'Repartir entre los dos lotes esquineros',
    'Place in the last lot':
        'Colocar en el último lote',
    'Corner lot width (0 = same as lot width)':
        'Ancho de lote esquinero (0 = igual al ancho de lote)',
    'Two rows of lots (split block along its centerline)':
        'Dos filas de lotes (dividir la manzana por su línea central)',
    'Frontage reference lines (optional)':
        'Líneas de frente de referencia (opcional)',
    'Output divided parcels':
        'Parcelas divididas de salida',
    'Failed to load the input layer. Please check your input data.':
        'No se pudo cargar la capa de entrada. Revise sus datos de entrada.',
    'The input layer uses a geographic CRS (degrees). Reproject it to a projected CRS (e.g. UTM) so the lot width can be measured in meters.':
        'La capa de entrada usa un SRC geográfico (grados). Reproyéctela a un SRC proyectado (p. ej. UTM) para que el ancho de lote se mida en metros.',
    'Lot width must be greater than zero.':
        'El ancho de lote debe ser mayor que cero.',
    'Failed to create the output layer.':
        'No se pudo crear la capa de salida.',
    'Feature {0} has no geometry. Skipping.':
        'La entidad {0} no tiene geometría. Se omite.',
    'ArcGeek MCP Server':
        'Servidor MCP de ArcGeek',
    '<b>QGIS MCP Server</b><br><small>Powered by <i>nkarasiak/qgis-mcp</i> {0} &mdash; {1} commands</small>':
        '<b>Servidor MCP de QGIS</b><br><small>Basado en <i>nkarasiak/qgis-mcp</i> {0} &mdash; {1} comandos</small>',
    'Start Server':
        'Iniciar servidor',
    'Stop Server':
        'Detener servidor',
    'Server: Stopped':
        'Servidor: detenido',
    'How to connect (Claude / Antigravity / Cursor / OpenCode)':
        'Cómo conectar (Claude / Antigravity / Cursor / OpenCode)',
    'Connect Claude / Antigravity / Cursor / OpenCode -- MCP Setup':
        'Conectar Claude / Antigravity / Cursor / OpenCode -- Configuración MCP',
    '<b>Works with:</b> Claude Code, Claude Desktop, Antigravity, Cursor, OpenCode, and any MCP-compatible client.':
        '<b>Funciona con:</b> Claude Code, Claude Desktop, Antigravity, Cursor, OpenCode y cualquier cliente compatible con MCP.',
    'Copy':
        'Copiar',
    '<small>Already configured with an older ArcGeek version? Replace the <code>--from</code> value with the one above and restart your AI client.</small>':
        '<small>¿Ya lo configuró con una versión anterior de ArcGeek? Reemplace el valor de <code>--from</code> por el de arriba y reinicie su cliente de IA.</small>',
    'Copy JSON config':
        'Copiar configuración JSON',
    'Close':
        'Cerrar',
    'Server Port:':
        'Puerto del servidor:',
    '<br><b>Step 1 &mdash; Install uv on {0}:</b>':
        '<br><b>Paso 1 &mdash; Instale uv en {0}:</b>',
    '<br><b>Step 2 &mdash; Add this to your MCP config file:</b><br><code>{0}</code>':
        '<br><b>Paso 2 &mdash; Agregue esto a su archivo de configuración MCP:</b><br><code>{0}</code>',
    'Server: failed to start - {0}':
        'Servidor: no se pudo iniciar - {0}',
    'Server: Running on port {0}':
        'Servidor: en ejecución en el puerto {0}',
    'This algorithm calculates various two-band spectral indices from satellite imagery.\n\n1. Choose the satellite type.\n2. Select the index you want to calculate.\n3. Select the two raster bands required for the index.\n\nAvailable indices and their required bands:\n- NDVI (NIR, Red)\n- NDWI (Green, NIR)\n- NDBI (SWIR, NIR)\n- NDMI (NIR, SWIR)\n- MSI (SWIR, NIR)\n- NBR (NIR, SWIR2)\n- AVI (NIR, Red)\n- SAVI (NIR, Red)\n\nBand numbers for each satellite:\nSentinel-2: \n Green: 3, Red: 4, NIR: 8, SWIR: 11, SWIR2: 12\nLandsat-7: \n Green: 2, Red: 3, NIR: 4, SWIR: 5, SWIR2: 7\nLandsat-8/9: \n Green: 3, Red: 4, NIR: 5, SWIR: 6, SWIR2: 7\n\nPlease ensure you select the correct bands for the chosen satellite and index.\nNote: Input bands must have the same resolution.':
        'Este algoritmo calcula varios índices espectrales de dos bandas a partir de imágenes satelitales.\n\n1. Elija el tipo de satélite.\n2. Seleccione el índice que desea calcular.\n3. Seleccione las dos bandas ráster que requiere el índice.\n\nÍndices disponibles y bandas que requieren:\n- NDVI (NIR, Red)\n- NDWI (Green, NIR)\n- NDBI (SWIR, NIR)\n- NDMI (NIR, SWIR)\n- MSI (SWIR, NIR)\n- NBR (NIR, SWIR2)\n- AVI (NIR, Red)\n- SAVI (NIR, Red)\n\nNúmero de banda en cada satélite:\nSentinel-2:\n Green: 3, Red: 4, NIR: 8, SWIR: 11, SWIR2: 12\nLandsat-7:\n Green: 2, Red: 3, NIR: 4, SWIR: 5, SWIR2: 7\nLandsat-8/9:\n Green: 3, Red: 4, NIR: 5, SWIR: 6, SWIR2: 7\n\nAsegúrese de elegir las bandas correctas para el satélite y el índice elegidos.\nNota: las bandas de entrada deben tener la misma resolución.',
    'Satellite type':
        'Tipo de satélite',
    'Index type':
        'Tipo de índice',
    'Higher number band':
        'Banda de número mayor',
    'Lower number band':
        'Banda de número menor',
    'Output index raster':
        'Ráster del índice de salida',
    'Info Message':
        'Mensaje informativo',
    'Starting index calculation...':
        'Iniciando el cálculo del índice...',
    'Index calculation completed successfully.':
        'Cálculo del índice completado correctamente.',
    'Input bands have different resolutions. Please use bands with matching resolutions.':
        'Las bandas de entrada tienen resoluciones diferentes. Use bandas con la misma resolución.',
    'An error occurred during index calculation: {0}':
        'Ocurrió un error durante el cálculo del índice: {0}',
    'Browse':
        'Examinar',
    'Capture and Save':
        'Capturar y guardar',
    'Success':
        'Listo',
    'Screen capture saved as {0} and added to the view':
        'Captura de pantalla guardada como {0} y agregada a la vista',
    'Output File:':
        'Archivo de salida:',
    'Zoom Factor:':
        'Factor de zoom:',
    'Please specify an output file':
        'Indique un archivo de salida',
    'File already exists. Please choose a different name.':
        'El archivo ya existe. Elija otro nombre.',
    'Failed to add captured image to the view':
        'No se pudo agregar la imagen capturada a la vista',
    'Generates an interactive topographic profile from a line and a DEM, with optional points.\nParameters:\n - Input Line Layer: Vector line used to sample the DEM.\n - Digital Elevation Model (DEM): Raster layer with elevation data.\n - Points Layer (optional): Vector points to label on the profile.\n - Name Field for Points (optional): Field containing point names.\n - Maximum Distance for Points (meters): Only points within this distance from the line are considered.\n - Apply smoothing?: If enabled (default), a moving average is applied:\n      • No smoothing if total distance < 1000 m.\n      • Moderate smoothing (window = 3) if distance is between 1000 and 10000 m.\n      • Heavy smoothing (window = 7) if distance > 10 km.\n - Invert profile direction?: If enabled, the profile is drawn from end to start. (The DEM-derived\n      Z-values for points are always obtained directly from the DEM.)\n - Output HTML File: Path to save the interactive HTML output.\nAll layers must share the same CRS.':
        'Genera un perfil topográfico interactivo a partir de una línea y un DEM, con puntos opcionales.\nParámetros:\n - Capa de líneas de entrada: línea vectorial usada para muestrear el DEM.\n - Modelo digital de elevación (DEM): ráster con los datos de elevación.\n - Capa de puntos (opcional): puntos vectoriales a etiquetar en el perfil.\n - Campo de nombre de los puntos (opcional): campo con los nombres de los puntos.\n - Distancia máxima de los puntos (metros): solo se consideran los puntos a esta distancia de la línea.\n - ¿Aplicar suavizado?: si está activo (por defecto), se aplica una media móvil:\n      • Sin suavizado si la distancia total es < 1000 m.\n      • Suavizado moderado (ventana = 3) si la distancia está entre 1000 y 10000 m.\n      • Suavizado fuerte (ventana = 7) si la distancia es > 10 km.\n - ¿Invertir la dirección del perfil?: si está activo, el perfil se dibuja del final al inicio. (Los valores Z\n      de los puntos siempre se obtienen directamente del DEM.)\n - Archivo HTML de salida: ruta donde se guarda el HTML interactivo.\nTodas las capas deben tener el mismo SRC.',
    'Input Line Layer':
        'Capa de líneas de entrada',
    'Digital Elevation Model (DEM)':
        'Modelo digital de elevación (DEM)',
    'Points Layer':
        'Capa de puntos',
    'Name Field for Points':
        'Campo de nombre de los puntos',
    'Maximum Distance for Points (meters)':
        'Distancia máxima de los puntos (metros)',
    'Apply smoothing?':
        '¿Aplicar suavizado?',
    'Invert profile direction?':
        '¿Invertir la dirección del perfil?',
    'Output HTML File':
        'Archivo HTML de salida',
    'No input line layer specified':
        'No se indicó la capa de líneas de entrada',
    'Line layer CRS does not match DEM CRS. Reloading line layer with DEM CRS.':
        'El SRC de la capa de líneas no coincide con el del DEM. Se recarga la capa de líneas con el SRC del DEM.',
    'No features in line layer':
        'La capa de líneas no tiene entidades',
    'Line is outside DEM bounds':
        'La línea está fuera de los límites del DEM',
    'No points found in line geometry':
        'No se encontraron puntos en la geometría de la línea',
    'No valid elevations obtained':
        'No se obtuvieron elevaciones válidas',
    'Not all layers share the same CRS. Please reproject them.':
        'No todas las capas tienen el mismo SRC. Reproyéctelas.',
    'CRS mismatch among layers.':
        'Los SRC de las capas no coinciden.',
    'Warning: Invalid elevation at ({0:.3f}, {1:.3f})':
        'Advertencia: elevación no válida en ({0:.3f}, {1:.3f})',
    'The input line is not continuous (contains branches or discontinuities).':
        'La línea de entrada no es continua (tiene ramificaciones o discontinuidades).',
    'Warning: Could not sample DEM at point {0}':
        'Advertencia: no se pudo muestrear el DEM en el punto {0}',
    'Generates tree planting points with different patterns inside polygons. If DEM provided: automatic slope analysis and keyline pattern available. Output fields: id, pattern, x, y, slope (degrees).':
        'Genera puntos de plantación con diferentes patrones dentro de polígonos. Si se indica un DEM: análisis automático de pendiente y patrón keyline disponible. Campos de salida: id, pattern, x, y, slope (grados).',
    'TREE PLANTING COMPLETE REPORT':
        'REPORTE COMPLETO DE PLANTACIÓN',
    '📏 POLYGON INFORMATION:':
        '📏 INFORMACIÓN DEL POLÍGONO:',
    '   • Total area: {0:.4f} ha':
        '   • Área total: {0:.4f} ha',
    '📐 SPACING CONFIGURATION:':
        '📐 CONFIGURACIÓN DEL ESPACIAMIENTO:',
    '🧮 THEORETICAL CALCULATIONS (trees/ha):':
        '🧮 CÁLCULOS TEÓRICOS (árboles/ha):',
    '   • Square/Rectangular pattern: {0:.1f} trees/ha':
        '   • Patrón cuadrado/rectangular: {0:.1f} árboles/ha',
    '   • Triangular/Quincunx pattern: {0:.1f} trees/ha':
        '   • Patrón triangular/tresbolillo: {0:.1f} árboles/ha',
    '   • Five of Diamonds pattern: {0:.1f} trees/ha':
        '   • Patrón cinco de oros: {0:.1f} árboles/ha',
    '🌳 ACTUAL RESULTS (with border margin):':
        '🌳 RESULTADOS REALES (con margen de borde):',
    '📊 DETAILED COUNT BY PATTERN:':
        '📊 CONTEO DETALLADO POR PATRÓN:',
    '📈 EFFICIENCY ANALYSIS (BORDER MARGIN IMPACT):':
        '📈 ANÁLISIS DE EFICIENCIA (EFECTO DEL MARGEN DE BORDE):',
    '   • Square/Rectangular efficiency: {0:.1f}%':
        '   • Eficiencia cuadrado/rectangular: {0:.1f}%',
    '   • Triangular/Quincunx efficiency: {0:.1f}%':
        '   • Eficiencia triangular/tresbolillo: {0:.1f}%',
    '   • Five of Diamonds efficiency: {0:.1f}%':
        '   • Eficiencia cinco de oros: {0:.1f}%',
    '💡 RECOMMENDATIONS:':
        '💡 RECOMENDACIONES:',
    'Planting pattern':
        'Patrón de plantación',
    'Triangular/Quincunx (Tresbolillo)':
        'Triangular/Tresbolillo',
    'Rectangular/Square':
        'Rectangular/Cuadrado',
    'Five of Diamonds (Cinco de Oros)':
        'Cinco de oros',
    'Keyline (requires DEM)':
        'Keyline (requiere DEM)',
    'Random Distribution':
        'Distribución aleatoria',
    'Border margin (distance from boundary)':
        'Margen de borde (distancia desde el límite)',
    'Spacing (supports 3x3, 3*3, 3-3)':
        'Espaciamiento (admite 3x3, 3*3, 3-3)',
    'DEM Layer (optional - if provided: keyline pattern available + automatic slope analysis)':
        'Capa DEM (opcional; si se indica: patrón keyline disponible + análisis automático de pendiente)',
    'Output planting points':
        'Puntos de plantación de salida',
    'Triangular/Quincunx planting count':
        'Conteo de plantación triangular/tresbolillo',
    'Rectangular/Square planting count':
        'Conteo de plantación rectangular/cuadrado',
    'Five of Diamonds planting count':
        'Conteo de plantación cinco de oros',
    'Keyline planting count':
        'Conteo de plantación keyline',
    'Random distribution planting count':
        'Conteo de plantación aleatoria',
    'Could not load source layer':
        'No se pudo cargar la capa de origen',
    'Keyline pattern requires a DEM layer':
        'El patrón keyline requiere una capa DEM',
    'Could not create output layer':
        'No se pudo crear la capa de salida',
    'DEM detected - performing automatic slope analysis...':
        'DEM detectado: realizando el análisis automático de pendiente...',
    'No DEM provided - slope analysis disabled.':
        'No se indicó DEM: análisis de pendiente desactivado.',
    'CRS transform for slope sampling prepared ({0} -> {1}).':
        'Transformación de SRC para muestrear la pendiente preparada ({0} -> {1}).',
    'Preparing Keyline pattern (optimized for performance)...':
        'Preparando el patrón keyline (optimizado para rendimiento)...',
    'Creating union geometry of all polygons...':
        'Creando la geometría unida de todos los polígonos...',
    'Clipping DEM to polygon area...':
        'Recortando el DEM al área de los polígonos...',
    'Clipped DEM created. Extent: {0}':
        'DEM recortado creado. Extensión: {0}',
    'Calculating slope from clipped DEM...':
        'Calculando la pendiente a partir del DEM recortado...',
    'Statistics for generated slope raster: MIN={0:.2f}, MAX={1:.2f}':
        'Estadísticas del ráster de pendiente generado: MÍN={0:.2f}, MÁX={1:.2f}',
    'Slope map generated successfully.':
        'Mapa de pendientes generado correctamente.',
    'Generating stream network for keyline analysis...':
        'Generando la red de drenaje para el análisis keyline...',
    'Generated {0} stream segments':
        'Se generaron {0} tramos de cauce',
    'Generating keyline pattern...':
        'Generando el patrón keyline...',
    'Generated {0} continuous points around streams':
        'Se generaron {0} puntos continuos alrededor de los cauces',
    'Total keyline points after intelligent combination: {0}':
        'Total de puntos keyline tras la combinación inteligente: {0}',
    'Generated {0} random points (min: {1}m, max: {2}m)':
        'Se generaron {0} puntos aleatorios (mín: {1} m, máx: {2} m)',
    '   • Spacing: {0} x {1} m (Square)':
        '   • Espaciamiento: {0} x {1} m (cuadrado)',
    '   • Spacing: {0} x {1} m (Rectangular)':
        '   • Espaciamiento: {0} x {1} m (rectangular)',
    '🏔️ SLOPE ANALYSIS:':
        '🏔️ ANÁLISIS DE PENDIENTE:',
    '   • Slope analysis: AUTOMATIC (DEM detected)':
        '   • Análisis de pendiente: AUTOMÁTICO (DEM detectado)',
    "   • 'slope' field added with values in degrees":
        "   • Se agregó el campo 'slope' con valores en grados",
    '   • Slope values sampled directly from DEM':
        '   • Valores de pendiente muestreados directamente del DEM',
    '   • Slope analysis: DISABLED (no DEM)':
        '   • Análisis de pendiente: DESACTIVADO (sin DEM)',
    '   • Keyline pattern: {0:.1f} trees/ha':
        '   • Patrón keyline: {0:.1f} árboles/ha',
    '   • Random pattern: {0:.1f} trees/ha':
        '   • Patrón aleatorio: {0:.1f} árboles/ha',
    '   • Triangular/Quincunx: {0} trees [SELECTED]':
        '   • Triangular/tresbolillo: {0} árboles [SELECCIONADO]',
    '   • Triangular/Quincunx: {0} trees':
        '   • Triangular/tresbolillo: {0} árboles',
    '   • Five of Diamonds: {0} trees [SELECTED]':
        '   • Cinco de oros: {0} árboles [SELECCIONADO]',
    '   • Five of Diamonds: {0} trees':
        '   • Cinco de oros: {0} árboles',
    '   • Keyline: {0} trees [SELECTED]':
        '   • Keyline: {0} árboles [SELECCIONADO]',
    '   • Random: {0} trees [SELECTED]':
        '   • Aleatorio: {0} árboles [SELECCIONADO]',
    '   • Triangular/Quincunx pattern is more efficient for this spacing':
        '   • El patrón triangular/tresbolillo es más eficiente para este espaciamiento',
    '   • Keyline pattern optimized for water management and erosion control':
        '   • Patrón keyline optimizado para el manejo del agua y el control de la erosión',
    '   • Slope analysis completed successfully':
        '   • Análisis de pendiente completado correctamente',
    "   • Slope values available in 'slope' field (degrees)":
        "   • Valores de pendiente disponibles en el campo 'slope' (grados)",
    '   • Use these values to plan planting according to topography':
        '   • Use estos valores para planificar la plantación según la topografía',
    '   • For slope analysis, provide a DEM in next execution':
        '   • Para el análisis de pendiente, indique un DEM en la próxima ejecución',
    'Invalid spacing format: {0}':
        'Formato de espaciamiento no válido: {0}',
    'Could not generate slope map, continuing without slope analysis...':
        'No se pudo generar el mapa de pendientes; se continúa sin análisis de pendiente...',
    'Slope analysis will be included automatically.':
        'El análisis de pendiente se incluirá automáticamente.',
    'Reusing clipped DEM area for Keyline processing...':
        'Reutilizando el área del DEM recortado para el proceso keyline...',
    'Could not prepare DEM for Keyline, using simplified approach...':
        'No se pudo preparar el DEM para keyline; se usa un método simplificado...',
    'The union geometry of the polygons is empty.':
        'La geometría unida de los polígonos está vacía.',
    'DEM clipping failed. Cannot generate slope map.':
        'Falló el recorte del DEM. No se puede generar el mapa de pendientes.',
    'The resulting slope raster is not valid.':
        'El ráster de pendiente resultante no es válido.',
    'Error generating slope map: {0}':
        'Error al generar el mapa de pendientes: {0}',
    'DEM clipping resulted in an invalid layer.':
        'El recorte del DEM produjo una capa no válida.',
    'Error clipping DEM: {0}':
        'Error al recortar el DEM: {0}',
    'Generated {0} contour continuation points':
        'Se generaron {0} puntos de continuación de curvas',
    'Keyline generation error: {0}':
        'Error al generar keyline: {0}',
    'Random points generation error: {0}':
        'Error al generar puntos aleatorios: {0}',
    '   • Square: {0} trees [SELECTED]':
        '   • Cuadrado: {0} árboles [SELECCIONADO]',
    '   • Rectangular: {0} trees [SELECTED]':
        '   • Rectangular: {0} árboles [SELECCIONADO]',
    '   • Square: {0} trees':
        '   • Cuadrado: {0} árboles',
    '   • Rectangular: {0} trees':
        '   • Rectangular: {0} árboles',
    '   • Keyline: {0} trees':
        '   • Keyline: {0} árboles',
    '   • Random: {0} trees':
        '   • Aleatorio: {0} árboles',
    '   • Square/Rectangular pattern is more efficient for this spacing':
        '   • El patrón cuadrado/rectangular es más eficiente para este espaciamiento',
    '   • Triangular and Square patterns have similar efficiency':
        '   • Los patrones triangular y cuadrado tienen una eficiencia similar',
    '   • Random pattern with minimum distance {0}m and maximum {1}m':
        '   • Patrón aleatorio con distancia mínima de {0} m y máxima de {1} m',
    'Feature {0} became empty after applying border margin':
        'La entidad {0} quedó vacía tras aplicar el margen de borde',
    'Contour generation error: {0}':
        'Error al generar curvas: {0}',
    '\n        This algorithm delineates a watershed basin based on a Digital Elevation Model (DEM) and a pour point.\n        It uses GRASS GIS algorithms for hydrological analysis and watershed delineation.\n        \n        The stream network is required and will be clipped to the basin boundary with three Strahler stream order calculations:\n        - sthr_original: Standard Strahler order calculation\n        - sthr_extend: Extended main channel along longest paths\n        - sthr_final: Main channel extended to all headwaters\n        \n        Parameters:\n            Input DEM: A raster layer representing the terrain elevation\n            Pour Point: The outlet point of the watershed. Snapping to streams is enabled\n            Input Stream Network: A line vector layer representing the stream network (required)\n            Smoothing Iterations: Number of iterations for smoothing the basin boundary (0-10)\n            Smoothing Offset: Offset value for smoothing (0.0-0.5)\n            Calculate Longest Flow Path: If enabled, extends the main channel from its headwater to the basin boundary\n            \n        Outputs:\n            Output Basin: A polygon layer representing the delineated watershed basin\n            Output Basin Stream Network: A line layer with Strahler orders and longest flow path extension\n        ':
        '\n        Este algoritmo delimita una cuenca hidrográfica a partir de un modelo digital de elevación (DEM) y un punto de salida.\n        Usa algoritmos de GRASS GIS para el análisis hidrológico y la delimitación de la cuenca.\n\n        La red de drenaje es obligatoria y se recorta al límite de la cuenca con tres cálculos del orden de Strahler:\n        - sthr_original: cálculo estándar del orden de Strahler\n        - sthr_extend: cauce principal extendido por los recorridos más largos\n        - sthr_final: cauce principal extendido hasta todas las cabeceras\n\n        Parámetros:\n            DEM de entrada: ráster con la elevación del terreno\n            Punto de salida: punto de cierre de la cuenca. Se ajusta automáticamente a los cauces\n            Red de drenaje de entrada: capa de líneas con la red de drenaje (obligatoria)\n            Iteraciones de suavizado: número de iteraciones para suavizar el límite de la cuenca (0-10)\n            Desplazamiento de suavizado: valor de desplazamiento del suavizado (0.0-0.5)\n            Calcular la ruta de flujo más larga: si está activo, extiende el cauce principal desde su cabecera hasta el límite de la cuenca\n\n        Salidas:\n            Cuenca: capa de polígonos con la cuenca delimitada\n            Red de drenaje de la cuenca: capa de líneas con los órdenes de Strahler y la extensión de la ruta de flujo más larga\n        ',
    'Starting Strahler order calculation...':
        'Iniciando el cálculo del orden de Strahler...',
    'Starting main channel extension...':
        'Iniciando la extensión del cauce principal...',
    'Starting final channel extension to headwaters...':
        'Iniciando la extensión final del cauce hasta las cabeceras...',
    'Finding farthest point from outlet...':
        'Buscando el punto más lejano a la salida...',
    'Calculating Longest Flow Path...':
        'Calculando la ruta de flujo más larga...',
    'Finding first intersection between drain and stream network...':
        'Buscando la primera intersección entre la trayectoria de drenaje y la red de cauces...',
    'Splitting drain at first stream intersection...':
        'Dividiendo la trayectoria de drenaje en la primera intersección con un cauce...',
    'Topologically merging longest flow path with stream network...':
        'Uniendo topológicamente la ruta de flujo más larga con la red de drenaje...',
    'Could not start editing stream layer':
        'No se pudo iniciar la edición de la capa de drenaje',
    'Could not commit changes to stream layer':
        'No se pudieron guardar los cambios en la capa de drenaje',
    'Strahler order calculation completed':
        'Cálculo del orden de Strahler completado',
    'Could not start editing stream layer for extension':
        'No se pudo iniciar la edición de la capa de drenaje para la extensión',
    'Could not commit changes for extend':
        'No se pudieron guardar los cambios de la extensión',
    'Main channel extension completed':
        'Extensión del cauce principal completada',
    'Could not start editing stream layer for final extension':
        'No se pudo iniciar la edición de la capa de drenaje para la extensión final',
    'Found main channel path with {0} segments':
        'Se encontró el recorrido del cauce principal con {0} tramos',
    'Could not commit final changes':
        'No se pudieron guardar los cambios finales',
    'Final channel extension completed':
        'Extensión final del cauce completada',
    'Farthest point: {0:.2f}, {1:.2f}':
        'Punto más lejano: {0:.2f}, {1:.2f}',
    'Distance from outlet: {0:.2f} m':
        'Distancia desde la salida: {0:.2f} m',
    'Could not find outlet point':
        'No se encontró el punto de salida',
    'Could not find farthest point on basin boundary':
        'No se encontró el punto más lejano en el límite de la cuenca',
    'Running r.drain from: {0}':
        'Ejecutando r.drain desde: {0}',
    'r.drain vector output: {0}':
        'Salida vectorial de r.drain: {0}',
    'r.drain produced {0} feature(s)':
        'r.drain generó {0} entidad(es)',
    'Drain has {0} vertices':
        'La trayectoria de drenaje tiene {0} vértices',
    'Distance first point to boundary: {0:.2f} m':
        'Distancia del primer punto al límite: {0:.2f} m',
    'Distance last point to boundary: {0:.2f} m':
        'Distancia del último punto al límite: {0:.2f} m',
    'Splitting at vertex {0}':
        'Dividiendo en el vértice {0}',
    'Created drain segment with {0} vertices, length: {1:.2f} m':
        'Tramo de la trayectoria de drenaje creado con {0} vértices, longitud: {1:.2f} m',
    'No valid drain segment geometry':
        'No hay una geometría válida del tramo de la trayectoria de drenaje',
    'No intersecting stream feature':
        'No hay un cauce que se intersecte',
    'Splitting stream at index {0}':
        'Dividiendo el cauce en el índice {0}',
    'Created merged geometry: {0} vertices, length: {1:.2f} m':
        'Geometría unida creada: {0} vértices, longitud: {1:.2f} m',
    'Topological merge completed successfully':
        'Unión topológica completada correctamente',
    'Pour Point (click on the river)':
        'Punto de salida (haga clic sobre el río)',
    'Input Stream Network':
        'Red de drenaje de entrada',
    'Smoothing Iterations':
        'Iteraciones de suavizado',
    'Smoothing Offset':
        'Desplazamiento de suavizado',
    'Calculate Longest Flow Path':
        'Calcular la ruta de flujo más larga',
    'Watershed Basin':
        'Cuenca',
    'Basin Stream Network':
        'Red de drenaje de la cuenca',
    'Input Stream Network must be a line layer':
        'La red de drenaje de entrada debe ser una capa de líneas',
    'Could not add sthr_original field':
        'No se pudo agregar el campo sthr_original',
    'Could not update feature {0}':
        'No se pudo actualizar la entidad {0}',
    'Could not add sthr_extend field':
        'No se pudo agregar el campo sthr_extend',
    'Could not commit changes':
        'No se pudieron guardar los cambios',
    'Could not add sthr_final field':
        'No se pudo agregar el campo sthr_final',
    'Error tracing main channel: {0}':
        'Error al trazar el cauce principal: {0}',
    'r.drain output is not valid':
        'La salida de r.drain no es válida',
    'Error running r.drain: {0}':
        'Error al ejecutar r.drain: {0}',
    'No points in drain geometry':
        'No hay puntos en la geometría de la trayectoria de drenaje',
    'Drain flows from first to last point':
        'La trayectoria de drenaje va del primer al último punto',
    'Drain flows from last to first point (reversed)':
        'La trayectoria de drenaje va del último al primer punto (invertida)',
    'No valid stream geometries found':
        'No se encontraron geometrías de cauce válidas',
    'No intersection found within {0}m tolerance':
        'No se encontró intersección dentro de la tolerancia de {0} m',
    'Error finding intersection: {0}':
        'Error al buscar la intersección: {0}',
    'Segment has less than 2 points':
        'El tramo tiene menos de 2 puntos',
    'Error splitting drain: {0}':
        'Error al dividir la trayectoria de drenaje: {0}',
    'Segment 1 free end distance to outlet: {0:.2f} m':
        'Distancia del extremo libre del tramo 1 a la salida: {0:.2f} m',
    'Segment 2 free end distance to outlet: {0:.2f} m':
        'Distancia del extremo libre del tramo 2 a la salida: {0:.2f} m',
    'No valid stream segment to keep':
        'No hay un tramo de cauce válido para conservar',
    'Merged geometry has less than 2 points':
        'La geometría unida tiene menos de 2 puntos',
    'Could not delete stream feature {0}':
        'No se pudo eliminar el cauce {0}',
    'Could not add merged feature':
        'No se pudo agregar la entidad unida',
    'Error during merge: {0}':
        'Error durante la unión: {0}',
    'Error processing stream network: {0}':
        'Error al procesar la red de drenaje: {0}',
    'Outlet point: {0:.2f}, {1:.2f}':
        'Punto de salida: {0:.2f}, {1:.2f}',
    'Headwater point of main channel: {0:.2f}, {1:.2f}':
        'Cabecera del cauce principal: {0:.2f}, {1:.2f}',
    'Found first intersection at drain vertex {0}':
        'Primera intersección encontrada en el vértice {0} de la trayectoria de drenaje',
    'Intersection point: {0:.2f}, {1:.2f}':
        'Punto de intersección: {0:.2f}, {1:.2f}',
    'Keeping segment 1 (closer to outlet)':
        'Se conserva el tramo 1 (más cercano a la salida)',
    'Keeping segment 2 (closer to outlet)':
        'Se conserva el tramo 2 (más cercano a la salida)',
    'No outlet point found, skipping longest flow path':
        'No se encontró la salida; se omite la ruta de flujo más larga',
    'Using default merge order':
        'Usando el orden de unión predeterminado',
    'No intersection found, skipping longest flow path':
        'No se encontró intersección; se omite la ruta de flujo más larga',
    'Recalculating Strahler orders after merge...':
        'Recalculando los órdenes de Strahler tras la unión...',
    'Merge failed, continuing with original network':
        'Falló la unión; se continúa con la red original',
    '\n        This algorithm generates a stream network, applies smoothing, and calculates both Strahler and Shreve stream orders.\n        It uses GRASS GIS algorithms for stream extraction, QGIS native smooth algorithm, and implements custom Strahler and Shreve order calculations.\n        \n        For large, high-resolution DEMs, the algorithm may automatically resample the input to a lower resolution to enable processing.\n        \n        Parameters:\n            Input DEM: A UTM raster layer representing the terrain elevation\n            Flow Accumulation Threshold: Minimum number of cells to form a stream (controls stream network density)\n            Smooth Iterations: Number of smoothing iterations to apply\n            Smooth Offset: Controls the smoothness of the output\n        Outputs:\n            Output Stream Network: A line layer representing the smoothed stream network with Strahler and Shreve orders\n        ':
        '\n        Este algoritmo genera una red de drenaje, la suaviza y calcula los órdenes de cauce de Strahler y Shreve.\n        Usa algoritmos de GRASS GIS para extraer los cauces, el suavizado nativo de QGIS y cálculos propios de los órdenes de Strahler y Shreve.\n\n        Con DEM grandes y de alta resolución, el algoritmo puede remuestrear automáticamente la entrada a una resolución menor para poder procesarla.\n\n        Parámetros:\n            DEM de entrada: ráster en UTM con la elevación del terreno\n            Umbral de acumulación de flujo: número mínimo de celdas para formar un cauce (controla la densidad de la red)\n            Iteraciones de suavizado: número de iteraciones de suavizado\n            Desplazamiento de suavizado: controla la suavidad de la salida\n        Salidas:\n            Red de drenaje: capa de líneas con la red suavizada y los órdenes de Strahler y Shreve\n        ',
    'Stream Network':
        'Red de drenaje',
    'Input DEM is too large to process efficiently even after resampling. Please use a smaller area or lower resolution DEM.':
        'El DEM de entrada es demasiado grande para procesarse eficientemente, incluso tras remuestrearlo. Use un área más pequeña o un DEM de menor resolución.',
    "This algorithm performs a weighted sum of multiple raster layers.\n\n1. Select multiple input raster layers.\n2. Enter the weights as comma-separated values (e.g., 50,50 for two layers).\n3. The weights must sum to 100%.\n4. IMPORTANT: The order of the weights must match the order of the selected rasters.\n   The first weight corresponds to the first raster, the second weight to the second raster, and so on.\n\nFor example, if you select rasters A, B, and C in that order, and enter weights '30,30,40',\nthe algorithm will apply 30% to raster A, 30% to raster B, and 40% to raster C.\n\nThe tool will automatically calculate the weighted sum based on your inputs.":
        "Este algoritmo realiza una suma ponderada de varias capas ráster.\n\n1. Seleccione varias capas ráster de entrada.\n2. Escriba los pesos como valores separados por comas (p. ej. 50,50 para dos capas).\n3. Los pesos deben sumar 100%.\n4. IMPORTANTE: el orden de los pesos debe coincidir con el orden de los rásteres seleccionados.\n   El primer peso corresponde al primer ráster, el segundo peso al segundo ráster, y así sucesivamente.\n\nPor ejemplo, si selecciona los rásteres A, B y C en ese orden y escribe los pesos '30,30,40',\nel algoritmo aplicará 30% al ráster A, 30% al ráster B y 40% al ráster C.\n\nLa herramienta calculará automáticamente la suma ponderada a partir de sus entradas.",
    'Input rasters':
        'Rásteres de entrada',
    'Weights (comma-separated)':
        'Pesos (separados por comas)',
    'Output Weighted Sum Raster':
        'Ráster de suma ponderada de salida',
    'Error calculating output raster: {0}':
        'Error al calcular el ráster de salida: {0}',
    'Compensated Channel Slope (Linear Regression)':
        'Pendiente compensada del cauce (regresión lineal)',
    'Compensated Channel Slope - LR (%)':
        'Pendiente compensada del cauce - RL (%)',
}
