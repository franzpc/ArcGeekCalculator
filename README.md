# ArcGeek Calculator Plugin
Version 2.8beta

ArcGeek Calculator is a comprehensive QGIS plugin that provides various hydrological, geomorphological, and spatial analysis tools. This version introduces a new Topographic Profile tool, along with existing tools for Global Curve Number calculation, 3D CAD exports, kriging analysis, satellite index calculation, basemap management, and more.

## Description
ArcGeek Calculator is a QGIS plugin that provides a comprehensive set of tools for coordinate calculations, conversions, spatial operations, watershed analysis, land use analysis, flood simulation, 3D CAD integration, and topographic profiling in QGIS. It's designed for GIS analysts, cartographers, surveyors, hydrologists, urban planners, and anyone working with spatial data.

## Key Features

### Geometric Tools
1. Calculate Coordinates: Add XY coordinates to point layers, convert to Decimal Degrees, and provide two formats of Degrees Minutes Seconds.
2. Calculate Line Geometry: Calculate length and azimuth for line features.
3. Calculate Polygon Geometry: Calculate area and perimeter for polygon features.
4. Go to XY: Quickly navigate to specific coordinates on the map and optionally create point markers.

### Point and Line Processing
5. Extract Ordered Points from Polygons: Extract and order points from the vertices of input polygons with bi-directional numbering.
6. Lines to Ordered Points: Convert line features to ordered point features.
7. Calculate Line from Coordinates and Table: Generate a line and points from starting coordinates and a table of distances and angles.

### Hydrological Analysis
8. Stream Network with Order: Generate a stream network with Strahler order.
9. Watershed Basin Delineation: Delineate watershed basins from a DEM and pour points.
10. Watershed Morphometric Analysis: Perform a comprehensive morphometric analysis of a watershed, calculating various parameters and providing their interpretations.
11. Calculate Global Curve Number: Calculate CN values using global datasets for hydrological analysis.

### Terrain Analysis
12. Topographic Profile: Generate interactive topographic profiles from lines and DEMs, with features including:
    - Optional point labeling with customizable distance threshold
    - Automatic profile smoothing based on total distance
    - Interactive HTML output with downloadable point data
    - Profile area visualization with modern color scheme
    - Detailed elevation statistics and distance measurements

### Land Analysis and Raster Processing
13. Land Use Change Detection: Analyze changes in land use between two time periods.
14. Weighted Sum Analysis: Perform weighted sum analysis on multiple raster layers.
15. Optimized Parcel Division: Divide rectangular parcels into lots of specified width.
16. Dam Flood Simulation: Simulate flooding based on a DEM and specified water level.

### Advanced Analysis and Remote Sensing
17. Kriging Analysis: Perform spatial interpolation using Kriging (requires external libraries).
18. Satellite Index Calculator: Calculate various satellite indices (NDVI, NDWI, etc.) for different satellites.

### Data Management and Export
19. Basemap Manager: Add and manage basemaps from Google Maps, Esri, Bing, and others.
20. Screen Capture: Capture and georeference the current map view.
21. Export to CSV: Export vector layer attributes to CSV format compatible with Excel.
22. Export Contours to 3D CAD: Export contour lines to DXF format preserving elevation values, making them compatible with AutoCAD, Civil 3D, BricsCAD, and other CAD software.

## External Libraries
Some tools in this plugin require external libraries:
- Kriging Analysis: Requires pykrige and scipy
- Hypsometric Curve (part of Watershed Morphometric Analysis): Interactive version requires plotly and numpy
- Topographic Profile: Requires plotly

## Support
If you encounter any issues or have any suggestions, please open an issue on our [issue tracker](https://github.com/franzpc/ArcGeekCalculator/issues).

## Support the Project
If you find ArcGeek Calculator useful, please consider supporting its development. Your contributions help maintain and improve the plugin.

You can make a donation via PayPal: [https://paypal.me/ArcGeek](https://paypal.me/ArcGeek)

Every contribution, no matter how small, is greatly appreciated and helps ensure the continued development of this tool.

## License
This project is licensed under the GNU General Public License v2.0 or later. See the [LICENSE](LICENSE) file for details.

## Author
ArcGeek

## Version History

2.8beta: Added Topographic Profile tool with interactive visualization and point labeling capabilities.

2.7beta: Added snapping and bug fixes for watershed delimitation.

2.6beta: Added Global Curve Number calculation and Export Contours to 3D CAD tools.

2.5beta: Fixed the bug that did not calculate the coordinates for the last row (Thanks to @russ-go), compensated slope suggested by Fernando Oñate.

2.4beta: Added Hypsometric Integral (HI), and Calculate Angle tools.

2.3beta: Added Kriging Analysis, Satellite Index Calculator, Basemap Manager, Screen Capture, and Export to CSV tools.

2.1beta: Fix errors and improve performance.

2.0beta: Added Dam Flood Simulation tool, correction of general errors.

1.9beta: Improved Optimized Parcel Division tool with two-pass small polygon merging

1.8beta: Added Land Use Change Detection, Weighted Sum Analysis, and Optimized Parcel Division tools

1.7beta: Added new tools for watershed analysis and geometric calculations

1.6beta: Enhanced "Extract Ordered Points from Polygons" with bi-directional numbering

1.5beta: Added "Watershed Morphometric Analysis" tool

1.4beta: Added "Extract Ordered Points from Polygons" functionalities

1.3beta: Added "Calculate Line from Coordinates and Table", and "Go to XY" functionalities

1.2beta: Added "Lines to Ordered Points" functionality

1.1beta: Enhanced functionality and bug fixes

1.0: Initial release with "Calculate Coordinates"

## How to cite?

If you use ArcGeek Calculator in your research or projects, please cite it as follows:

Pucha-Cofrep, Franz. (2024). ArcGeek Calculator (Version 2.x) [QGIS Plugin]. GitHub. https://github.com/franzpc/ArcGeekCalculator

For in-text citations, you can use: (Pucha-Cofrep, 2024) for parenthetical citations or Pucha-Cofrep (2024) for narrative citations. The citation format follows the American Psychological Association Style 7th Edition (APA 7).