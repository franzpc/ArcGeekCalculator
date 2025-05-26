# ArcGeek Calculator Plugin
Version 3.0beta

ArcGeek Calculator is a comprehensive QGIS plugin that provides various hydrological, geomorphological, spatial analysis, machine learning, and optimization tools. This version introduces new advanced algorithms including Enhanced Image Classification, Least Cost Path Finder, and Tree Planting Pattern Generator, along with existing tools for Global Curve Number calculation, 3D CAD exports, kriging analysis, satellite index calculation, basemap management, and more.

## Description
ArcGeek Calculator is a QGIS plugin that provides a comprehensive set of tools for coordinate calculations, conversions, spatial operations, watershed analysis, land use analysis, flood simulation, 3D CAD integration, topographic profiling, machine learning classification, path optimization, and forestry planning in QGIS. It's designed for GIS analysts, cartographers, surveyors, hydrologists, urban planners, foresters, and anyone working with spatial data.

## Key Features

### Geometric Tools
1. **Calculate Coordinates**: Add XY coordinates to point layers, convert to Decimal Degrees, and provide two formats of Degrees Minutes Seconds.
2. **Calculate Line Geometry**: Calculate length and azimuth for line features.
3. **Calculate Polygon Geometry**: Calculate area and perimeter for polygon features.
4. **Calculate Angles**: Calculate internal and external angles at vertices of lines or polygons.
5. **Go to XY**: Quickly navigate to specific coordinates on the map and optionally create point markers.

### Point and Line Processing
6. **Extract Ordered Points from Polygons**: Extract and order points from the vertices of input polygons with bi-directional numbering.
7. **Lines to Ordered Points**: Convert line features to ordered point features.
8. **Calculate Line from Coordinates and Table**: Generate a line and points from starting coordinates and a table of distances and angles.

### Hydrological Analysis
9. **Stream Network with Order**: Generate a stream network with Strahler order.
10. **Watershed Basin Delineation**: Delineate watershed basins from a DEM and pour points with enhanced main channel extension capabilities.
11. **Watershed Morphometric Analysis**: Perform a comprehensive morphometric analysis of a watershed, calculating various parameters and providing their interpretations.
12. **Calculate Global Curve Number**: Calculate CN values using global datasets for hydrological analysis.

### Terrain Analysis
13. **Topographic Profile**: Generate interactive topographic profiles from lines and DEMs, with features including:
    - Optional point labeling with customizable distance threshold
    - Automatic profile smoothing based on total distance
    - Interactive HTML output with downloadable point data
    - Profile area visualization with modern color scheme
    - Detailed elevation statistics and distance measurements

### Path Optimization and Analysis
14. **Least Cost Path Finder**: Advanced path optimization algorithm with multiple optimization levels and flat terrain sensitivity for drainage channels, hiking trails, and utility corridors.

### Land Analysis and Raster Processing
15. **Land Use Change Detection**: Analyze changes in land use between two time periods.
16. **Weighted Sum Analysis**: Perform weighted sum analysis on multiple raster layers.
17. **Optimized Parcel Division**: Divide rectangular parcels into lots of specified width.
18. **Dam Flood Simulation**: Simulate flooding based on a DEM and specified water level.

### Machine Learning and Classification
19. **Enhanced Image Classification**: Advanced machine learning classification with multiple algorithms (Random Forest, SVM, GMM, K-NN) and scientific accuracy assessment following remote sensing standards.

### Forestry and Agricultural Planning
20. **Tree Planting Pattern Generator**: Generate optimal tree planting patterns including Triangular/Quincunx, Rectangular/Square, and Five of Diamonds patterns with configurable spacing and comparative density analysis.

### Advanced Analysis and Remote Sensing
21. **Kriging Analysis**: Perform spatial interpolation using Kriging (requires external libraries).
22. **Satellite Index Calculator**: Calculate various satellite indices (NDVI, NDWI, etc.) for different satellites.

### Data Management and Export
23. **Basemap Manager**: Add and manage basemaps from Google Maps, Esri, Bing, and others.
24. **Screen Capture**: Capture and georeference the current map view.
25. **Export to CSV**: Export vector layer attributes to CSV format compatible with Excel.
26. **Export Contours to 3D CAD**: Export contour lines to DXF format preserving elevation values, making them compatible with AutoCAD, Civil 3D, BricsCAD, and other CAD software.

## External Libraries
Some tools in this plugin require external libraries:
- **Kriging Analysis**: Requires pykrige and scipy
- **Hypsometric Curve** (part of Watershed Morphometric Analysis): Interactive version requires plotly and numpy
- **Topographic Profile**: Requires plotly
- **Enhanced Image Classification**: Requires scikit-learn, numpy; matplotlib and seaborn for full report functionality

## Installation Instructions for External Libraries
For algorithms requiring external libraries, install them using pip:

```bash
# For Kriging Analysis
pip install pykrige scipy

# For Enhanced Image Classification (basic functionality)
pip install scikit-learn numpy

# For Enhanced Image Classification (full reports with charts)
pip install scikit-learn numpy matplotlib seaborn

# For Topographic Profile and Hypsometric Curves
pip install plotly numpy
```

## Support
If you encounter any issues or have any suggestions, please open an issue on our [issue tracker](https://github.com/franzpc/ArcGeekCalculator/issues).

## Support the Project
If you find ArcGeek Calculator useful, please consider supporting its development. Your contributions help maintain and improve the plugin.

You can make a donation via PayPal: [https://paypal.me/ArcGeek](https://paypal.me/ArcGeek)

Every contribution, no matter how small, is greatly appreciated and helps ensure the continued development of this tool.

## License
This project is licensed under the GNU General Public License v2.0 or later. See the [LICENSE](LICENSE) file for details.

## Author
ArcGeek - Franz Pucha-Cofrep

## Version History

**3.0**: Major update with new advanced algorithms:
- Added Enhanced Image Classification with machine learning algorithms and scientific reporting
- Added Least Cost Path Finder with advanced optimization options and flat terrain sensitivity
- Added Tree Planting Pattern Generator for forestry and agricultural planning
- Enhanced Watershed Basin Delineation with main channel extension capabilities
- Improved overall performance and user experience

**2.8beta**: Added Topographic Profile tool with interactive visualization and point labeling capabilities. Delineated watershed basin include Extend Main Channel.

**2.7beta**: Added snapping and bug fixes for watershed delimitation.

**2.6beta**: Added Global Curve Number calculation and Export Contours to 3D CAD tools.

**2.5beta**: Fixed the bug that did not calculate the coordinates for the last row (Thanks to @russ-go), compensated slope suggested by Fernando Oñate.

**2.4beta**: Added Hypsometric Integral (HI), and Calculate Angle tools.

**2.3beta**: Added Kriging Analysis, Satellite Index Calculator, Basemap Manager, Screen Capture, and Export to CSV tools.

**2.1beta**: Fix errors and improve performance.

**2.0beta**: Added Dam Flood Simulation tool, correction of general errors.

**1.9beta**: Improved Optimized Parcel Division tool with two-pass small polygon merging

**1.8beta**: Added Land Use Change Detection, Weighted Sum Analysis, and Optimized Parcel Division tools

**1.7beta**: Added new tools for watershed analysis and geometric calculations

**1.6beta**: Enhanced "Extract Ordered Points from Polygons" with bi-directional numbering

**1.5beta**: Added "Watershed Morphometric Analysis" tool

**1.4beta**: Added "Extract Ordered Points from Polygons" functionalities

**1.3beta**: Added "Calculate Line from Coordinates and Table", and "Go to XY" functionalities

**1.2beta**: Added "Lines to Ordered Points" functionality

**1.1beta**: Enhanced functionality and bug fixes

**1.0**: Initial release with "Calculate Coordinates"

## How to cite?

If you use ArcGeek Calculator in your research or projects, please cite it as follows:

Pucha-Cofrep, Franz. (2024). ArcGeek Calculator (Version 3.x) [QGIS Plugin]. GitHub. https://github.com/franzpc/ArcGeekCalculator

For in-text citations, you can use: (Pucha-Cofrep, 2024) for parenthetical citations or Pucha-Cofrep (2024) for narrative citations. The citation format follows the American Psychological Association Style 7th Edition (APA 7).