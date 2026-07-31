"""QGIS 3.x / 4.x enum compatibility shim.

QGIS 4.x (Qt6/PyQt6) moves most enums into the ``Qgis`` namespace and
fully-qualified enum forms.  This module resolves the correct value at
import time so the rest of the plugin stays clean.

Strategy: try the **new** form first, fall back to the old one.
"""

from qgis.core import (
    Qgis,
    QgsAggregateCalculator,
    QgsLayoutExporter,
    QgsMapLayer,
    QgsProcessingParameterDefinition,
    QgsRasterBandStats,
    QgsVectorSimplifyMethod,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QIODevice, Qt
from qgis.PyQt.QtWidgets import QToolButton

# ── Layer types ──────────────────────────────────────────────────────
try:
    LAYER_VECTOR = Qgis.LayerType.Vector
except AttributeError:
    LAYER_VECTOR = QgsMapLayer.LayerType.VectorLayer

try:
    LAYER_RASTER = Qgis.LayerType.Raster
except AttributeError:
    LAYER_RASTER = QgsMapLayer.LayerType.RasterLayer

# ── Message levels ───────────────────────────────────────────────────
try:
    MSG_INFO = Qgis.MessageLevel.Info
except AttributeError:
    MSG_INFO = Qgis.MessageLevel.Info

try:
    MSG_WARNING = Qgis.MessageLevel.Warning
except AttributeError:
    MSG_WARNING = Qgis.MessageLevel.Warning

try:
    MSG_CRITICAL = Qgis.MessageLevel.Critical
except AttributeError:
    MSG_CRITICAL = Qgis.MessageLevel.Critical

# ── Geometry types ───────────────────────────────────────────────────
try:
    GEOM_POLYGON = Qgis.GeometryType.Polygon
except AttributeError:
    GEOM_POLYGON = QgsWkbTypes.GeometryType.PolygonGeometry

try:
    GEOM_LINE = Qgis.GeometryType.Line
except AttributeError:
    GEOM_LINE = QgsWkbTypes.GeometryType.LineGeometry

# ── Raster stats ─────────────────────────────────────────────────────
try:
    RASTER_STATS_ALL = Qgis.RasterBandStatistic.All
except AttributeError:
    RASTER_STATS_ALL = QgsRasterBandStats.Stats.All

# ── Layout export result ─────────────────────────────────────────────
try:
    LAYOUT_SUCCESS = Qgis.LayoutResult.Success
except AttributeError:
    LAYOUT_SUCCESS = QgsLayoutExporter.ExportResult.Success

# ── Processing parameter flags ───────────────────────────────────────
try:
    PROCESSING_OPTIONAL = Qgis.ProcessingParameterFlag.Optional
except AttributeError:
    PROCESSING_OPTIONAL = QgsProcessingParameterDefinition.Flag.FlagOptional

# ── Aggregate functions ──────────────────────────────────────────────
try:
    AGG_COUNT = Qgis.Aggregate.Count
    AGG_SUM = Qgis.Aggregate.Sum
    AGG_MEAN = Qgis.Aggregate.Mean
    AGG_MIN = Qgis.Aggregate.Min
    AGG_MAX = Qgis.Aggregate.Max
    AGG_STDEV = Qgis.Aggregate.StDev
    AGG_ARRAY = Qgis.Aggregate.ArrayAggregate
except AttributeError:
    AGG_COUNT = QgsAggregateCalculator.Aggregate.Count
    AGG_SUM = QgsAggregateCalculator.Aggregate.Sum
    AGG_MEAN = QgsAggregateCalculator.Aggregate.Mean
    AGG_MIN = QgsAggregateCalculator.Aggregate.Min
    AGG_MAX = QgsAggregateCalculator.Aggregate.Max
    AGG_STDEV = QgsAggregateCalculator.Aggregate.StDev
    AGG_ARRAY = QgsAggregateCalculator.Aggregate.ArrayAggregate

# ── Qt IO / widget enums ─────────────────────────────────────────────
try:
    IODEVICE_WRITEONLY = QIODevice.OpenModeFlag.WriteOnly
except AttributeError:
    IODEVICE_WRITEONLY = QIODevice.OpenModeFlag.WriteOnly

try:
    TOOLBUTTON_MENU_POPUP = QToolButton.ToolButtonPopupMode.MenuButtonPopup
except AttributeError:
    TOOLBUTTON_MENU_POPUP = QToolButton.ToolButtonPopupMode.MenuButtonPopup

try:
    TOOLBUTTON_ICON_ONLY = Qt.ToolButtonStyle.ToolButtonIconOnly
except AttributeError:
    TOOLBUTTON_ICON_ONLY = Qt.ToolButtonStyle.ToolButtonIconOnly

# ── Vector simplification hints ─────────────────────────────────────
try:
    SIMPLIFY_GEOMETRY = QgsVectorSimplifyMethod.SimplifyHint.GeometrySimplification
    SIMPLIFY_ANTIALIAS = QgsVectorSimplifyMethod.SimplifyHint.AntialiasingSimplification
except AttributeError:
    SIMPLIFY_GEOMETRY = QgsVectorSimplifyMethod.SimplifyHint.GeometrySimplification
    SIMPLIFY_ANTIALIAS = QgsVectorSimplifyMethod.SimplifyHint.AntialiasingSimplification
