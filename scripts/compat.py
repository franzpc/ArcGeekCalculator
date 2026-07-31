"""QGIS 3.x / 4.x enum compatibility shim.

QGIS 4.x (Qt6/PyQt6) moves most enums into the ``Qgis`` namespace and
fully-qualified enum forms.  This module resolves the correct value at
import time so the rest of the plugin stays clean.

Strategy: try the **new** form first, fall back to the old one.
"""

from qgis.core import (
    Qgis,
    QgsAbstractDatabaseProviderConnection,
    QgsAggregateCalculator,
    QgsColorRampShader,
    QgsContrastEnhancement,
    QgsLayoutExporter,
    QgsMapLayer,
    QgsProcessingParameterDefinition,
    QgsRasterBandStats,
    QgsSingleBandGrayRenderer,
    QgsVectorLayerExporter,
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
SIMPLIFY_GEOMETRY = QgsVectorSimplifyMethod.SimplifyHint.GeometrySimplification
SIMPLIFY_ANTIALIAS = QgsVectorSimplifyMethod.SimplifyHint.AntialiasingSimplification

# ── Raster shader interpolation / classification ─────────────────────
# Qgis.ShaderInterpolationMethod / ShaderClassificationMethod arrived in
# 3.38; the QgsColorRampShader nested forms cover earlier releases.
try:
    SHADER_INTERPOLATED = Qgis.ShaderInterpolationMethod.Linear
    SHADER_DISCRETE = Qgis.ShaderInterpolationMethod.Discrete
    SHADER_EXACT = Qgis.ShaderInterpolationMethod.Exact
except AttributeError:
    SHADER_INTERPOLATED = QgsColorRampShader.Type.Interpolated
    SHADER_DISCRETE = QgsColorRampShader.Type.Discrete
    SHADER_EXACT = QgsColorRampShader.Type.Exact

try:
    SHADER_CLASS_CONTINUOUS = Qgis.ShaderClassificationMethod.Continuous
    SHADER_CLASS_EQUAL_INTERVAL = Qgis.ShaderClassificationMethod.EqualInterval
    SHADER_CLASS_QUANTILE = Qgis.ShaderClassificationMethod.Quantile
except AttributeError:
    SHADER_CLASS_CONTINUOUS = QgsColorRampShader.ClassificationMode.Continuous
    SHADER_CLASS_EQUAL_INTERVAL = QgsColorRampShader.ClassificationMode.EqualInterval
    SHADER_CLASS_QUANTILE = QgsColorRampShader.ClassificationMode.Quantile

# ── Raster contrast enhancement / gray gradient ──────────────────────
try:
    _CEA = QgsContrastEnhancement.ContrastEnhancementAlgorithm
    CONTRAST_NONE = _CEA.NoEnhancement
    CONTRAST_STRETCH_MINMAX = _CEA.StretchToMinimumMaximum
    CONTRAST_CLIP_MINMAX = _CEA.ClipToMinimumMaximum
    CONTRAST_STRETCH_CLIP_MINMAX = _CEA.StretchAndClipToMinimumMaximum
except AttributeError:
    CONTRAST_NONE = QgsContrastEnhancement.NoEnhancement
    CONTRAST_STRETCH_MINMAX = QgsContrastEnhancement.StretchToMinimumMaximum
    CONTRAST_CLIP_MINMAX = QgsContrastEnhancement.ClipToMinimumMaximum
    CONTRAST_STRETCH_CLIP_MINMAX = QgsContrastEnhancement.StretchAndClipToMinimumMaximum

try:
    GRAY_BLACK_TO_WHITE = QgsSingleBandGrayRenderer.Gradient.BlackToWhite
    GRAY_WHITE_TO_BLACK = QgsSingleBandGrayRenderer.Gradient.WhiteToBlack
except AttributeError:
    GRAY_BLACK_TO_WHITE = QgsSingleBandGrayRenderer.BlackToWhite
    GRAY_WHITE_TO_BLACK = QgsSingleBandGrayRenderer.WhiteToBlack

# ── Database provider connections ────────────────────────────────────
try:
    _TF = QgsAbstractDatabaseProviderConnection.TableFlag
    CONN_TABLE_VECTOR = _TF.Vector
    CONN_TABLE_RASTER = _TF.Raster
    CONN_TABLE_VIEW = _TF.View
    CONN_TABLE_ASPATIAL = _TF.Aspatial
except AttributeError:
    CONN_TABLE_VECTOR = QgsAbstractDatabaseProviderConnection.Vector
    CONN_TABLE_RASTER = QgsAbstractDatabaseProviderConnection.Raster
    CONN_TABLE_VIEW = QgsAbstractDatabaseProviderConnection.View
    CONN_TABLE_ASPATIAL = QgsAbstractDatabaseProviderConnection.Aspatial

try:
    _CAP = QgsAbstractDatabaseProviderConnection.Capability
    CONN_CAP_SCHEMAS = _CAP.Schemas
    CONN_CAP_SQL_LAYERS = _CAP.SqlLayers
    CONN_CAP_EXECUTE_SQL = _CAP.ExecuteSql
except AttributeError:
    CONN_CAP_SCHEMAS = QgsAbstractDatabaseProviderConnection.Schemas
    CONN_CAP_SQL_LAYERS = QgsAbstractDatabaseProviderConnection.SqlLayers
    CONN_CAP_EXECUTE_SQL = QgsAbstractDatabaseProviderConnection.ExecuteSql

# ── Vector layer export result ───────────────────────────────────────
try:
    EXPORT_SUCCESS = Qgis.VectorExportResult.Success
except AttributeError:
    EXPORT_SUCCESS = QgsVectorLayerExporter.NoError
