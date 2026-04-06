# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ArcGeek Calculator is a QGIS plugin (Python). It runs inside QGIS — there is no standalone build, test runner, or CLI. To test changes you must reload the plugin inside QGIS (use the Plugin Reloader plugin) or restart QGIS.

## Repository structure

```
plugin.py          # Main plugin class (ArcGeekCalculator). Registers all menu actions and wires tools together.
__init__.py        # QGIS entry point — only calls classFactory() from plugin.py
scripts/           # One file per tool
scripts/qgis_mcp/  # MCP Server tool (AI integration via Claude Desktop)
icons/             # PNG icons referenced by plugin.py
metadata.txt       # QGIS plugin metadata (version, tags, description)
```

## Two tool patterns

Every tool follows one of two patterns:

**1. Processing Algorithm** (`QgsProcessingAlgorithm` subclass)
Used for geoprocessing tools that appear in the Processing Toolbox. Examples: `basin_analysis_algorithm.py`, `watershed_basin.py`, `land_use_change_algorithm.py`.
- Registered via `processing.execAlgorithmDialog(self.algorithms[name])` in `plugin.py`
- Must implement `initAlgorithm()`, `processAlgorithm()`, `name()`, `displayName()`, `group()`, `groupId()`, `createInstance()`

**2. Dialog tool** (`QDialog` subclass)
Used for interactive tools without a processing pipeline. Examples: `go_to_xy.py`, `screen_capture.py`, `scripts/qgis_mcp/qgis_mcp_plugin.py`.
- Opened directly via a method in `plugin.py` (e.g. `toggle_mcp_dialog`, `run_go_to_xy`)
- Dialog exec compatibility pattern required (see below)

## Qt5 / Qt6 — QGIS 3 / QGIS 4 compatibility

QGIS 3 uses PyQt5, QGIS 4 uses PyQt6. Enum access changed completely in PyQt6. Always use this pattern for Qt enums:

```python
# Flags / WindowType / DockWidgetArea / AlignmentFlag
try:
    align_center = Qt.AlignmentFlag.AlignCenter
except AttributeError:
    align_center = Qt.AlignCenter

# Layer types
try:
    VECTOR_LAYER_TYPE = Qgis.LayerType.Vector
    RASTER_LAYER_TYPE = Qgis.LayerType.Raster
except AttributeError:
    VECTOR_LAYER_TYPE = QgsMapLayer.VectorLayer
    RASTER_LAYER_TYPE = QgsMapLayer.RasterLayer
```

Dialog exec compatibility:
```python
try:
    dialog.exec()
except AttributeError:
    dialog.exec_()
```

`layer.geometryType()` and `feature.geometry().type()` return enums in QGIS 4 — wrap with `int()` before JSON serialization or string formatting.

## MCP Server architecture

`scripts/qgis_mcp/` has two separate processes that communicate via TCP socket on port 9876:

- **`qgis_mcp_plugin.py`** — runs inside QGIS. `QgisMCPServer` listens on a socket using a `QTimer` (non-blocking, 100ms polling) to stay on the Qt main thread. Executes QGIS commands received as JSON.
- **`server_mcp/qgis_mcp_server.py`** — runs outside QGIS (via `uv`). Acts as the MCP server that Claude Desktop connects to. Translates MCP tool calls into JSON commands sent to the QGIS socket.

Commands are plain JSON: `{"type": "command_name", "params": {...}}`.

## Adding a new tool

1. Create `scripts/my_tool.py` with the appropriate class (Algorithm or Dialog).
2. Import it in `plugin.py`.
3. For algorithms: add to the `self.algorithms` dict and call `self.add_action(...)` using `self.run_algorithm('key')`.
4. For dialogs: add a method and wire it with `self.add_action(...)`.
5. Add an icon to `icons/` (PNG).
6. Update `metadata.txt`: bump `version`, update changelog section and tags.
7. Update `README.md`: version number, tool list, and citation line.

## Versioning

Version is maintained in three places — keep them in sync:
- `metadata.txt` → `version=`
- `README.md` → title line and citation
- `README.md` → Version History section
