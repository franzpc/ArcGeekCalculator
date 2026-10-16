"""Plugin side of qgis-mcp, vendored from https://github.com/nkarasiak/qgis-mcp.

Copied unchanged from qgis_mcp_plugin/ at tag v0.15.1, except
constants.plugin_version(), which returns the upstream version instead of
reading a metadata.txt. Upstream's plugin.py, configurator.py and icons are
not included: ArcGeek Calculator provides its own dialog (scripts/qgis_mcp.py).

To update: copy the same files from a newer tag, re-apply that one patch and
bump UPSTREAM_VERSION in constants.py.

Copyright (c) 2025 Nicolas Karasiak. GPL-2.0-or-later, see LICENSE.
"""
