"""Values shared by the server, the plugin entry point and the configurator.

Stdlib-only and free of ``qgis`` imports, like :mod:`wire`, :mod:`errors` and
:mod:`registry`. Keeping them here is what lets ``configurator`` read the
settings prefix without importing ``plugin``, which imports it back.
"""

import errno
import os

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9876

# The plugin package directory - where metadata.txt and icons/ live. Anchored
# here rather than on each module's own ``__file__`` so a module inside a
# subpackage (``handlers/``) still resolves them correctly.
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))

# QgsSettings key prefix for every setting the plugin persists.
SETTINGS_PREFIX = "qgis_mcp"

# Cap what is accepted as a peer's version string: it arrives over the socket
# and ends up in front of the user. Mirrors MAX_VERSION_LENGTH in
# src/qgis_mcp/protocol.py, which the client applies before sending.
MAX_VERSION_LENGTH = 32

# Same treatment for the checkout path a peer announces. Mirrors
# MAX_PATH_LENGTH in src/qgis_mcp/protocol.py.
MAX_PATH_LENGTH = 160

# ArcGeek Calculator: this package is vendored without upstream's metadata.txt,
# so report the upstream release it was copied from. diagnose compares this
# against the MCP server's version, and the two must match for a clean result.
UPSTREAM_VERSION = "0.15.1"


def plugin_version():
    """The qgis-mcp release this vendored copy corresponds to."""
    return UPSTREAM_VERSION


# "someone else already holds this port". EADDRINUSE is the usual answer, and is
# what a second QGIS window gets since both sides set SO_EXCLUSIVEADDRUSE. Windows
# answers WSAEACCES instead when the other holder used plain SO_REUSEADDR, which
# means the same thing here - the spin box floor is 1024, so EACCES cannot be the
# "privileged port" case.
ADDR_IN_USE = frozenset({errno.EADDRINUSE, errno.EACCES, 10048, 10013})
