"""Registry of the handler methods reachable from the socket.

Stdlib-only and free of ``qgis`` imports, like :mod:`wire` and :mod:`errors`.
Must stay Python 3.9-compatible (see ``tests/test_py39_compat.py``).

Why this exists: dispatch used to build a 119-entry ``{"name": self.name}``
dict inside ``_dispatch``, so the table was rebuilt on every single message and
had to be edited in two places for every new handler - the method and the
table. The mapping was identity for all 119 entries, so the table carried no
information the method name didn't already have.

``@command`` marks a method as callable from the wire and is collected at import
time. Dispatch then resolves ``getattr(self, cmd_type)`` after checking
membership. The check is what keeps this safe: without it, a client could name
any attribute of the server - ``stop``, ``_dispatch``, ``clients`` - and reach
it. Only decorated methods are reachable.
"""

COMMANDS = set()


def command(fn):
    """Mark a ``QgisMCPServer`` method as reachable from the socket."""
    COMMANDS.add(fn.__name__)
    return fn
