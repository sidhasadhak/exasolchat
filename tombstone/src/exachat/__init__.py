"""exachat has been renamed to talonsight.

This package is a compatibility shim — it re-exports everything from talonsight
so existing code keeps working, but emits a DeprecationWarning to prompt migration.

Migrate with:
    pip uninstall exachat
    pip install talonsight
"""

import warnings

warnings.warn(
    "\n\nexachat has been renamed to talonsight.\n"
    "Please migrate:\n"
    "    pip uninstall exachat\n"
    "    pip install talonsight\n"
    "The 'exachat' package will not receive further updates.\n",
    DeprecationWarning,
    stacklevel=2,
)

from talonsight import *  # noqa: F401, F403
from talonsight import TalonSight, QueryResult, ConnectionConfig  # noqa: F401

# Legacy alias kept for any code using the old class name
ExasolChat = TalonSight  # noqa: F401
