# pint setup
import os
import warnings

os.environ['PINT_ARRAY_PROTOCOL_FALLBACK'] = "0"

from pint import UnitRegistry, set_application_registry

ureg = UnitRegistry()
ureg.setup_matplotlib(True)
set_application_registry(ureg)

Q_ = ureg.Quantity

with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  Q_([])

# local imports
from .features import *
from .plot import *
from .line_delay import *
from .notch import *
