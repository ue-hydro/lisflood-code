"""OpenWQ water-quality coupling for OS-LISFLOOD (one-way, optional).

The coupler reads LISFLOOD's water stores and transfers each step and advances
the OpenWQ solute balance through the generic ``openwq_py`` pybind11 bridge
(OpenWQ_hydrolink.{h,cpp}), either in-process (option ``OpenWQ``) or offline
from a recording (option ``OpenWQRecord`` + replay.py). See coupler.py for the
compartment/flux mapping and README.md for the build. Importing this package
does NOT import the compiled ``openwq_py`` module -- that happens lazily inside
the coupler's setup().
"""
from .coupler import (LisfloodOpenWQCoupler, LisfloodOpenWQRecorder, COMPARTMENTS,
                      TRANSFERS)

__all__ = ["LisfloodOpenWQCoupler", "LisfloodOpenWQRecorder", "COMPARTMENTS", "TRANSFERS"]
