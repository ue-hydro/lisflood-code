"""OpenWQ water-quality coupling for OS-LISFLOOD (one-way, optional).

The coupler reads LISFLOOD's end-of-step water state and fluxes and advances
the OpenWQ solute balance through the generic ``openwq_py`` pybind11 bridge
(OpenWQ_hydrolink.{h,cpp}). See coupler.py for the compartment/flux mapping and
README.md for the build. Importing this package does NOT import the compiled
``openwq_py`` module -- that happens lazily inside the coupler's setup().
"""
from .coupler import LisfloodOpenWQCoupler, COMPARTMENTS, STORE_VAR

__all__ = ["LisfloodOpenWQCoupler", "COMPARTMENTS", "STORE_VAR"]
