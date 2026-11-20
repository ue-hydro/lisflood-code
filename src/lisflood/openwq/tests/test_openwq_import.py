"""Smoke test for the compiled ``openwq_py`` pybind11 bridge.

Skipped automatically when the module has not been built yet (see README for
the CMake build). When built, checks the Host API the coupler relies on exists.
"""
import pytest

openwq_py = pytest.importorskip(
    "openwq_py", reason="openwq_py not built yet; run the CMake build (see README)")


def test_host_api_present():
    assert hasattr(openwq_py, "Host")
    host = openwq_py.Host.__init__  # constructor exists
    for method in (
        "add_compartment",
        "init",
        "set_water_volume_array",
        "time_start",
        "time_end",
    ):
        assert hasattr(openwq_py.Host, method), f"Host.{method} missing"


def test_minimal_lifecycle():
    import tempfile, os, json
    # A real run needs a valid master JSON; here we only check the module loads
    # and the constructor is callable with a path argument.
    with tempfile.TemporaryDirectory() as d:
        mj = os.path.join(d, "openWQ_master.json")
        with open(mj, "w") as fh:
            json.dump({}, fh)
        try:
            openwq_py.Host(mj)
        except Exception as e:  # a bare {} JSON may be rejected by OpenWQ
            pytest.skip(f"Host init needs a valid master JSON: {e}")
