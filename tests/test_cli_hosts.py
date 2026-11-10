"""Which game hosts are reached without TLS."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tne_sdk.cli import _is_local_host


@pytest.mark.parametrize("host", [
    "localhost", "localhost:8000", "127.0.0.1:8000", "127.0.0.1", "192.168.1.50:8000", "10.0.0.7",
    "172.16.4.2:9000", "rig.local:8000", "host.docker.internal:8000", "::1",
])
def test_this_machine_and_the_local_network_are_plain(host):
    assert _is_local_host(host) is True


@pytest.mark.parametrize("host", ["api.null.firespawn.ai", "example.com:443", "8.8.8.8:8000", "localhost.evil.com"])
def test_everything_else_is_secure(host):
    assert _is_local_host(host) is False
