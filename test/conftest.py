import importlib

import pytest

from gwpopulation import backend as _backend

TEST_BACKENDS = list()
for backend in _backend.SUPPORTED_BACKENDS:
    try:
        importlib.import_module(backend)
    except ImportError:
        continue
    TEST_BACKENDS.append(backend)


@pytest.fixture(params=TEST_BACKENDS)
def backend(request):
    pytest.importorskip(request.param)
    return request.param
