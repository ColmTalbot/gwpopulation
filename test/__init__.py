import importlib

from gwpopulation import backend as _backend

TEST_BACKENDS = list()
for backend in _backend.SUPPORTED_BACKENDS:
    try:
        importlib.import_module(backend)
    except ImportError:
        continue
    TEST_BACKENDS.append(backend)
