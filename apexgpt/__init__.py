"""ApexGPT - a GPT-style transformer language model built from scratch.

Architecture
------------
``core``       cross-cutting concerns: config, paths, device, seeding
``models``     the M in MVC: the GPT architecture and sampling primitives
``features``   feature-based vertical slices, each with a service + a view
``api``        optional deployable HTTP inference service

Each feature slice is a self-contained vertical: ``features/<name>/service.py``
holds the business logic and ``features/<name>/cli.py`` (or ``gui.py``) is the
view/controller that drives it. Nothing in ``features`` imports from another
feature, so slices can be extracted into standalone services later.
"""

__version__ = "1.1.0"

__all__ = ["__version__"]
