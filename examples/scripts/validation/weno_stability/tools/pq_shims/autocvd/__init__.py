"""Job-local stand-in for autocvd under the pq queue.

pq already assigns CUDA_VISIBLE_DEVICES; the real autocvd ignores that, waits
for a GPU that is free on the whole node and overrides the assignment. Put
this directory first on PYTHONPATH for queued jobs only.
"""
import os


def autocvd(*args, **kwargs):
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("pq autocvd shim used outside a queued job")
