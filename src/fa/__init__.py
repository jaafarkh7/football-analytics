"""Football analytics package."""
import os

# Our matrices are small; multi-threaded BLAS makes them ~20x slower, not faster.
# Must be set before numpy is imported, which is why it lives here.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
