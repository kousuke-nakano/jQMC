"""Workflow-level constants mirroring jqmc.setting for validation.

These thresholds are used by the workflow layer to reject invalid
parameters *before* submitting an HPC job, avoiding wasted queue time.
"""

# Minimal post-processing parameters (GFMC / LRDMC)
GFMC_MIN_WARMUP_STEPS = 25
GFMC_MIN_BIN_BLOCKS = 10
GFMC_MIN_COLLECT_STEPS = 5
