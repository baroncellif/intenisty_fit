"""Physical and SPCAT constants used by the intensity model."""

C2 = 1.438776877
# Catalogue temperature assumed only when a species has no readable .int TEMP
# (SPCAT's default).  Normally each species uses the TEMP of its own .int.
DEFAULT_CAT_TEMPERATURE_K = 300.0
CM1_PER_MHZ = 1.0 / 29979.2458
