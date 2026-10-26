"""Appends design-space exploration results to `dse.csv` in the directory that `ZIG_CU_KERNELS_DSE_DIR`
names, and logs nothing when the variable is unset."""

import csv
import fcntl
import os
import time
from pathlib import Path

DIRECTORY_VARIABLE = "ZIG_CU_KERNELS_DSE_DIR"
DESIGN_VARIABLES = {
    "based_on": "ZIG_CU_KERNELS_DSE_BASED_ON",
    "axis": "ZIG_CU_KERNELS_DSE_AXIS",
    "value": "ZIG_CU_KERNELS_DSE_VALUE",
}
FILE_NAME = "dse.csv"
FIELDS = ("unix_time", "kernel", "based_on", "axis", "value", "measurement", "unit", "gpu_temperature_c")


def directory():
    path = os.environ.get(DIRECTORY_VARIABLE)
    return Path(path) if path else None


def enabled():
    return directory() is not None


def log(kernel, measurement, unit, based_on=None, axis=None, value=None, temperature_c=None):
    folder = directory()
    if folder is None:
        return False
    given = {"based_on": based_on, "axis": axis, "value": value}
    design = {field: os.environ.get(variable, "") if given[field] is None else given[field]
              for field, variable in DESIGN_VARIABLES.items()}
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / FILE_NAME, "a+", newline="") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.seek(0)
        header = file.readline().rstrip("\r\n")
        if header and header != ",".join(FIELDS):
            raise ValueError(f"{file.name} has the columns {header}; set {DIRECTORY_VARIABLE} to a new directory")
        writer = csv.DictWriter(file, FIELDS)
        if not header:
            writer.writeheader()
        writer.writerow({"unix_time": f"{time.time():.3f}", "kernel": kernel, **design, "measurement": measurement,
                         "unit": unit, "gpu_temperature_c": "" if temperature_c is None else f"{temperature_c:.1f}"})
    return True
