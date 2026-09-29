#!/usr/bin/env python3
"""
Run sequential Pangu-Weather forecasts using Earth2Studio.

Default experiment
------------------
Initialization dates : 2021-11-24 through 2022-04-01, inclusive
Initialization hour  : 00 UTC
Forecast length      : 168 h (28 x 6-hour Pangu steps)
Output               : one NetCDF file per initialization
ERA5 cache            : shared cache reusable by Pangu/GraphCast/Aurora

Example
-------
python run_pangu_cpu_forecasts.py \
    --start 2021-11-24 \
    --end 2022-04-01
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------
# Base configuration
# ---------------------------------------------------------------------

DEFAULT_BASE = Path("/glade/work/sakor/AI_NWP")
DEFAULT_OUTPUT_DIR = Path("/glade/derecho/scratch/sakor/bulk/era5_init/pw")

FORECAST_HOURS = 168
PANGU_STEP_HOURS = 6
NSTEPS = FORECAST_HOURS // PANGU_STEP_HOURS  # 28

# Set cache environment variables BEFORE importing Earth2Studio/model code.
# The ERA5 cache is intentionally shared across AI-NWP models.
BASE = Path(os.environ.get("AI_NWP_BASE", DEFAULT_BASE))
MODEL_CACHE = BASE / "cache" / "models"
ERA5_CACHE = BASE / "cache" / "era5"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
LOG_DIR = BASE / "logs" / "pangu"

for directory in (MODEL_CACHE, ERA5_CACHE, OUTPUT_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

os.environ["EARTH2STUDIO_MODEL_CACHE"] = str(MODEL_CACHE)
os.environ["EARTH2STUDIO_DATA_CACHE"] = str(ERA5_CACHE)
os.environ.setdefault("EARTH2STUDIO_PACKAGE_TIMEOUT", "3600")

# Imports that depend on the cache/environment configuration.
import onnxruntime as ort
import torch
import xarray as xr

import earth2studio
from earth2studio.data import NCAR_ERA5
from earth2studio.io import NetCDF4Backend
from earth2studio.models.px import Pangu6
from earth2studio.run import deterministic


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential Pangu 168-h forecasts from daily 00 UTC ERA5 initializations."
    )
    parser.add_argument(
        "--start",
        default="2021-11-24",
        help="First initialization date, YYYY-MM-DD (inclusive).",
    )
    parser.add_argument(
        "--end",
        default="2022-04-01",
        help="Last initialization date, YYYY-MM-DD (inclusive).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite NetCDF files that already exist. Default is to skip them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If one initialization fails, log it and continue to the next date.",
    )
    parser.add_argument(
        "--base",
        default=str(BASE),
        help=f"Project base directory for caches/logs. Default: {BASE}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Forecast output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def date_range(start: datetime, end: datetime):
    """Yield daily dates from start through end, inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("pangu_batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def cache_stats(path: Path) -> tuple[int, int]:
    """
    Return (number_of_files, total_bytes) for the shared ERA5 cache.

    This is used only once before/after each initialization so the log gives
    a simple indication of whether new cached data appeared.
    """
    nfiles = 0
    nbytes = 0

    if not path.exists():
        return nfiles, nbytes

    for item in path.rglob("*"):
        if item.is_file():
            try:
                stat = item.stat()
                nfiles += 1
                nbytes += stat.st_size
            except OSError:
                pass

    return nfiles, nbytes


def human_bytes(nbytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(nbytes)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def validate_netcdf(path: Path) -> None:
    """Open the completed NetCDF file and perform basic sanity checks."""
    with xr.open_dataset(path) as ds:
        if "lead_time" not in ds.coords and "lead_time" not in ds.dims:
            raise RuntimeError(f"{path} does not contain a lead_time coordinate/dimension.")
        if len(ds.data_vars) == 0:
            raise RuntimeError(f"{path} contains no forecast variables.")


def make_paths(init_dt: datetime, output_dir: Path, log_dir: Path):
    stamp = init_dt.strftime("%Y%m%d_00z")
    final_output = output_dir / f"pangu_{stamp}_168h.nc"
    partial_output = output_dir / f".pangu_{stamp}_168h.partial.nc"
    init_log = log_dir / f"pangu_{stamp}.log"
    return final_output, partial_output, init_log


def append_init_log(path: Path, message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with path.open("a") as handle:
        handle.write(f"{timestamp} | {message}\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # Allow --base to override the defaults used above.
    run_base = Path(args.base)
    model_cache = run_base / "cache" / "models"
    era5_cache = run_base / "cache" / "era5"
    output_dir = Path(args.output_dir)
    log_dir = run_base / "logs" / "pangu"

    for directory in (model_cache, era5_cache, output_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ["EARTH2STUDIO_MODEL_CACHE"] = str(model_cache)
    os.environ["EARTH2STUDIO_DATA_CACHE"] = str(era5_cache)

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if end < start:
        raise ValueError("--end must be on or after --start")

    init_dates = list(date_range(start, end))
    master_log = log_dir / (
        f"pangu_batch_{start:%Y%m%d}_{end:%Y%m%d}.log"
    )
    logger = setup_logger(master_log)

    logger.info("=" * 78)
    logger.info("PANGU-WEATHER FORECAST BATCH")
    logger.info("=" * 78)
    logger.info("Earth2Studio version : %s", earth2studio.__version__)
    logger.info("Python executable    : %s", sys.executable)
    logger.info("PyTorch version      : %s", torch.__version__)
    logger.info("ONNX Runtime version : %s", ort.__version__)
    logger.info("Initialization range : %s 00 UTC -> %s 00 UTC", args.start, args.end)
    logger.info("Number of cycles     : %d", len(init_dates))
    logger.info("Forecast length      : %d h", FORECAST_HOURS)
    logger.info("Pangu step interval  : %d h", PANGU_STEP_HOURS)
    logger.info("Forecast steps       : %d", NSTEPS)
    logger.info("Output directory     : %s", output_dir)
    logger.info("Shared ERA5 cache    : %s", era5_cache)
    logger.info("Model cache          : %s", model_cache)
    logger.info("Master log           : %s", master_log)

    providers = ort.get_available_providers()
    logger.info("ONNX providers       : %s", ", ".join(providers))

    # Automatically use GPU only when BOTH PyTorch and ONNX Runtime
    # have CUDA support. Otherwise fall back cleanly to CPU.
    cuda_available = (
        torch.cuda.is_available()
        and "CUDAExecutionProvider" in providers
    )

    if cuda_available:
        device = torch.device("cuda:0")
        logger.info("Inference device     : %s", device)
        logger.info("CUDA device          : %s", torch.cuda.get_device_name(0))
    else:
        if "CPUExecutionProvider" not in providers:
            raise RuntimeError(
                "Neither a usable CUDAExecutionProvider nor "
                "CPUExecutionProvider is available."
            )
        device = torch.device("cpu")
        logger.info("Inference device     : %s", device)
        if torch.cuda.is_available() and "CUDAExecutionProvider" not in providers:
            logger.warning(
                "PyTorch sees a CUDA GPU, but ONNX Runtime does not provide "
                "CUDAExecutionProvider; falling back to CPU."
            )
        elif not torch.cuda.is_available():
            logger.info("CUDA not available; using CPU.")

    torch.set_default_device(device.type)

    # -----------------------------------------------------------------
    # Load Pangu ONCE and reuse the same model for all initialization dates.
    # -----------------------------------------------------------------
    logger.info("-" * 78)
    logger.info("Loading Pangu6 model once for the entire batch...")
    model_start = time.perf_counter()

    package = Pangu6.load_default_package()
    model = Pangu6.load_model(package)
    model = model.to(device)

    logger.info(
        "Pangu6 loaded in %.1f s.",
        time.perf_counter() - model_start,
    )
    logger.info("Model input variables: %d", len(model.input_coords()["variable"]))

    # -----------------------------------------------------------------
    # Create ONE ERA5 datasource backed by a persistent shared cache.
    #
    # Pangu, GraphCast and Aurora can all point EARTH2STUDIO_DATA_CACHE
    # to the same directory. Cached fields/times that overlap can be reused.
    # A model may still fetch additional fields if its input requirements differ.
    # -----------------------------------------------------------------
    logger.info("-" * 78)
    logger.info("Initializing NCAR ERA5 datasource...")
    data = NCAR_ERA5(cache=True, verbose=True)
    logger.info("ERA5 datasource ready.")

    initial_cache_files, initial_cache_bytes = cache_stats(era5_cache)
    logger.info(
        "ERA5 cache at batch start: %d files, %s",
        initial_cache_files,
        human_bytes(initial_cache_bytes),
    )

    n_complete = 0
    n_skipped = 0
    n_failed = 0
    batch_start = time.perf_counter()

    # -----------------------------------------------------------------
    # Sequential forecast loop
    # -----------------------------------------------------------------
    for index, init_dt in enumerate(init_dates, start=1):
        init_iso = init_dt.strftime("%Y-%m-%dT00:00:00")
        final_output, partial_output, init_log = make_paths(
            init_dt, output_dir, log_dir
        )

        banner = (
            f"[{index:03d}/{len(init_dates):03d}] "
            f"Initialization {init_dt:%Y-%m-%d} 00 UTC"
        )

        logger.info("=" * 78)
        logger.info(banner)
        logger.info("Target output: %s", final_output)
        append_init_log(init_log, "=" * 70)
        append_init_log(init_log, banner)
        append_init_log(init_log, f"Target output: {final_output}")

        # Restart/resume behavior.
        if final_output.exists() and not args.overwrite:
            try:
                validate_netcdf(final_output)
                logger.info("STATUS: SKIPPED -- valid output already exists.")
                append_init_log(init_log, "STATUS: SKIPPED -- valid output already exists.")
                n_skipped += 1
                continue
            except Exception as exc:
                logger.warning(
                    "Existing output failed validation (%s); it will be rerun.",
                    exc,
                )

        if partial_output.exists():
            partial_output.unlink()

        if args.overwrite and final_output.exists():
            final_output.unlink()

        cache_before_files, cache_before_bytes = cache_stats(era5_cache)
        logger.info(
            "ERA5 cache before cycle: %d files, %s",
            cache_before_files,
            human_bytes(cache_before_bytes),
        )
        append_init_log(
            init_log,
            f"ERA5 cache before cycle: {cache_before_files} files, "
            f"{human_bytes(cache_before_bytes)}",
        )

        cycle_start = time.perf_counter()

        try:
            # Write to a temporary file. It is promoted to the final name
            # only after deterministic() returns and the NetCDF validates.
            io = NetCDF4Backend(
                file_name=str(partial_output),
                backend_kwargs={"mode": "w"},
            )

            logger.info(
                "Starting Pangu forecast: %s -> +%d h (%d steps)",
                init_iso,
                FORECAST_HOURS,
                NSTEPS,
            )
            append_init_log(
                init_log,
                f"START forecast {init_iso}; +{FORECAST_HOURS} h; {NSTEPS} steps",
            )

            deterministic(
                time=[init_iso],
                nsteps=NSTEPS,
                prognostic=model,
                data=data,
                io=io,
                device=device,
                verbose=True,
            )

            # Release the backend before validation/rename.
            del io

            validate_netcdf(partial_output)
            partial_output.replace(final_output)

            elapsed = time.perf_counter() - cycle_start
            output_size = final_output.stat().st_size

            cache_after_files, cache_after_bytes = cache_stats(era5_cache)
            new_files = cache_after_files - cache_before_files
            new_bytes = cache_after_bytes - cache_before_bytes

            logger.info("STATUS: COMPLETE")
            logger.info("Cycle runtime        : %.1f s (%.2f min)", elapsed, elapsed / 60.0)
            logger.info("NetCDF size          : %s", human_bytes(output_size))
            logger.info(
                "ERA5 cache after cycle: %d files, %s",
                cache_after_files,
                human_bytes(cache_after_bytes),
            )

            if new_files == 0 and new_bytes == 0:
                logger.info(
                    "ERA5 cache change     : none detected; requested data were likely already cached."
                )
            else:
                logger.info(
                    "ERA5 cache change     : %+d files, %+s",
                    new_files,
                    human_bytes(new_bytes),
                )

            append_init_log(init_log, "STATUS: COMPLETE")
            append_init_log(init_log, f"Cycle runtime: {elapsed:.1f} s")
            append_init_log(init_log, f"NetCDF: {final_output}")
            append_init_log(init_log, f"NetCDF size: {human_bytes(output_size)}")
            append_init_log(
                init_log,
                f"ERA5 cache change: {new_files:+d} files, {human_bytes(new_bytes)}",
            )

            n_complete += 1

        except Exception as exc:
            n_failed += 1
            elapsed = time.perf_counter() - cycle_start

            logger.error("STATUS: FAILED after %.1f s", elapsed)
            logger.error("%s: %s", type(exc).__name__, exc)
            logger.error(traceback.format_exc())

            append_init_log(init_log, f"STATUS: FAILED after {elapsed:.1f} s")
            append_init_log(init_log, f"{type(exc).__name__}: {exc}")
            append_init_log(init_log, traceback.format_exc())

            # Keep the partial file for diagnosis, but it will be removed
            # automatically before a retry of this initialization.
            if not args.continue_on_error:
                logger.error(
                    "Stopping batch. Re-run the same command to resume; "
                    "completed NetCDF files will be skipped."
                )
                return 1

            logger.warning("--continue-on-error is set; moving to next initialization.")

    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------
    elapsed_batch = time.perf_counter() - batch_start
    final_cache_files, final_cache_bytes = cache_stats(era5_cache)

    logger.info("=" * 78)
    logger.info("BATCH SUMMARY")
    logger.info("=" * 78)
    logger.info("Requested cycles : %d", len(init_dates))
    logger.info("Completed        : %d", n_complete)
    logger.info("Skipped existing : %d", n_skipped)
    logger.info("Failed           : %d", n_failed)
    logger.info(
        "Batch runtime     : %.2f h",
        elapsed_batch / 3600.0,
    )
    logger.info(
        "ERA5 cache final : %d files, %s",
        final_cache_files,
        human_bytes(final_cache_bytes),
    )
    logger.info("Outputs          : %s", output_dir)
    logger.info("Logs             : %s", log_dir)
    logger.info("=" * 78)

    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
