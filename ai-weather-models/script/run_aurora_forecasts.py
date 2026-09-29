#!/usr/bin/env python3
"""
Run sequential Microsoft Aurora forecasts with Earth2Studio.

Default experiment
------------------
Initialization dates : 2021-11-24 through 2022-04-01, inclusive
Initialization hour  : 00 UTC
Forecast length      : 168 h (28 x 6-hour Aurora steps)
Output               : one NetCDF file per initialization
ERA5 cache           : shared with Pangu / GraphCast / other AI-NWP models

Aurora input history
--------------------
Aurora requires two ERA5 analysis frames for every forecast initialization:
    t - 6 h and t0

Earth2Studio determines these required times from Aurora's input coordinates.
The shared NCAR_ERA5 cache avoids downloading already cached variables/times again.

Example
-------
python run_aurora_gpu_forecasts.py \
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
# Configuration
# ---------------------------------------------------------------------

DEFAULT_BASE = Path("/glade/work/sakor/AI_NWP")
DEFAULT_OUTPUT_DIR = Path(
    "/glade/derecho/scratch/sakor/bulk/era5_init/aurora"
)

FORECAST_HOURS = 168
AURORA_STEP_HOURS = 6
NSTEPS = FORECAST_HOURS // AURORA_STEP_HOURS  # 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential Aurora 168-h forecasts from daily 00 UTC "
            "NCAR ERA5 initializations with automatic CPU/GPU selection."
        )
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
        "--base",
        default=str(DEFAULT_BASE),
        help=f"Project base directory for caches/logs. Default: {DEFAULT_BASE}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Forecast output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite valid NetCDF files that already exist.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log a failed initialization and continue to the next date.",
    )
    parser.add_argument(
        "--no-autocast",
        action="store_true",
        help=(
            "Disable Aurora BF16 backbone autocast when running on CUDA. "
            "Autocast is never enabled for CPU runs."
        ),
    )
    return parser.parse_args()


def date_range(start: datetime, end: datetime):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("aurora_batch")
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
    """Return (number of files, total bytes) in the shared ERA5 cache."""
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


def make_paths(
    init_dt: datetime,
    output_dir: Path,
    log_dir: Path,
) -> tuple[Path, Path, Path]:
    stamp = init_dt.strftime("%Y%m%d_00z")
    final_output = output_dir / f"aurora_{stamp}_168h.nc"
    partial_output = output_dir / f".aurora_{stamp}_168h.partial.nc"
    init_log = log_dir / f"aurora_{stamp}.log"
    return final_output, partial_output, init_log


def append_init_log(path: Path, message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with path.open("a") as handle:
        handle.write(f"{timestamp} | {message}\n")


def main() -> int:
    args = parse_args()

    # -----------------------------------------------------------------
    # Shared directories and Earth2Studio cache configuration.
    #
    # IMPORTANT:
    # This is the SAME ERA5 cache used by Pangu. GraphCast and other
    # Earth2Studio models should point here as well.
    # -----------------------------------------------------------------
    base = Path(args.base)
    model_cache = base / "cache" / "models"
    era5_cache = base / "cache" / "era5"
    output_dir = Path(args.output_dir)
    log_dir = base / "logs" / "aurora"

    for directory in (model_cache, era5_cache, output_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Set BEFORE importing Earth2Studio.
    os.environ["EARTH2STUDIO_MODEL_CACHE"] = str(model_cache)
    os.environ["EARTH2STUDIO_DATA_CACHE"] = str(era5_cache)
    os.environ.setdefault("EARTH2STUDIO_PACKAGE_TIMEOUT", "3600")

    # Helpful for long GPU rollouts with variable-sized allocations.
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )

    # Model/package imports occur only after cache environment is configured.
    import torch
    import xarray as xr

    import earth2studio
    from earth2studio.data import NCAR_ERA5
    from earth2studio.io import NetCDF4Backend
    from earth2studio.models.px import Aurora
    from earth2studio.run import deterministic

    def validate_netcdf(path: Path) -> None:
        with xr.open_dataset(path) as ds:
            if "lead_time" not in ds.coords and "lead_time" not in ds.dims:
                raise RuntimeError(
                    f"{path} does not contain a lead_time coordinate/dimension."
                )
            if len(ds.data_vars) == 0:
                raise RuntimeError(f"{path} contains no forecast variables.")

    # -----------------------------------------------------------------
    # Initialization dates
    # -----------------------------------------------------------------
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )

    if end < start:
        raise ValueError("--end must be on or after --start")

    init_dates = list(date_range(start, end))

    master_log = log_dir / (
        f"aurora_batch_{start:%Y%m%d}_{end:%Y%m%d}.log"
    )
    logger = setup_logger(master_log)

    # -----------------------------------------------------------------
    # Automatic device selection
    # -----------------------------------------------------------------
    logger.info("=" * 78)
    logger.info("AURORA FORECAST BATCH")
    logger.info("=" * 78)
    logger.info("Earth2Studio version : %s", earth2studio.__version__)
    logger.info("Python executable    : %s", sys.executable)
    logger.info("PyTorch version      : %s", torch.__version__)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        props = torch.cuda.get_device_properties(device)
        total_vram = props.total_memory

        logger.info("Inference device     : %s", device)
        logger.info("GPU                  : %s", props.name)
        logger.info("GPU VRAM             : %s", human_bytes(total_vram))
        logger.info("CUDA runtime         : %s", torch.version.cuda)
    else:
        device = torch.device("cpu")
        logger.info("Inference device     : %s", device)
        logger.info("CUDA available       : False")
        logger.info("Running Aurora on CPU.")
    logger.info(
        "Initialization range : %s 00 UTC -> %s 00 UTC",
        args.start,
        args.end,
    )
    logger.info("Number of cycles     : %d", len(init_dates))
    logger.info("Forecast length      : %d h", FORECAST_HOURS)
    logger.info("Aurora step interval : %d h", AURORA_STEP_HOURS)
    logger.info("Forecast steps       : %d", NSTEPS)
    logger.info("Output directory     : %s", output_dir)
    logger.info("Shared ERA5 cache    : %s", era5_cache)
    logger.info("Model cache          : %s", model_cache)
    logger.info("Master log           : %s", master_log)

    # Enable CUDA-specific matrix optimizations only on GPU.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # -----------------------------------------------------------------
    # Load Aurora ONCE and reuse it for every initialization.
    # -----------------------------------------------------------------
    logger.info("-" * 78)
    logger.info("Loading Aurora model once for the entire batch...")
    model_start = time.perf_counter()

    package = Aurora.load_default_package()
    model = Aurora.load_model(package)

    # Enable Aurora BF16 backbone autocast only for CUDA runs.
    # CPU runs remain in the model's normal precision.
    if device.type == "cuda" and not args.no_autocast:
        if hasattr(model, "model"):
            if hasattr(model.model, "autocast"):
                model.model.autocast = True
            if hasattr(model.model, "autocast_backbone"):
                model.model.autocast_backbone = True
            if hasattr(model.model, "autocast_dtype"):
                model.model.autocast_dtype = torch.bfloat16
            logger.info("Aurora BF16 backbone autocast: ENABLED")
        else:
            logger.warning(
                "Could not access Aurora core model; autocast setting was not changed."
            )
    elif device.type == "cuda":
        logger.info("Aurora BF16 backbone autocast: DISABLED by command line")
    else:
        logger.info("Aurora BF16 backbone autocast: not used on CPU")

    model = model.to(device)
    model.eval()

    logger.info(
        "Aurora loaded in %.1f s.",
        time.perf_counter() - model_start,
    )

    input_coords = model.input_coords()
    logger.info(
        "Model input variables : %d",
        len(input_coords["variable"]),
    )
    logger.info(
        "Required history      : %s",
        input_coords["lead_time"],
    )

    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        logger.info("GPU allocated after model load : %s", human_bytes(allocated))
        logger.info("GPU reserved after model load  : %s", human_bytes(reserved))

    # -----------------------------------------------------------------
    # ONE NCAR ERA5 datasource using the SAME persistent cache as Pangu.
    #
    # NCAR_ERA5(cache=True) checks Earth2Studio's data cache before
    # downloading remote data. Already cached variable/time combinations
    # are reused; only missing Aurora requirements are fetched.
    # -----------------------------------------------------------------
    logger.info("-" * 78)
    logger.info("Initializing NCAR ERA5 datasource...")
    data = NCAR_ERA5(cache=True, verbose=True)
    logger.info("ERA5 datasource ready.")

    initial_cache_files, initial_cache_bytes = cache_stats(era5_cache)
    logger.info(
        "Shared ERA5 cache at batch start: %d files, %s",
        initial_cache_files,
        human_bytes(initial_cache_bytes),
    )

    n_complete = 0
    n_skipped = 0
    n_failed = 0
    batch_start = time.perf_counter()

    # -----------------------------------------------------------------
    # Sequential daily forecast loop
    # -----------------------------------------------------------------
    for index, init_dt in enumerate(init_dates, start=1):
        init_iso = init_dt.strftime("%Y-%m-%dT00:00:00")
        history_iso = (init_dt - timedelta(hours=6)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

        final_output, partial_output, init_log = make_paths(
            init_dt,
            output_dir,
            log_dir,
        )

        banner = (
            f"[{index:03d}/{len(init_dates):03d}] "
            f"Initialization {init_dt:%Y-%m-%d} 00 UTC"
        )

        logger.info("=" * 78)
        logger.info(banner)
        logger.info("Aurora input history: %s and %s", history_iso, init_iso)
        logger.info("Target output       : %s", final_output)

        append_init_log(init_log, "=" * 70)
        append_init_log(init_log, banner)
        append_init_log(
            init_log,
            f"Aurora input history: {history_iso} and {init_iso}",
        )
        append_init_log(init_log, f"Target output: {final_output}")

        # -------------------------------------------------------------
        # Restart/resume: valid completed output means no rerun.
        # -------------------------------------------------------------
        if final_output.exists() and not args.overwrite:
            try:
                validate_netcdf(final_output)
                logger.info(
                    "STATUS: SKIPPED -- valid output already exists."
                )
                append_init_log(
                    init_log,
                    "STATUS: SKIPPED -- valid output already exists.",
                )
                n_skipped += 1
                continue
            except Exception as exc:
                logger.warning(
                    "Existing output failed validation (%s); rerunning.",
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
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            # Write to a temporary NetCDF first. It gets renamed only
            # after successful completion and validation.
            io = NetCDF4Backend(
                file_name=str(partial_output),
                backend_kwargs={"mode": "w"},
            )

            logger.info(
                "Starting Aurora forecast: %s -> +%d h (%d steps)",
                init_iso,
                FORECAST_HOURS,
                NSTEPS,
            )
            append_init_log(
                init_log,
                f"START forecast {init_iso}; "
                f"+{FORECAST_HOURS} h; {NSTEPS} steps",
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

            del io

            # Make sure queued GPU work has finished before timing/logging.
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            validate_netcdf(partial_output)
            partial_output.replace(final_output)

            elapsed = time.perf_counter() - cycle_start
            output_size = final_output.stat().st_size

            peak_vram = (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            )

            cache_after_files, cache_after_bytes = cache_stats(era5_cache)
            new_files = cache_after_files - cache_before_files
            new_bytes = cache_after_bytes - cache_before_bytes

            logger.info("STATUS: COMPLETE")
            logger.info(
                "Cycle runtime        : %.1f s (%.2f min)",
                elapsed,
                elapsed / 60.0,
            )
            if peak_vram is not None:
                logger.info("Peak GPU memory      : %s", human_bytes(peak_vram))
            logger.info("NetCDF size          : %s", human_bytes(output_size))
            logger.info(
                "ERA5 cache after cycle: %d files, %s",
                cache_after_files,
                human_bytes(cache_after_bytes),
            )

            if new_files == 0 and new_bytes == 0:
                logger.info(
                    "ERA5 cache change     : none detected; requested "
                    "initialization data were already cached."
                )
            else:
                logger.info(
                    "ERA5 cache change     : %+d files, %s added",
                    new_files,
                    human_bytes(max(new_bytes, 0)),
                )

            append_init_log(init_log, "STATUS: COMPLETE")
            append_init_log(init_log, f"Cycle runtime: {elapsed:.1f} s")
            if peak_vram is not None:
                append_init_log(
                    init_log,
                    f"Peak GPU memory: {human_bytes(peak_vram)}",
                )
            append_init_log(init_log, f"NetCDF: {final_output}")
            append_init_log(
                init_log,
                f"NetCDF size: {human_bytes(output_size)}",
            )
            append_init_log(
                init_log,
                f"ERA5 cache change: {new_files:+d} files, "
                f"{human_bytes(max(new_bytes, 0))} added",
            )

            n_complete += 1

            # Release unused CUDA allocator blocks before the next date.
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as exc:
            n_failed += 1
            elapsed = time.perf_counter() - cycle_start

            logger.error("STATUS: FAILED after %.1f s", elapsed)
            logger.error("%s: %s", type(exc).__name__, exc)
            logger.error(traceback.format_exc())

            append_init_log(
                init_log,
                f"STATUS: FAILED after {elapsed:.1f} s",
            )
            append_init_log(
                init_log,
                f"{type(exc).__name__}: {exc}",
            )
            append_init_log(init_log, traceback.format_exc())

            if device.type == "cuda":
                torch.cuda.empty_cache()

            if not args.continue_on_error:
                logger.error(
                    "Stopping batch. Re-submit the same command to resume; "
                    "completed NetCDF files will be skipped."
                )
                return 1

            logger.warning(
                "--continue-on-error is set; moving to next initialization."
            )

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
    logger.info("Batch runtime     : %.2f h", elapsed_batch / 3600.0)
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
