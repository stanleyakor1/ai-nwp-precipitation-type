#!/usr/bin/env python3
"""
Run sequential GraphCast Operational forecasts with Earth2Studio.

Default experiment
------------------
Initialization dates : 2021-11-24 through 2022-04-01, inclusive
Initialization hour  : 00 UTC
Forecast length      : 168 h (28 x 6-hour GraphCast steps)
Output               : one NetCDF file per initialization
ERA5 cache           : shared with Pangu / Aurora / other AI-NWP models

GraphCast input history
-----------------------
GraphCast Operational requires two ERA5 analysis frames for every forecast
initialization: t - 6 h and t0.

Earth2Studio determines the required variables/history from the model input
coordinates. The shared NCAR_ERA5 cache avoids downloading already cached
variables/times again.

Device behavior
---------------
GraphCast runs internally with JAX/XLA. This runner uses CUDA only when BOTH
PyTorch and JAX can see a GPU. Otherwise it falls back to CPU.

Examples
--------
python run_graphcast_forecasts.py \
    --start 2021-11-24 \
    --end 2022-04-01

python run_graphcast_forecasts.py \
    --start 2021-11-24 \
    --end 2021-11-24 \
    --forecast-hours 6
"""

from __future__ import annotations

import argparse
import inspect
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
    "/glade/derecho/scratch/sakor/bulk/era5_init/graphcast"
)

DEFAULT_FORECAST_HOURS = 168
GRAPHCAST_STEP_HOURS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential GraphCast Operational forecasts from daily 00 UTC "
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
        "--forecast-hours",
        type=int,
        default=DEFAULT_FORECAST_HOURS,
        help=(
            "Forecast length in hours. Must be a positive multiple of "
            f"{GRAPHCAST_STEP_HOURS}. Default: {DEFAULT_FORECAST_HOURS}"
        ),
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
    return parser.parse_args()


def date_range(start: datetime, end: datetime):
    """Yield daily initialization times from start through end, inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def setup_logger(log_file: Path) -> logging.Logger:
    """Create a logger that writes to stdout and the master log."""
    logger = logging.getLogger("graphcast_batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

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
    """Convert bytes to a human-readable value."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(nbytes)

    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{value:.2f} TiB"


def make_paths(
    init_dt: datetime,
    forecast_hours: int,
    output_dir: Path,
    log_dir: Path,
) -> tuple[Path, Path, Path]:
    stamp = init_dt.strftime("%Y%m%d_00z")
    final_output = output_dir / f"graphcast_{stamp}_{forecast_hours}h.nc"
    partial_output = (
        output_dir / f".graphcast_{stamp}_{forecast_hours}h.partial.nc"
    )
    init_log = log_dir / f"graphcast_{stamp}.log"
    return final_output, partial_output, init_log


def append_init_log(path: Path, message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with path.open("a") as handle:
        handle.write(f"{timestamp} | {message}\n")


def main() -> int:
    args = parse_args()

    # -----------------------------------------------------------------
    # Forecast configuration
    # -----------------------------------------------------------------
    if args.forecast_hours <= 0:
        raise ValueError("--forecast-hours must be greater than zero.")

    if args.forecast_hours % GRAPHCAST_STEP_HOURS != 0:
        raise ValueError(
            f"--forecast-hours must be a multiple of {GRAPHCAST_STEP_HOURS} hours."
        )

    nsteps = args.forecast_hours // GRAPHCAST_STEP_HOURS

    # -----------------------------------------------------------------
    # Shared directories and Earth2Studio cache configuration.
    #
    # IMPORTANT:
    # This is the SAME ERA5 cache used by Pangu and Aurora.
    # -----------------------------------------------------------------
    base = Path(args.base)
    model_cache = base / "cache" / "models"
    era5_cache = base / "cache" / "era5"
    output_dir = Path(args.output_dir)
    log_dir = base / "logs" / "graphcast"

    for directory in (model_cache, era5_cache, output_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Set BEFORE importing Earth2Studio.
    os.environ["EARTH2STUDIO_MODEL_CACHE"] = str(model_cache)
    os.environ["EARTH2STUDIO_DATA_CACHE"] = str(era5_cache)
    os.environ.setdefault("EARTH2STUDIO_PACKAGE_TIMEOUT", "3600")

    # Prevent JAX from reserving nearly all GPU memory immediately.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    # -----------------------------------------------------------------
    # Imports after runtime/cache configuration
    # -----------------------------------------------------------------
    import jax
    import jaxlib
    import numpy as np
    import torch
    import xarray as xr

    import earth2studio
    from earth2studio.data import NCAR_ERA5
    from earth2studio.io import NetCDF4Backend
    from earth2studio.models.px import GraphCastOperational
    from earth2studio.run import deterministic
    from earth2studio.lexicon.ncar import NCAR_ERA5Lexicon

    def validate_netcdf(path: Path) -> None:
        with xr.open_dataset(path) as ds:
            if "lead_time" not in ds.coords and "lead_time" not in ds.dims:
                raise RuntimeError(
                    f"{path} does not contain a lead_time coordinate/dimension."
                )
            if len(ds.data_vars) == 0:
                raise RuntimeError(f"{path} contains no forecast variables.")

    def patch_ncar_vertical_velocity_lexicon() -> list[str]:
        """
        Compatibility patch for Earth2Studio 0.11.x.

        Earth2Studio >=0.12 expanded the NCAR ERA5 lexicon to include
        pressure-level vertical velocity (ERA5 parameter 135, short name w).
        GraphCastOperational requires these fields. Older Earth2Studio
        releases may therefore raise KeyError('w50') even though the NCAR
        ERA5 archive contains the data.

        The mapping below follows the current official Earth2Studio NCAR
        lexicon exactly. It modifies only the in-memory lexicon for this
        process; it does not edit the installed package.
        """
        pressure_levels = [
            1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175,
            200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650,
            700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000,
        ]

        added = []

        for level_index, pressure_level in enumerate(pressure_levels):
            key = f"w{pressure_level}"

            if key not in NCAR_ERA5Lexicon.VOCAB:
                NCAR_ERA5Lexicon.VOCAB[key] = (
                    f"e5.oper.an.pl::128_135_w::ll025sc::{level_index}"
                )
                added.append(key)

        return added

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
        f"graphcast_batch_{start:%Y%m%d}_{end:%Y%m%d}.log"
    )
    logger = setup_logger(master_log)

    # -----------------------------------------------------------------
    # Automatic device selection
    #
    # GraphCast is JAX-based. CUDA is used only when BOTH PyTorch and JAX
    # can see a GPU. Otherwise the runner falls back to CPU.
    # -----------------------------------------------------------------
    logger.info("=" * 78)
    logger.info("GRAPHCAST OPERATIONAL FORECAST BATCH")
    logger.info("=" * 78)
    logger.info("Earth2Studio version : %s", earth2studio.__version__)
    logger.info("Python executable    : %s", sys.executable)
    logger.info("PyTorch version      : %s", torch.__version__)
    logger.info("JAX version          : %s", jax.__version__)
    logger.info("JAXLIB version       : %s", jaxlib.__version__)

    # Avoid probing CUDA/JAX-GPU on CPU-only PBS allocations. CUDA-enabled
    # PyTorch/JAX installations can otherwise emit noisy driver/plugin errors
    # even though a CPU run is perfectly valid.
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    has_nvidia_device = any(Path("/dev").glob("nvidia[0-9]*"))
    gpu_allocation_visible = (
        cuda_visible not in ("", "-1", "NoDevFiles")
        or has_nvidia_device
    )

    if gpu_allocation_visible:
        torch_cuda_available = torch.cuda.is_available()
    else:
        torch_cuda_available = False
        logger.info(
            "No GPU allocation/device detected; skipping CUDA and JAX-GPU probes."
        )

    if torch_cuda_available:
        try:
            jax_gpu_devices = list(jax.devices("gpu"))
            jax_gpu_error = None
        except Exception as exc:
            jax_gpu_devices = []
            jax_gpu_error = repr(exc)
    else:
        jax_gpu_devices = []
        jax_gpu_error = None

    use_gpu = torch_cuda_available and bool(jax_gpu_devices)

    if use_gpu:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        props = torch.cuda.get_device_properties(device)
        logger.info("Inference device     : %s", device)
        logger.info("PyTorch GPU          : %s", props.name)
        logger.info("GPU VRAM             : %s", human_bytes(props.total_memory))
        logger.info("Torch CUDA runtime   : %s", torch.version.cuda)
        logger.info("JAX GPU device       : %s", jax_gpu_devices[0])
    else:
        device = torch.device("cpu")
        logger.info("Inference device     : %s", device)
        logger.info("Torch CUDA available : %s", torch_cuda_available)
        logger.info("JAX GPU devices      : %s", jax_gpu_devices)

        if jax_gpu_error is not None:
            logger.info("JAX GPU check        : %s", jax_gpu_error)

        logger.info(
            "Running GraphCast on CPU because no common PyTorch/JAX GPU "
            "is available."
        )

    logger.info(
        "Initialization range : %s 00 UTC -> %s 00 UTC",
        args.start,
        args.end,
    )
    logger.info("Number of cycles     : %d", len(init_dates))
    logger.info("Forecast length      : %d h", args.forecast_hours)
    logger.info("GraphCast step       : %d h", GRAPHCAST_STEP_HOURS)
    logger.info("Forecast steps       : %d", nsteps)
    logger.info("Output directory     : %s", output_dir)
    logger.info("Shared ERA5 cache    : %s", era5_cache)
    logger.info("Model cache          : %s", model_cache)
    logger.info("Master log           : %s", master_log)

    # -----------------------------------------------------------------
    # Load GraphCast ONCE and reuse it for every initialization.
    # -----------------------------------------------------------------
    logger.info("-" * 78)
    logger.info("Loading GraphCastOperational model once for the entire batch...")
    model_start = time.perf_counter()

    package = GraphCastOperational.load_default_package()
    model = GraphCastOperational.load_model(package)
    model.eval()

    logger.info(
        "GraphCastOperational loaded in %.1f s.",
        time.perf_counter() - model_start,
    )

    input_coords = model.input_coords()
    model_variables = [str(v) for v in input_coords["variable"]]

    logger.info("Model input variables : %d", len(model_variables))
    logger.info("Required history      : %s", input_coords["lead_time"])

    # -------------------------------------------------------------
    # Verify GraphCast <-> NCAR ERA5 variable compatibility.
    #
    # Earth2Studio 0.11.x can be missing the pressure-level vertical
    # velocity mappings required by GraphCastOperational. Patch those
    # mappings in memory using the official newer Earth2Studio mapping,
    # then verify that every GraphCast input is supported.
    # -------------------------------------------------------------
    missing_before = [
        variable
        for variable in model_variables
        if variable not in NCAR_ERA5Lexicon.VOCAB
    ]

    if missing_before:
        logger.warning(
            "NCAR ERA5 lexicon is missing %d GraphCast input(s): %s",
            len(missing_before),
            ", ".join(missing_before),
        )

    added_w_variables = patch_ncar_vertical_velocity_lexicon()

    if added_w_variables:
        logger.warning(
            "Applied Earth2Studio 0.11.x NCAR vertical-velocity "
            "compatibility patch for %d variables.",
            len(added_w_variables),
        )
        logger.info(
            "Added NCAR ERA5 lexicon variables: %s",
            ", ".join(added_w_variables),
        )

    missing_after = [
        variable
        for variable in model_variables
        if variable not in NCAR_ERA5Lexicon.VOCAB
    ]

    if missing_after:
        raise RuntimeError(
            "NCAR_ERA5 is still missing GraphCast-required variables after "
            "the compatibility patch: "
            + ", ".join(missing_after)
        )

    logger.info(
        "NCAR ERA5 compatibility: all %d GraphCast inputs are supported.",
        len(model_variables),
    )

    history_offsets_hours: list[int] = []
    for offset in input_coords["lead_time"]:
        offset_hours = int(offset / np.timedelta64(1, "h"))
        history_offsets_hours.append(offset_hours)

    logger.info("History offsets (h)   : %s", history_offsets_hours)

    # -----------------------------------------------------------------
    # ONE NCAR ERA5 datasource using the SAME persistent cache as Pangu
    # and Aurora.
    #
    # Already cached variable/time combinations are reused. Only missing
    # GraphCast initialization requirements are fetched.
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

    # -----------------------------------------------------------------
    # Sequential daily forecast loop
    # -----------------------------------------------------------------
    n_complete = 0
    n_skipped = 0
    n_failed = 0
    batch_start = time.perf_counter()

    for index, init_dt in enumerate(init_dates, start=1):
        init_iso = init_dt.strftime("%Y-%m-%dT00:00:00")

        required_times = [
            init_dt + timedelta(hours=offset_hours)
            for offset_hours in history_offsets_hours
        ]
        required_times_text = ", ".join(
            dt.strftime("%Y-%m-%d %H UTC") for dt in required_times
        )

        final_output, partial_output, init_log = make_paths(
            init_dt=init_dt,
            forecast_hours=args.forecast_hours,
            output_dir=output_dir,
            log_dir=log_dir,
        )

        banner = (
            f"[{index:03d}/{len(init_dates):03d}] "
            f"Initialization {init_dt:%Y-%m-%d} 00 UTC"
        )

        logger.info("=" * 78)
        logger.info(banner)
        logger.info("GraphCast input history: %s", required_times_text)
        logger.info("Target output          : %s", final_output)

        append_init_log(init_log, "=" * 70)
        append_init_log(init_log, banner)
        append_init_log(
            init_log,
            f"GraphCast input history: {required_times_text}",
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
                append_init_log(
                    init_log,
                    f"Existing output failed validation: {exc}",
                )

        if partial_output.exists():
            partial_output.unlink()

        if args.overwrite and final_output.exists():
            final_output.unlink()

        # -------------------------------------------------------------
        # Shared ERA5 cache state before this cycle
        # -------------------------------------------------------------
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
            # Write to a temporary NetCDF. Promote to the final name only
            # after successful completion and validation.
            io = NetCDF4Backend(
                file_name=str(partial_output),
                backend_kwargs={"mode": "w"},
            )

            logger.info(
                "Starting GraphCast forecast: %s -> +%d h (%d steps)",
                init_iso,
                args.forecast_hours,
                nsteps,
            )
            append_init_log(
                init_log,
                f"START forecast {init_iso}; "
                f"+{args.forecast_hours} h; {nsteps} steps",
            )

            # Earth2Studio workflow signatures differ across releases.
            # Build keyword arguments from the installed deterministic()
            # signature so this runner works with older versions such as
            # Earth2Studio 0.11.0 as well as newer releases.
            deterministic_kwargs = {
                "time": [init_iso],
                "nsteps": nsteps,
                "prognostic": model,
                "data": data,
                "io": io,
                "device": device,
            }

            deterministic_parameters = inspect.signature(
                deterministic
            ).parameters

            if "verbose" in deterministic_parameters:
                deterministic_kwargs["verbose"] = True

            logger.info(
                "Earth2Studio deterministic() parameters: %s",
                ", ".join(deterministic_parameters.keys()),
            )

            deterministic(**deterministic_kwargs)

            del io

            # GraphCast computation is JAX/XLA-based. We intentionally do
            # not report PyTorch allocator memory as GraphCast VRAM.
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            validate_netcdf(partial_output)
            partial_output.replace(final_output)

            elapsed = time.perf_counter() - cycle_start
            output_size = final_output.stat().st_size

            cache_after_files, cache_after_bytes = cache_stats(era5_cache)
            new_files = cache_after_files - cache_before_files
            new_bytes = cache_after_bytes - cache_before_bytes

            logger.info("STATUS: COMPLETE")
            logger.info(
                "Cycle runtime         : %.1f s (%.2f min)",
                elapsed,
                elapsed / 60.0,
            )
            logger.info("NetCDF size           : %s", human_bytes(output_size))
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
            append_init_log(init_log, f"NetCDF: {final_output}")
            append_init_log(
                init_log,
                f"NetCDF size: {human_bytes(output_size)}",
            )
            append_init_log(
                init_log,
                f"ERA5 cache after cycle: {cache_after_files} files, "
                f"{human_bytes(cache_after_bytes)}",
            )
            append_init_log(
                init_log,
                f"ERA5 cache change: {new_files:+d} files, "
                f"{human_bytes(max(new_bytes, 0))} added",
            )

            n_complete += 1

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
                    "completed valid NetCDF files will be skipped."
                )
                return 1

            logger.warning(
                "--continue-on-error is set; moving to next initialization."
            )

    # -----------------------------------------------------------------
    # Final batch summary
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
    logger.info("Batch runtime    : %.2f h", elapsed_batch / 3600.0)
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
    # GraphCast/JAX can occasionally abort during Python interpreter teardown
    # after a completely successful forecast (for example:
    # "free(): invalid pointer").
    #
    # main() only returns 0 after inference completes, the NetCDF is validated
    # and promoted to its final filename, and the batch summary is written.
    # Flush Python-managed output and bypass third-party native destructors.
    # Genuine forecast failures still return a non-zero code from main().
    exit_code = main()

    logging.shutdown()

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(exit_code)
