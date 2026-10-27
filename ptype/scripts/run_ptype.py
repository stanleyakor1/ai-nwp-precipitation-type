#!/usr/bin/env python

import argparse
from pathlib import Path
import xarray as xr

from ainwp_ptype_original import AINWPPreprocessor, PipelineConfig, process_initialization
from ainwp_ptype_original.ednn import load_saved_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--surface-geopotential", type=Path, required=True)
    parser.add_argument("--ednn-model", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-uncertainties", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = PipelineConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        surface_geopotential_path=args.surface_geopotential,
        output_uncertainties=not args.no_uncertainties,
        overwrite=args.overwrite,
    )
    model, scaler, features = load_saved_model(args.ednn_model, args.scaler)
    with xr.open_dataset(args.input, decode_timedelta=False) as source:
        processed = AINWPPreprocessor().process(source)
        outputs = process_initialization(processed, model, scaler, features, config)
    print(f"Saved or validated {len(outputs)} original-numerics leads in {args.output_dir}")


if __name__ == "__main__":
    main()
