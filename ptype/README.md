
```bash
python -m pip install --no-deps -e .

python scripts/run_ptype.py \
  --input /path/to/forecast.nc \
  --model-name graphcast \
  --surface-geopotential /path/to/surface_geopotential.nc \
  --ednn-model /path/to/model.keras \
  --scaler /path/to/scaler.json \
  --output-dir /path/to/original_numerics_output
```
