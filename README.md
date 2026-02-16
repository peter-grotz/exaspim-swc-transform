# exaspim-swc-transform

Script-based replacement for notebook-driven ExaSPIM SWC transform.

## What it does
- Reads refined SWCs from `/data/final-world` (or `--swc-dir`).
- Resolves registration assets from a registration bundle (`--transform-dir`).
- Applies sample -> exaSPIM -> CCF transforms via `aind_exaspim_register_cells.RegistrationPipeline`.
- Writes transformed CCF-space SWCs to `/results/aligned_swcs`.
- Writes run metadata and manifests:
  - `/results/data_process.json`
  - `/results/process_report.json`
  - `/results/manifests/inputs_manifest.json`
  - `/results/manifests/outputs_manifest.json`

## CLI
```bash
python -m exaspim_swc_transform.cli \
  --swc-dir /data/final-world \
  --transform-dir /data/reg_XXXXXX_to_ccf_v1.5 \
  --manual-df-path /data/manual-displacement/field.nrrd
```

## Naming style
- `--naming-style preserve` keeps original stem.
- `--naming-style suffix` appends `__space-ccf_res-10um`.
- `--parity-mode` enforces notebook-era defaults:
  - output dir: `/results/aligned`

## Code Ocean run wrapper
- Primary entrypoint: `python run.py`
- Convenience wrapper: `./run <transform_dir> [manual_df_path]`
- Parity wrapper usage: `PARITY_MODE=1 ./run <transform_dir> [manual_df_path]`
