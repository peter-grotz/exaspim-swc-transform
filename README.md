# exaspim-swc-transform

Unified post-refinement capsule:
1) transform SWCs into CCF space,
2) resample to 10um,
3) assign structure types,
4) write final SWC and annotated JSON outputs,
5) emit processing metadata.

## Final output layout
- `/results/metadata/`
  - `processing.json` (AIND Processing schema)
  - `data_process.json` (compatibility alias)
  - `process_report.json`
  - `runtime_args.json`
  - `manifests/inputs_manifest.json`
  - `manifests/outputs_manifest.json`
- `/results/ccf_space_reconstructions/`
  - `swcs/*.swc`
  - `jsons/*.json`

## CLI
```bash
python run.py \
  --swc-dir /data/final-world \
  --transform-dir /data/reg_XXXXXX_to_ccf_v1.5 \
  --manual-df-path /data/manual-displacement/field.nrrd \
  --output-root /results/ccf_space_reconstructions \
  --metadata-dir /results/metadata \
  --node-spacing-um 10
```

## Naming style
- `--naming-style preserve` keeps original stem.
- `--naming-style suffix` appends `__space-ccf_res-10um`.

## Code Ocean wrapper
- Primary entrypoint: `python run.py`
- Convenience wrapper: `./run <transform_dir> [manual_df_path]`
