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
  --dataset-id XXXXXX \
  --manual-df-path /data/manual-XXXXXX-displacement-field \
  --output-root /results/ccf_space_reconstructions \
  --metadata-dir /results/metadata \
  --node-spacing-um 10
```

### Transform resolution strategy
- Shared assets are fixed by default (override only if needed):
  - `--exaspim-to-ccf-affine-path` default `/data/reg_exaspim_template_to_ccf_withGradMap_10um_v1.0/0GenericAffine.mat`
  - `--exaspim-to-ccf-inverse-warp-path` default `/data/reg_exaspim_template_to_ccf_withGradMap_10um_v1.0/1InverseWarp.nii.gz`
  - `--ccf-template-path` default `/data/allen_mouse_ccf/average_template/average_template_10.nii.gz`
  - `--exaspim-template-path` default `/data/exaspim_template_7subjects_nomask_10um_round6_template_only/fixed_median.nii.gz`
- Per-sample assets are resolved by strict naming convention under `--transform-dir`:
  - `{dataset_id}_to_exaSPIM_SyN_0GenericAffine.mat`
  - `{dataset_id}_to_exaSPIM_SyN_1InverseWarp.nii.gz`
  - `registration_metadata/acquisition_{dataset_id}.json`
  - `registration_metadata/{dataset_id}_10um_loaded_zarr_img.nii.gz`
  - `registration_metadata/{dataset_id}_10um_resampled_zarr_img.nii.gz`
- Per-sample assets can all be overridden explicitly:
  - `--acquisition-file-path`
  - `--loaded-zarr-image-path`
  - `--resampled-zarr-image-path`
  - `--sample-to-exaspim-affine-path`
  - `--sample-to-exaspim-inverse-warp-path`
- Manual displacement field:
  - if `--manual-df-path` is a file, use it directly
  - if it is a directory, the capsule searches dataset-specific names; use `--manual-df-filename` to force a specific file
  - omit `--manual-df-path` to skip manual DF

## Naming style
- `--naming-style preserve` keeps original stem.
- `--naming-style suffix` appends `__space-ccf_res-10um`.

## Code Ocean wrapper
- Primary entrypoint: `python run.py`
- Convenience wrapper: `./run <transform_dir> [manual_df_path]`
