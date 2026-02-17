# exaspim-swc-transform

Transform-only capsule stage for moving SWC coordinates into CCF space.

Outputs:
- `/results/aligned_swcs/*.swc`
- `/results/metadata/processing.json`

Run:
```bash
python run.py --swc-dir /data/final-world --transform-dir /data/reg_XXXXXX_to_ccf
```
