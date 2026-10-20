# Sanitization Notice

This repository contains path-sanitized versions of environment metadata files.
To prevent leaking local workstation home paths, absolute user home path prefixes in recorded environment files have been replaced with `<project>`.

These published files are path-sanitized and are not byte-identical to the original raw evidence files (which are preserved locally in the pre-rename backup). All recorded numbers, metrics, timestamps, and parameters in these files remain completely unchanged.

## Sanitized Files

| File | Line / Field | Original SHA-256 | Published SHA-256 | Change |
|---|---|---|---|---|
| `pilot_outputs/run/environment.json` | Line 3, `"executable"` | `6f2af0b004c48a5b0ac3cf6ebbe8383162153e495f9808afdfb52e52fab19468` | `725bf865a274410b3369efe1015a8b348d52b0cbf4d05315f022822916d8bb0d` | Replaced absolute user home path prefix with `<project>` |
| `study_outputs/approved_comparison/environment.json` | Line 3, `"executable"` | `786327cf85bab07b1a33bcb7ea5fa0fa2ae624c6de751b2c1f55e17b214cce51` | `646d5fbd3d06079f552cd88a355541909cf49bb4448afe35400ca9e2a7b263b1` | Replaced absolute user home path prefix with `<project>` |

No fingerprinted sources (`runner.py`, `model.py`, `data.py`, `grokfast.py`) contained absolute home paths.
