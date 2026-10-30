# Path rename map

Date and time tokens were removed from file and directory names for publication.
No file contents, recorded timestamps, run_id values, or numbers were changed
by this rename.

## Rename table

| Old name | New name | Note |
|---|---|---|
| `pilot_outputs/run_20261020_163459/` | `pilot_outputs/run/` | Date `20261020` and time `163459` removed |
| `study_outputs/approved_comparison_20261020/` | `study_outputs/approved_comparison/` | Date `20261020` removed |

Files inside these directories were not renamed; they inherited the new parent
path automatically.

## Old identifiers remaining inside frozen content

The following frozen raw evidence files still contain the original date-bearing
identifiers in their content. This is intentional — frozen files are not edited
for cosmetic consistency. The directory rename is sufficient.

| File (published path) | Line | Token | Field/context |
|---|---|---|---|
| `study_outputs/approved_comparison/study_summary.json` | 2 | `approved_comparison_20261020` | `run_id` value |
| `pilot_outputs/run/summary.json` | 2 | `run_20261020_163459` | `run_id` value |
| `pilot_outputs/run/pilot_report.md` | 3 | `run_20261020_163459` | Run id heading |
| `study_outputs/approved_comparison/study_summary_corrected.json` | 4 | `approved_comparison_20261020` | `run_id` value (recorded, not a path) |

## Old identifiers remaining inside fingerprinted source code

`runner.py` is a fingerprinted source (recorded SHA-256 `75b4029268c2b9d71718e582b4d4c57979698773e927ef028612c25bbcd2e111`
in the original single-seed study `config.json` at initial commit `aca0c07`) and
must not be edited. It contains:

| File | Line | Token | Context |
|---|---|---|---|
| `runner.py` | 31 | `approved_comparison_20261020` | `STUDY_NAME` constant (generates output directory names at runtime) |

## Old identifiers remaining in `recompute_study.py` (non-frozen, retained as run_id fallback)

| File | Line | Token | Context |
|---|---|---|---|
| `recompute_study.py` | 291 | `approved_comparison_20261020` | Fallback default for `run_id` field read from `study_summary.json` |
