# Raw data layout (not committed)

Place hospital extracts here. Filenames may be Chinese; column names used by the
code are listed in `preprocess/config.py`.

```
raw/
├── wait_times/
│   ├── 2024/          # 侯药-2024MM.xls
│   └── 2025/
├── dispense/          # outpatient dispense timing workbook (.xlsx)
├── layout/            # shelf location map (.xlsx)
├── schedules/         # daily staff rosters
└── machine_changes/   # optional historical machine in/out log
```

Update absolute-relative filenames in `preprocess/config.py` if your exports
use different names. Never commit identifiable patient-level extracts.
