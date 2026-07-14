from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DISPENSE_FILE = PROJECT_ROOT / "raw" / "dispense" / "门诊药房药品调配时间统计-20251226.xlsx"
LAYOUT_FILE = PROJECT_ROOT / "raw" / "layout" / "方位图.xlsx"

WAIT_DATA_DIRS = {
    2024: PROJECT_ROOT / "raw" / "wait_times" / "2024",
    2025: PROJECT_ROOT / "raw" / "wait_times" / "2025",
}

WAIT_COLS = [
    "处方编号",
    "药品编码",
    "药品名称及规格",
    "报到时间",
    "呼叫时间",
    "摆药完成时间",
    "发药E化时间",
]

MACHINE_DISPENSE_SEC = 17.0
