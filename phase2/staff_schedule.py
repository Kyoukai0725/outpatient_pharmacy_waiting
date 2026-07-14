"""Parse daily on-duty staff from 2024/2025 schedules.

Window count is fixed at 6; schedules provide daily dispensing/dispensing-window/total on-duty counts.
Shift codes: OF=off, BE/BQ/CD/BM etc.=on duty.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from phase2.config import PROJECT_ROOT

N_WINDOWS = 6

# Schedules live under raw/schedules (at repo root = PROJECT_ROOT)
SCHEDULE_FILES = [
    PROJECT_ROOT / "raw" / "schedules" / "2024年班表-V2.0.xlsx",
    PROJECT_ROOT / "raw" / "schedules" / "2025年班表(1).xlsx",
]


def is_off_code(code) -> bool:
    if pd.isna(code):
        return True
    raw = str(code).strip()
    if not raw:
        return True
    s = raw.upper()
    if s.startswith("OF"):
        return True
    if "假" in raw or "休" in raw:
        return True
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return True
    if s in {"NAN", "NONE", "岗位", "时数"}:
        return True
    return False


def _parse_year_month(sheet: str) -> tuple[int, int] | None:
    m = re.search(r"(20\d{2})[.\-]?(\d{1,2})", sheet)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def parse_month_sheet(path: Path, sheet: str) -> list[dict]:
    ym = _parse_year_month(sheet)
    if ym is None:
        return []
    year, month = ym

    df = pd.read_excel(path, sheet_name=sheet, header=None)
    day_row = None
    for r in range(min(6, df.shape[0])):
        nums = [
            v
            for v in df.iloc[r]
            if isinstance(v, (int, float, np.integer, np.floating))
            and not pd.isna(v)
            and 1 <= float(v) <= 31
        ]
        if len(nums) >= 20:
            day_row = r
            break
    if day_row is None:
        return []

    day_cols = []
    for c in range(df.shape[1]):
        v = df.iat[day_row, c]
        if (
            isinstance(v, (int, float, np.integer, np.floating))
            and not pd.isna(v)
            and 1 <= float(v) <= 31
        ):
            day_cols.append((c, int(v)))

    role_col = 3
    start = day_row + 2
    seen_one = False
    records = []

    for c, day in day_cols:
        if day == 1:
            seen_one = True
        if not seen_one and day >= 21:
            y, mth = (year - 1, 12) if month == 1 else (year, month - 1)
        else:
            y, mth = year, month
        try:
            d = date(y, mth, day)
        except ValueError:
            continue

        on_duty = dispense = window = 0
        for r in range(start, df.shape[0]):
            role = df.iat[r, role_col]
            name = df.iat[r, 2]
            code = df.iat[r, c]
            if pd.isna(name) and pd.isna(role):
                continue
            if is_off_code(code):
                continue
            role_s = "" if pd.isna(role) else str(role).strip()
            if role_s in {"岗位", "时数"} or role_s.startswith("合计"):
                continue
            on_duty += 1
            if role_s.startswith("调"):
                dispense += 1
            elif role_s.startswith("发"):
                window += 1

        records.append(
            {
                "日期": d,
                "在岗人数": on_duty,
                "调配人数": dispense,
                "发药人数": window,
                "窗口数": N_WINDOWS,
                "有效窗口": min(N_WINDOWS, window) if window > 0 else N_WINDOWS,
            }
        )
    return records


def load_daily_staff(files: list[Path] | None = None) -> pd.DataFrame:
    files = files or SCHEDULE_FILES
    rows: list[dict] = []
    for path in files:
        if not path.exists():
            continue
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            if re.match(r"^20\d{2}(\d{2}|\.\d{1,2})$", sheet):
                rows.extend(parse_month_sheet(path, sheet))
    staff = pd.DataFrame(rows)
    if staff.empty:
        raise FileNotFoundError("Could not parse any daily staff from schedules")
    staff = staff.drop_duplicates("日期", keep="last").sort_values("日期")
    # Capacity proxy: fixed windows + dispensing staff
    staff["产能代理"] = staff["有效窗口"] + 0.5 * staff["调配人数"]
    return staff.reset_index(drop=True)


def save_daily_staff(staff: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staff.to_parquet(path, index=False)
