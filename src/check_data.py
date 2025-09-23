#!/usr/bin/env python3
import os
from pathlib import Path

DATA_DIR = Path("data/processed")

print("=" * 50)
print("KIỂM TRA DỮ LIỆU")
print("=" * 50)

if DATA_DIR.exists():
    files = list(DATA_DIR.glob("*.csv"))
    print(f"Tìm thấy {len(files)} files CSV:")
    for f in files:
        print(f"  ✅ {f.name}")
else:
    print("❌ Thư mục data/processed không tồn tại!")

