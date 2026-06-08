"""下載 MESA(NSRR)的一個子集:EDF 訊號 + NSRR 事件標註。

只抓兩個 subfolder:
  polysomnography/edfs/                 (含 ECG 的整夜 PSG)
  polysomnography/annotations-events-nsrr/  (apnea/hypopnea 事件 XML)

token 從 --token 或環境變數 NSRR_TOKEN 取(不寫死在檔案裡)。

範例:
  python download_mesa.py --pattern "mesa-sleep-00[0-3]*" --data-dir D:/mesa --token 12345-XXXX
  python download_mesa.py --pattern "mesa-sleep-0[01]*"   --data-dir D:/mesa   # ~200 人
"""
import argparse
import os
from pathlib import Path

from sleepecg import download_nsrr, set_nsrr_token


def count(root: Path, glob: str) -> int:
    return len(list(root.rglob(glob)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="mesa-sleep-00[0-3]*",
                   help="fnmatch 樣式,套用到檔名。預設 ~40 人 (0001-0039)。")
    p.add_argument("--data-dir", default="D:/mesa")
    p.add_argument("--token", default=os.environ.get("NSRR_TOKEN"))
    args = p.parse_args()

    if not args.token:
        raise SystemExit("缺少 token:用 --token 或設環境變數 NSRR_TOKEN。")
    set_nsrr_token(args.token)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] pattern={args.pattern!r}  data_dir={data_dir}")
    print("[download] EDF 訊號 ...", flush=True)
    download_nsrr(db_slug="mesa", subfolder="polysomnography/edfs",
                  pattern=args.pattern, shallow=True, data_dir=str(data_dir))
    print("[download] NSRR 事件標註 ...", flush=True)
    download_nsrr(db_slug="mesa", subfolder="polysomnography/annotations-events-nsrr",
                  pattern=args.pattern, shallow=True, data_dir=str(data_dir))

    n_edf = count(data_dir, "mesa-sleep-*.edf")
    n_xml = count(data_dir, "mesa-sleep-*-nsrr.xml")
    print(f"\n[download] 完成:edf={n_edf}  xml={n_xml}  (under {data_dir})")
    if n_edf == 0:
        print("[download] 警告:沒抓到任何 EDF — 檢查 token / DAUA 是否核准 / pattern。")


if __name__ == "__main__":
    main()
