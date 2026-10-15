"""把 family_search 的日志解析成一张裁决表(避免手工誊数字出错)。

用法:
    python tools/parse_family_logs.py logs/family_run3.log logs/family_run4.log

只读日志、不重跑;数字全部来自日志里的原始输出。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

FAM = re.compile(r"^族:(\S+)\s")
POOL = re.compile(r"测试窗\(样本外统计量\)\s+CAGR\s+([+-][\d.]+)%\s+中位\s+([+-][\d.]+)%"
                  r"\s+基准\s+([+-][\d.]+)%\s+超额\s+([+-][\d.]+)pp")
CONT = re.compile(r"连续净值 CAGR\s+([+-][\d.]+)%\s+基准\s+([+-][\d.]+)%\s+"
                  r"超额\s+([+-][\d.]+)pp\s+平均敞口\s+([\d.]+)%")
FOLDS = re.compile(r"逐折测试窗 CAGR:(.*)")
NULL90 = re.compile(r"随机对照:≥中位\s+(\d+)/(\d+)\s+窗\s+≥p90\s+(\d+)/(\d+)\s+窗")
DSR = re.compile(r"DSR\(Sharpe\s+([+-][\d.]+),\s+k=(\d+)\)\s+=\s+([\d.]+)")
VERD = re.compile(r"→\s+(.+?)\s+DSR")
BASE = re.compile(r"^\[基准\]|测试窗\s+CAGR\s+([+-][\d.]+)%")


def parse(paths: list[str]) -> dict:
    rows: dict[str, dict] = {}
    cur = None
    baseline = None
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            m = FAM.match(line.strip())
            if m:
                cur = m.group(1)
                rows.setdefault(cur, {})
                continue
            if "测试窗" in line and "CAGR" in line and (baseline is None or cur is None):
                mm = POOL.search(line.strip())
                if mm and baseline is None:
                    baseline = dict(cagr=float(mm.group(1)), bench=float(mm.group(3)),
                                    excess=float(mm.group(4)))
                continue
            if cur is None:
                continue
            for pat, keys in ((POOL, ("cagr", "median", "bench", "excess")),
                              (CONT, ("cont", "cont_bench", "cont_excess", "expo")),
                              (NULL90, ("w50", "n50", "w90", "n90")),
                              (DSR, ("sharpe", "k", "dsr"))):
                mm = pat.search(line.strip())
                if mm:
                    for k, v in zip(keys, mm.groups()):
                        rows[cur][k] = float(v) if "." in v or k in ("sharpe", "dsr") else int(v)
            mm = FOLDS.search(line)
            if mm:
                rows[cur]["folds"] = [float(x) for x in re.findall(r"([+-][\d.]+)%", mm.group(1))]
            mm = VERD.search(line)
            if mm:
                rows[cur]["verdict"] = mm.group(1).strip()
    return {"baseline": baseline, "families": rows}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 0
    d = parse(argv)
    b = d["baseline"]
    print("=" * 118)
    if b:
        print(f"基准线 uptrend-pullback: 池化 CAGR {b['cagr']:+.2f}%  基准 {b['bench']:+.2f}%  "
              f"超额 {b['excess']:+.2f}pp")
    print("=" * 118)
    print(f"{'族':<12}{'池化CAGR':>9}{'池化超额':>10}{'连续CAGR':>9}{'连续超额':>10}"
          f"{'敞口':>6}{'折正':>5}{'≥p50':>6}{'≥p90':>6}{'DSR':>7}{'CAGR/敞口':>9}  裁决")
    print("-" * 118)
    for fid, r in d["families"].items():
        if "cagr" not in r:
            print(f"{fid:<12}{'(无有效窗口)':>28}")
            continue
        folds = r.get("folds", [])
        pos = sum(1 for f in folds if f > 0)
        print(f"{fid:<12}{r['cagr']:>+8.2f}%{r.get('excess', float('nan')):>+9.2f}pp"
              f"{r.get('cont', float('nan')):>+8.2f}%{r.get('cont_excess', float('nan')):>+9.2f}pp"
              f"{r.get('expo', float('nan')):>5.1f}%{pos:>3}/{len(folds)}"
              f"{int(r.get('w50', 0)):>4}/{int(r.get('n50', 0))}"
              f"{int(r.get('w90', 0)):>4}/{int(r.get('n90', 0))}"
              f"{r.get('dsr', float('nan')):>7.3f}"
              f"{r.get('cagr', float('nan'))/max(r.get('expo', float('nan')), 0.1):>8.2f}"
              f"  {r.get('verdict', '?')}")
    print("-" * 118)
    print("折正 = 4 个测试窗里 CAGR 为正的窗数;≥p50/≥p90 = 打赢随机对照的窗数")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
