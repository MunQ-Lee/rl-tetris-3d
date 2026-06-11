"""Push the best checkpoint to GitHub when training performance improves.

Run periodically (e.g. every 10 minutes). It reads checkpoints/train_log.jsonl,
computes the best performance reached so far, and if it beat the last pushed
performance it force-adds the agents' checkpoints + the training log and pushes.

Performance metric (matches train.py):
    perf = mean_lines * 100 + mean_score + mean_return
We also push whenever a new high in cleared layers or game score appears.
"""
import json
import os
import subprocess
import sys

CKPT_DIR = "checkpoints"
LOG = os.path.join(CKPT_DIR, "train_log.jsonl")
# Competitive run keeps two agents' checkpoints; push whichever exist.
CKPTS = [os.path.join(CKPT_DIR, f) for f in
         ("best_p0.pt", "best_p1.pt", "latest_p0.pt", "latest_p1.pt")]
STATE = os.path.join(CKPT_DIR, ".push_state.json")
MARGIN = 0.2  # minimum perf gain required to push (noise guard; greedy-lines scale)


def sh(args):
    return subprocess.run(args, cwd=os.path.dirname(os.path.abspath(__file__)),
                          capture_output=True, text=True)


def load_rows():
    if not os.path.exists(LOG):
        return []
    rows = []
    for line in open(LOG):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def perf_of(r):
    return r.get("mean_lines", 0) * 100.0 + r.get("mean_score", 0) + r.get("mean_return", 0)


def main():
    rows = load_rows()
    existing = [c for c in CKPTS if os.path.exists(c)]
    if not rows or not existing:
        print("nothing to push yet (no log or checkpoint)")
        return 0

    best_row = max(rows, key=perf_of)
    cur = {
        "perf": perf_of(best_row),
        "lines": max(r.get("mean_lines", 0) for r in rows),
        "score": max(r.get("mean_score", 0) for r in rows),
        "update": rows[-1].get("update", 0),
        "step": rows[-1].get("step", 0),
    }

    prev = {"perf": -1e18, "lines": -1, "score": -1}
    if os.path.exists(STATE):
        try:
            prev = json.load(open(STATE))
        except Exception:  # noqa: BLE001
            pass

    improved = (cur["perf"] > prev["perf"] + MARGIN
                or cur["lines"] > prev.get("lines", -1)
                or cur["score"] > prev.get("score", -1))
    if not improved:
        print(f"no improvement (perf {cur['perf']:.1f} <= last {prev['perf']:.1f}); skip push")
        return 0

    # Stage best checkpoint + training log (both are normally gitignored).
    sh(["git", "add", "-f", *existing, LOG])
    msg = (f"Training improved: perf {cur['perf']:.1f}, "
           f"lines {cur['lines']:.2f}, score {cur['score']:.1f} "
           f"@ update {cur['update']} (step {cur['step']})")
    commit = sh(["git", "-c", "user.email=causslab@gmail.com",
                 "-c", "user.name=MunQ-Lee", "commit", "-m",
                 msg + "\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"])
    if commit.returncode != 0 and "nothing to commit" in (commit.stdout + commit.stderr):
        print("checkpoint unchanged since last commit; skip push")
        return 0
    push = sh(["git", "push", "origin", "HEAD"])
    if push.returncode != 0:
        print("PUSH FAILED:\n" + push.stderr)
        return 1

    json.dump(cur, open(STATE, "w"))
    print("PUSHED ->", msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
