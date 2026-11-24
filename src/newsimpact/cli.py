"""newsimpact — forecast next-window realized volatility from headlines + prices."""
import argparse

from newsimpact.env import load_env


def main() -> None:
    p = argparse.ArgumentParser(prog="newsimpact", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch-bars", help="optional: Alpaca daily OHLC for SYMBOLS -> BARS_PATH")
    sub.add_parser("fetch-news", help="optional: Alpaca headlines for SYMBOLS -> NEWS_PATH")
    e = sub.add_parser("embed", help="headlines -> frozen embeddings (one shard)")
    e.add_argument("--shard", type=int, default=0)
    e.add_argument("--shards", type=int, default=1)
    e.add_argument("--limit", type=int, default=0, help="first N headlines only (smoke test)")
    e.add_argument("--device", default="cuda")
    sub.add_parser("build", help="sliding-window samples, inputs and targets")
    sub.add_parser("train", help="fit every arm on every random split")
    sub.add_parser("holdout", help="one time-ordered split: unread test period")
    sub.add_parser("evaluate", help="score all splits vs HAR and persistence")
    a = p.parse_args()
    cfg = load_env()

    if a.cmd == "fetch-bars":
        from newsimpact.fetch import bars
        bars(cfg)
    elif a.cmd == "fetch-news":
        from newsimpact.fetch import news
        news(cfg)
    elif a.cmd == "embed":
        from newsimpact.text import embed
        embed(cfg, a.shard, a.shards, a.limit, a.device)
    elif a.cmd == "build":
        from newsimpact.dataset import build
        build(cfg)
    elif a.cmd == "train":
        from newsimpact.train import run
        run(cfg)
    elif a.cmd == "holdout":
        from newsimpact.train import run_holdout
        run_holdout(cfg)
    else:
        from newsimpact.evaluate import main as evaluate
        evaluate(cfg)


if __name__ == "__main__":
    main()
