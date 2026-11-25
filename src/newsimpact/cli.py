"""newsimpact — forecast next-window realized volatility from headlines + prices."""
import argparse

from newsimpact.env import load_env


def main() -> None:
    p = argparse.ArgumentParser(prog="newsimpact", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed", help="headlines -> frozen embeddings (one shard)")
    e.add_argument("--shard", type=int, default=0)
    e.add_argument("--shards", type=int, default=1)
    e.add_argument("--limit", type=int, default=0, help="first N headlines only (smoke test)")
    e.add_argument("--device", default="cuda")
    sub.add_parser("build", help="sliding-window samples, inputs and targets")
    t = sub.add_parser("train", help="train one arm/seed")
    t.add_argument("--arm", default="text", choices=("text", "shuffled", "price"))
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    sub.add_parser("evaluate", help="score all runs on test vs persistence and HAR")
    sub.add_parser("fetch-news", help="optional: Alpaca news archive -> NEWS_PATH")
    sub.add_parser("fetch-bars", help="optional: Alpaca daily bars -> BARS_PATH")
    a = p.parse_args()
    cfg = load_env()

    if a.cmd == "embed":
        from newsimpact.text import embed
        embed(cfg, a.shard, a.shards, a.limit, a.device)
    elif a.cmd == "build":
        from newsimpact.dataset import build
        build(cfg)
    elif a.cmd == "train":
        from newsimpact.train import run
        run(a.arm, a.seed, cfg, a.device)
    elif a.cmd == "evaluate":
        from newsimpact.evaluate import main as evaluate
        evaluate(cfg)
    elif a.cmd == "fetch-news":
        from newsimpact.fetch_news import main as fetch_news
        fetch_news()
    else:
        from newsimpact.fetch_bars import main as fetch_bars
        fetch_bars()


if __name__ == "__main__":
    main()
