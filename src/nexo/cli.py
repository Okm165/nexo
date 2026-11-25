"""nexo — forecast next-window realized volatility from headlines + prices."""
import argparse

from nexo.env import load_env


def main() -> None:
    p = argparse.ArgumentParser(prog="nexo", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch-bars", help="optional: Alpaca daily OHLC for SYMBOLS -> BARS_PATH")
    sub.add_parser("fetch-news", help="optional: Alpaca headlines for SYMBOLS -> NEWS_PATH")
    e = sub.add_parser("embed", help="headlines -> frozen embeddings (one shard)")
    e.add_argument("--shard", type=int, default=0)
    e.add_argument("--shards", type=int, default=1)
    e.add_argument("--limit", type=int, default=0, help="first N headlines only (smoke test)")
    e.add_argument("--device", default="cuda")
    sub.add_parser("build", help="sliding-window samples, inputs and targets")
    sub.add_parser("train", help="fit every arm on every walk-forward fold")
    sub.add_parser("forecast", help="predict the next window from the latest close")
    sub.add_parser("evaluate", help="score every fold's test period, with DM tests")
    a = p.parse_args()
    cfg = load_env()

    if a.cmd == "fetch-bars":
        from nexo.fetch import bars
        bars(cfg)
    elif a.cmd == "fetch-news":
        from nexo.fetch import news
        news(cfg)
    elif a.cmd == "embed":
        from nexo.text import embed
        embed(cfg, a.shard, a.shards, a.limit, a.device)
    elif a.cmd == "build":
        from nexo.dataset import build
        build(cfg)
    elif a.cmd == "train":
        from nexo.train import run
        run(cfg)
    elif a.cmd == "forecast":
        from nexo.train import forecast
        forecast(cfg)
    else:
        from nexo.evaluate import main as evaluate
        evaluate(cfg)


if __name__ == "__main__":
    main()
