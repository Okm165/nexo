"""Per-service environment loading. Precedence: OS env > .env.<service> > .env."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_env(service: str = "train") -> dict[str, str]:
    """Merge .env and every .env.* file; OS environment always wins.

    The `service` name is accepted for call-site clarity; all service files
    are merged (they hold disjoint keys), so a stage never misses a path it
    needs. Unknown keys in the environment are exposed too, which is what
    makes `EMB_LIMIT=200 cmd` actually work.
    """
    cfg: dict[str, str] = {}
    paths = sorted(p for p in ROOT.glob(".env.*") if p.name != ".env.example")
    for path in [ROOT / ".env", *paths]:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    cfg.update({k: v for k, v in os.environ.items()})
    return cfg
