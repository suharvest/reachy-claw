"""Production launcher for the CLIENT-LOOP Reachy app.

``ovs-agent run <name>`` resolves ``ovs_agent.apps.<name>.app:App`` and
cannot find ``reachy_claw.clientloop`` under its own ``apps/`` tree. This
launcher mirrors ``ovs_agent.cli.main`` but loads
``ReachyClawClientLoopApp`` directly from an explicit ``--config`` path
(defaulting to the baked ``clientloop/config.yaml``), then drives the
full ``app.run()`` event loop — the same path ``voice_arm`` is run on.

Installed as the ``reachy-claw-clientloop`` console script.

Run:
  reachy-claw-clientloop                 # uses baked config.yaml
  reachy-claw-clientloop --config /x.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from ovs_agent.config import Config, load_config

from reachy_claw.clientloop.app import ReachyClawClientLoopApp

_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reachy-claw-clientloop")
    ap.add_argument(
        "--config",
        type=Path,
        default=_CONFIG,
        help="path to YAML config (default: baked clientloop/config.yaml)",
    )
    args = ap.parse_args(argv)

    cfg_path: Path = args.config
    if cfg_path.exists():
        cfg = load_config(cfg_path)
    else:
        print(f"config not found: {cfg_path}; using defaults", file=sys.stderr)
        cfg = Config()

    logging.basicConfig(
        level=getattr(logging, str(cfg.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("reachy-claw-clientloop")
    log.info(
        "launching ReachyClawClientLoopApp: llm_backend=%s model=%s "
        "base_url=%s slv_url=%s tools_enabled=%s",
        cfg.llm_backend, cfg.llm_model, cfg.llm_base_url,
        cfg.slv_url, cfg.tools_enabled,
    )

    app = ReachyClawClientLoopApp(cfg)
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
