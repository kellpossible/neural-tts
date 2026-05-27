#!/usr/bin/env python3
"""Install systemd units + speechd module config to ~/.config (user scope).

Idempotent. Run after `mise run sync-daemon` and at least one
`mise run install-provider <name>`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    config_dir,
    confirm,
    daemon_python,
    die,
    info,
    load_registry,
    render_template,
    repo_root,
    require,
    speechd_config_dir,
    speechd_modules_dir,
    speechd_system_conf,
    speechd_user_conf,
    systemd_user_dir,
    warn,
)

def _module_binary() -> str:
    return str(repo_root() / "bin" / "sd-neural-tts")


def _add_module_line() -> str:
    return f'AddModule "neural-tts" "{_module_binary()}" "neural-tts.conf"'


# Legacy line we may need to remove from existing user configs that were
# installed before we switched to the native module.
_LEGACY_ADD_MODULE_LINE = (
    'AddModule "neural-tts-generic" "sd_generic" "neural-tts-generic.conf"'
)


def preflight() -> None:
    info("preflight checks")
    require("paplay", "Fedora: dnf install pulseaudio-utils  (or pipewire-pulse)")
    require("systemctl")
    require("uv", "https://docs.astral.sh/uv/")
    if not shutil.which("espeak-ng"):
        warn("espeak-ng not found; kokoro-onnx uses misaki G2P which prefers it. "
             "Fedora: dnf install espeak-ng")
    if not daemon_python().exists():
        die("daemon venv not found; run `mise run sync-daemon` first")
    registry = load_registry()
    installed = [n for n, m in registry.items() if (repo_root() / m["project_dir"] / ".venv" / "bin" / "python").exists()]
    if not installed:
        warn("no provider installed yet — run `mise run install-provider kokoro-onnx`. "
             "Install will continue, but the daemon won't synthesize until a provider is installed.")


def install_unit_files() -> None:
    info("installing systemd unit files")
    unit_dir = systemd_user_dir()
    unit_dir.mkdir(parents=True, exist_ok=True)

    src = repo_root() / "share" / "systemd"
    # Copy both socket units verbatim.
    for name in ("neural-tts.socket", "neural-tts-control.socket"):
        dst = unit_dir / name
        dst.write_text((src / name).read_text())
        info(f"  wrote {dst}")
    # Render the service template.
    svc_tpl = (src / "neural-tts.service.in").read_text()
    svc = render_template(svc_tpl, REPO=str(repo_root()), DAEMON_PY=str(daemon_python()))
    dst = unit_dir / "neural-tts.service"
    dst.write_text(svc)
    info(f"  wrote {dst}")


def seed_daemon_config() -> None:
    cfg = config_dir() / "config.toml"
    if cfg.exists():
        info(f"daemon config already at {cfg} (leaving as-is)")
        return
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        "[provider]\n"
        'default = "kokoro-onnx"\n\n'
        "[supervisor]\n"
        "idle_timeout_seconds = 600\n"
        "eager_startup = false\n"
    )
    info(f"  wrote {cfg}")


def install_speechd_module_conf() -> None:
    info("installing speechd module config")
    speechd_modules_dir().mkdir(parents=True, exist_ok=True)
    src = repo_root() / "share" / "speech-dispatcher" / "neural-tts.conf"
    dst = speechd_modules_dir() / "neural-tts.conf"
    dst.write_text(src.read_text())
    info(f"  wrote {dst}")
    # Remove the old sd_generic module conf if present
    legacy = speechd_modules_dir() / "neural-tts-generic.conf"
    if legacy.exists():
        info(f"  removing legacy {legacy}")
        legacy.unlink()


_ASSUME_YES = False


def patch_speechd_conf() -> None:
    info("checking user speechd.conf")
    user_conf = speechd_user_conf()
    if not user_conf.exists():
        system_conf = speechd_system_conf()
        if not system_conf.exists():
            die(f"neither {user_conf} nor {system_conf} exists; install speech-dispatcher first")
        info(f"  seeding {user_conf} from {system_conf}")
        speechd_config_dir().mkdir(parents=True, exist_ok=True)
        user_conf.write_text(system_conf.read_text())

    text = user_conf.read_text()

    # Drop the legacy sd_generic AddModule line if it's still there.
    if _LEGACY_ADD_MODULE_LINE in text:
        info("  removing legacy sd_generic AddModule line")
        lines = []
        skip_next_blank = False
        for line in text.splitlines():
            if line.strip() == _LEGACY_ADD_MODULE_LINE:
                skip_next_blank = True
                continue
            if line.strip() == "# Added by neural-tts-daemon installer":
                skip_next_blank = True
                continue
            if skip_next_blank and not line.strip():
                skip_next_blank = False
                continue
            skip_next_blank = False
            lines.append(line)
        text = "\n".join(lines).rstrip() + "\n"
        user_conf.write_text(text)

    add_module = _add_module_line()
    if add_module in text:
        info("  AddModule line already present, skipping")
        return

    if not confirm(
        f"Append the following line to {user_conf}?\n  {add_module}",
        default_no=False,
        assume_yes=_ASSUME_YES,
    ):
        warn("AddModule line not added; add this line manually to "
             f"{user_conf}:\n  {add_module}")
        return

    with user_conf.open("a") as f:
        f.write(f"\n# Added by neural-tts-daemon installer\n{add_module}\n")
    info("  appended AddModule line")


def systemd_enable() -> None:
    info("reloading systemd and enabling sockets")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(
        [
            "systemctl",
            "--user",
            "enable",
            "--now",
            "neural-tts.socket",
            "neural-tts-control.socket",
        ],
        check=True,
    )


def refresh_speechd() -> None:
    info("restarting speech-dispatcher (next client respawn will read new config)")
    # `pkill speech-dispatcher` doesn't match because comm is truncated to 15 chars.
    # `pkill -f speech-dispatcher` matches our own shell. Match the absolute path
    # of the main binary instead — modules (sd_espeak-ng, sd_dummy) get reaped
    # along with their parent.
    subprocess.run(
        ["pkill", "-u", os.environ.get("USER", ""), "-f", "^/usr/bin/speech-dispatcher"]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="install.py")
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Assume yes for all prompts (non-interactive install)",
    )
    args = parser.parse_args(argv)

    global _ASSUME_YES
    _ASSUME_YES = args.yes

    preflight()
    install_unit_files()
    seed_daemon_config()
    install_speechd_module_conf()
    patch_speechd_conf()
    systemd_enable()
    refresh_speechd()

    info("install complete")
    info("verify with: `bin/neural-tts-ctl status` and `spd-say -o neural-tts 'hello'`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
