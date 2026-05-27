#!/usr/bin/env python3
"""Reverse the install. Removes user-scope unit files and the speechd module config.

Leaves `~/.local/share/neural-tts-daemon/` (models, cloned voices) and provider
venvs intact — those are user data, deleted explicitly by the user if desired.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    confirm,
    info,
    speechd_modules_dir,
    speechd_user_conf,
    systemd_user_dir,
    warn,
)

ADD_MODULE_LINES = [
    'AddModule "neural-tts-generic" "sd_generic" "neural-tts-generic.conf"',  # legacy
]


def _candidate_addmodule_lines(text: str) -> list[str]:
    out = list(ADD_MODULE_LINES)
    # Match the current native-module form regardless of repo path.
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('AddModule "neural-tts" "') and s.endswith('"neural-tts.conf"'):
            out.append(s)
    return out


def main(argv: list[str] | None = None) -> int:
    info("stopping & disabling units")
    subprocess.run(
        [
            "systemctl",
            "--user",
            "disable",
            "--now",
            "neural-tts.socket",
            "neural-tts-control.socket",
            "neural-tts.service",
        ]
    )

    for name in ("neural-tts.service", "neural-tts.socket", "neural-tts-control.socket"):
        f = systemd_user_dir() / name
        if f.exists():
            info(f"  removing {f}")
            f.unlink()

    for name in ("neural-tts.conf", "neural-tts-generic.conf"):
        f = speechd_modules_dir() / name
        if f.exists():
            info(f"  removing {f}")
            f.unlink()

    user_conf = speechd_user_conf()
    if user_conf.exists():
        text = user_conf.read_text()
        targets = _candidate_addmodule_lines(text)
        if any(t in text for t in targets):
            if confirm(
                f"Remove the neural-tts AddModule line(s) from {user_conf}?",
                default_no=False,
            ):
                new_lines = []
                skip_next_blank = False
                for line in text.splitlines():
                    if line.strip() in targets:
                        skip_next_blank = True
                        continue
                    if line.strip() == "# Added by neural-tts-daemon installer":
                        skip_next_blank = True
                        continue
                    if skip_next_blank and not line.strip():
                        skip_next_blank = False
                        continue
                    skip_next_blank = False
                    new_lines.append(line)
                user_conf.write_text("\n".join(new_lines).rstrip() + "\n")
                info("  removed AddModule line(s)")

    subprocess.run(["systemctl", "--user", "daemon-reload"])
    info("restarting speech-dispatcher")
    subprocess.run(
        ["pkill", "-u", os.environ.get("USER", ""), "-f", "^/usr/bin/speech-dispatcher"]
    )

    info("uninstall complete")
    warn(
        "models and config files left in:\n"
        f"  ~/.config/neural-tts-daemon/\n"
        f"  ~/.local/share/neural-tts-daemon/\n"
        f"  providers/*/.venv\n"
        f"Remove manually if you want to fully clean up."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
