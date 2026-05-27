"""Module entry point: `python -m kde_tts_daemon`."""

import sys

from .daemon import main

if __name__ == "__main__":
    sys.exit(main())
