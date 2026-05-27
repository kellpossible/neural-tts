"""Module entry point: `python -m neural_tts_daemon`."""

import sys

from .daemon import main

if __name__ == "__main__":
    sys.exit(main())
