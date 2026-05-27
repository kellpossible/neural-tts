"""Single source of truth for which upstream LongCat-AudioDiT commit we use.

Pinning the commit pins everything upstream-side: the model loader code,
the audiodit/ package, AND their requirements.txt. Bumping the pin is the
one-liner change to upgrade.
"""

LONGCAT_UPSTREAM_REPO = "https://github.com/meituan-longcat/LongCat-AudioDiT.git"
LONGCAT_UPSTREAM_COMMIT = "12c76b51d2a8aa6b6c9af5b25cd5ff8f7aa8178a"
