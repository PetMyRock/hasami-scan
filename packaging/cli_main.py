"""Entry point of hasami.exe (the command line: hasami list, diag, scan ...)."""

import sys

from hasami.cli import main

if __name__ == "__main__":
    sys.exit(main())
