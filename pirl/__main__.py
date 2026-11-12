"""Package entry point; command implementation lives in :mod:`pirl.cli`."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
