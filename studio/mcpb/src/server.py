"""Entry point for the POUNCE Studio Claude Desktop extension (.mcpb).

All of the server lives in pounce-solver's ``pounce.studio`` package, which
the host installs from PyPI per ``pyproject.toml`` beside this file. This
file only starts it.
"""
from pounce.studio import main

if __name__ == "__main__":
    main()
