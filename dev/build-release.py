from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
from pathlib import Path
import logging as lg
import os
import re
import subprocess as sp
import sys
import tempfile as tf
from typing import *


LOG = lg.getLogger("script")
lg.basicConfig(
    format="%(message)s",
    level=lg.INFO
)


def process_args() -> Namespace:
    ap = ArgumentParser(description="Builds and pushes a wheel to PyPI so as to release a new version.")
    ap.add_argument(
        "version",
        help="New version number. Should be higher than latest version."
    )
    return ap.parse_args()


Version = Tuple[str]


def as_version(v: str) -> Version:
    return v.split(".")


def v2s(v: Version) -> str:
    return ".".join(v)


def find_version_number(lines: Sequence[str]) -> Tuple[int, Version]:
    rx_version = re.compile(r'version="([0-9a-zA-Z.]+)",')
    for i, line in enumerate(lines):
        match = rx_version.match(line.strip())
        if match:
            break
    else:
        lg.error(
            "Can't seem to find the version definition line in setup.py.  Please contact the author of this script."
        )
        sys.exit(2)
    return i, as_version(match.group(1))


def go_to_root() -> None:
    os.chdir(str(Path(sys.argv[0]).parent.parent))


@contextmanager
def backing_up_setup_py() -> Iterator[Path]:
    path_setup_py = Path("setup.py")
    if not path_setup_py.is_file():
        lg.error(
            "Cannot find file setup.py. This script relies on modifying this file. "
            "Was the setup tooling changed?"
        )
        sys.exit(1)

    with tf.NamedTemporaryFile() as file:
        file.write(path_setup_py.read_bytes())
        file.flush()
        try:
            yield path_setup_py
        except:
            lg.info("Undo version number update")
            file.seek(0, 0)
            path_setup_py.write_bytes(file.read())
            raise


def set_new_version_number(path_setup_py: Path, version_new: Version) -> None:
    lines = path_setup_py.read_text(encoding="utf-8").split("\n")
    i_version, version_old = find_version_number(lines)
    if version_new <= version_old:
        lg.warning(
            f"Warning: the new version number {v2s(version_new)} is not posterior to the old one ({v2s(version_old)})."
        )
        answer = input("Please confirm whether to carry on [y/N]: ")
        if not answer.lower().startswith("y"):
            lg.error("Abort.")
            sys.exit(0)
    lines[i_version] = f'    version="{v2s(version_new)}",\n'


@contextmanager
def temporary_virtual_env() -> Iterator[Path]:
    try:
        with tf.TemporaryDirectory() as d:
            sp.run([sys.executable, "-m", "venv", str(d)], check=True)
            yield Path(d)
    finally:
        LOG.info("Temporary virtual environment dismantled")


def executable(d: Path) -> Path:
    return d / "bin"


def python(d: Path) -> Path:
    return executable(d) / "python"


def pip(d: Path) -> Path:
    return executable(d) / "pip"


def twine(d: Path) -> Path:
    return executable(d) / "twine"


if __name__ == "__main__":
    args = process_args()
    go_to_root()
    with backing_up_setup_py() as path_setup_py:
        lg.info("Updating version number")
        set_new_version_number(path_setup_py, as_version(args.version))
        lg.info("Preparing temporary virtual environment")
        with temporary_virtual_env() as dir_venv:
            lg.info("Setting up necessary packages in virtual environment")
            sp.run([pip(dir_venv), "install", "build", "setuptools", "twine"], check=True)
            lg.info("Building source distribution and wheel")
            sp.run([python(dir_venv), "-m", "build"], check=True)
            lg.info("Pushing distribution files to PyPI using twine")
            sp.run([twine(dir_venv), "upload", "dist/*"], check=True)
    print("""\
--------------------------------------------------------------------------------------------------
Package upload was successful.

Running this script has altered the `setup.py' file in the repository. You should now review this
change, commit it to the repository, and push this to Github.
--------------------------------------------------------------------------------------------------
""")
