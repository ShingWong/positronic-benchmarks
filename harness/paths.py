# =====================================================================
# Project Positronic — Polytemporal Cognitive Engram Memory Substrate
# Copyright (C) 2026 Shing Wong. All Rights Reserved.
# =====================================================================
# This program is DUAL-LICENSED. You may redistribute and/or modify it 
# under the terms of the GNU Affero General Public License as published by the 
# Free Software Foundation, either version 3 of the License, or (at your 
# option) any later version.
#
# Alternatively, commercial entities, multi-tenant instances, and Managed 
# Service Providers (MSPs) may utilize this program under a separate, 
# proprietary Commercial License Waiver issued directly by the copyright 
# holder, completely exempt from the network-use copyleft restrictions of 
# the AGPLv3 Section 13.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License 
# along with this program. If not, see <https://gnu.org>.
# =====================================================================

"""Locate the memeng engine and related paths without hardcoding a machine.

Everything here resolves in this order:

1. An explicit environment variable, so a checkout anywhere on disk works.
2. An already-installed ``memeng`` (the normal case after ``pip install``).
3. A sibling ``positronic-engram`` checkout, for a bare workspace clone.

There is deliberately no absolute fallback: a path baked into a published
repository only ever works on the machine that wrote it, and it discloses
that machine's directory layout to everyone else.
"""

import importlib.util
import os
import sys
from pathlib import Path

#: Repository root of this project (the directory containing ``harness/``).
ROOT = Path(__file__).resolve().parents[1]

#: Where an optional private corpus may live, if the operator has one.
PRIVATE_DIR_ENV = "POSITRONIC_PRIVATE_DIR"

#: Where a HuggingFace dataset cache may live, for the longmemeval suite.
HF_CACHE_ENV = "HF_DATASETS_CACHE"


def memeng_installed() -> bool:
    """True when ``memeng`` is importable without any path help."""
    try:
        return importlib.util.find_spec("memeng") is not None
    except (ImportError, ValueError):
        return False


def engram_src() -> Path | None:
    """Directory to put on sys.path to import memeng, or None if unavailable.

    Honours ``POSITRONIC_ENGRAM_SRC`` first, then a sibling checkout.
    """
    env = os.environ.get("POSITRONIC_ENGRAM_SRC", "").strip()
    candidates = []
    if env:
        candidates.append(Path(env))
    # sibling checkout: <workspace>/positronic-engram/engine/src
    candidates.append(ROOT.parent / "positronic-engram" / "engine" / "src")
    # nested layout, if this repo was cloned inside the engine checkout
    candidates.append(ROOT / "positronic-engram" / "engine" / "src")
    for c in candidates:
        if (c / "memeng").is_dir():
            return c
    return None


def ensure_memeng() -> str:
    """Make ``memeng`` importable. Returns a short description of the source."""
    if memeng_installed():
        return "installed"
    src = engram_src()
    if src is None:
        raise RuntimeError(
            "memeng is not importable. Either `pip install memeng`, or set "
            "POSITRONIC_ENGRAM_SRC to positronic-engram/engine/src, or clone "
            "positronic-engram next to this repository."
        )
    sys.path.insert(0, str(src))
    return f"sys.path: {src}"


def private_dir() -> Path | None:
    """Optional private corpus directory, or None. Never a default path.

    The private corpus is not part of this repository. Without the
    environment variable set, suites that can replay real data fall back to
    synthetic data rather than guessing at a location.
    """
    env = os.environ.get(PRIVATE_DIR_ENV, "").strip()
    if not env:
        return None
    p = Path(env)
    return p if p.is_dir() else None


def hf_cache() -> Path | None:
    """HuggingFace cache root for the longmemeval suite, or None if unset."""
    env = os.environ.get(HF_CACHE_ENV, "").strip()
    if not env:
        return None
    p = Path(env)
    return p if p.is_dir() else None
