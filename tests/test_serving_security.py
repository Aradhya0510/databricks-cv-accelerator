"""Input handling for the serving wrappers.

The PyFunc predict path runs on request-controlled input inside a serving
container, so the loader must not be talked into reading local files or
fetching remote URLs.
"""

from __future__ import annotations

import base64
import io

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from src.serving.pyfunc import _BaseCVPyFuncModel  # noqa: E402


def _b64_png(color=(10, 20, 30)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_accepts_base64_string():
    img = _BaseCVPyFuncModel._load_image(_b64_png())
    assert img.size == (8, 8)
    assert img.mode == "RGB"


def test_accepts_image_field_in_a_record():
    img = _BaseCVPyFuncModel._load_image({"image": _b64_png()})
    assert img.size == (8, 8)


def test_accepts_raw_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    assert _BaseCVPyFuncModel._load_image(buf.getvalue()).size == (8, 8)


def test_local_file_paths_are_not_read(tmp_path):
    """A path-looking string must not become a file read."""
    secret = tmp_path / "secret.png"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(secret)

    with pytest.raises(ValueError, match="base64"):
        _BaseCVPyFuncModel._load_image(str(secret))


def test_etc_passwd_style_input_is_rejected():
    with pytest.raises(ValueError, match="base64"):
        _BaseCVPyFuncModel._load_image("/etc/passwd")


def test_url_records_are_refused_explicitly():
    """No server-side fetch — and say so, rather than failing obscurely."""
    with pytest.raises(ValueError, match="URL inputs are not supported"):
        _BaseCVPyFuncModel._load_image({"url": "http://169.254.169.254/latest/meta-data/"})


def test_unknown_record_shape_names_what_it_wanted():
    with pytest.raises(ValueError, match="Expected one of"):
        _BaseCVPyFuncModel._load_image({"picture": "..."})


def test_non_image_base64_is_rejected():
    junk = base64.b64encode(b"not an image at all").decode()
    with pytest.raises(Exception):
        _BaseCVPyFuncModel._load_image(junk)
