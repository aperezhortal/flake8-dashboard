from pathlib import PurePath, PurePosixPath, Path

from flake8_dashboard.utils import relative_path


def test_path_handling():
    """Test that the relative paths are handled always as Linux paths."""

    home = Path.cwd()
    subdirs = ["test", "a", "b", "c"]
    full_path = home.joinpath(PurePath(*subdirs))

    rel_path = relative_path(full_path, home)

    assert rel_path == str(PurePosixPath(*subdirs))
