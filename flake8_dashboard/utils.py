# -*- coding: utf-8 -*-
"""Common utilities."""

import os

import numpy as np
from astroid import MANAGER, AstroidSyntaxError
from pathlib import PurePosixPath, PurePath


class ASTWalker:
    """
    Statements counter for Python source codes.

    Usage
    =====
    my_counter = ASTWalker()
    statements = my_counter.count_statements("path/to/my/file")
    """

    # Tested with astroid 2.3.0.dev0

    def __init__(self):
        self.nbstatements = 0

    def count_statements(self, filepath):
        self.nbstatements = 0
        try:
            ast_node = MANAGER.ast_from_file(filepath, source=True)
            self._walk(ast_node)
        except AstroidSyntaxError:
            self.nbstatements = np.nan
        return self.nbstatements

    def _walk(self, astroid_node):
        """
        Recurse in the astroid node children and count the statements.
        """
        if astroid_node.is_statement:
            self.nbstatements += 1

        # recurse on children
        for child in astroid_node.get_children():
            self._walk(child)


def create_dir(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def relative_path(full_path, common_prefix):
    """Returns the relative path as a linux path to avoid problems with
    regular expressions in windows paths.
    """

    rel_path = PurePath(full_path).relative_to(PurePath(common_prefix))
    return rel_path.as_posix()


def full_split(_path):
    """
    Return a list with all the intermediate paths.
    The input path must be a POSIX path string (i.e., Linux or OSX).
    """
    intermediate_paths = list()

    _path = PurePosixPath(_path)

    if _path.is_absolute():
        _path = _path.relative_to("/")

    parts = _path.parts

    for i in range(1, len(parts)):
        intermediate_paths.append(PurePosixPath(*parts[0:i]).as_posix())

    return intermediate_paths


def interp_to_previous(x, xp, fp, **kwargs):
    """
    One-dimensional linear interpolation.

    Returns the one-dimensional nearest-lower-neighbor interpolant to a function
    with given discrete data points (`xp`, `fp`), evaluated at `x`.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.

    xp : 1-D sequence of floats
        The x-coordinates of the data points, must be increasing if argument
        `period` is not specified. Otherwise, `xp` is internally sorted after
        normalizing the periodic boundaries with ``xp = xp % period``.

    fp : 1-D sequence of float or complex
        The y-coordinates of the data points, same length as `xp`.

    Returns
    -------
    y : float or complex (corresponding to fp) or ndarray
        The interpolated values, same shape as `x`.

    Raises
    ------
    ValueError
        If `xp` and `fp` have different length
        If `xp` or `fp` are not 1-D sequences
        If `period == 0`

    Notes
    -----
    Does not check that the x-coordinate sequence `xp` is increasing.
    If `xp` is not increasing, the results are nonsense.
    A simple check for increasing is::

        np.all(np.diff(xp) > 0)

    Use previous neighbour of x_new, y_new = f(x_new).

    Adapted from scipy.interpolate.interpolate.py
    """
    del kwargs  # Unused, needeed for compatibility with np.interp function.

    xp = np.asarray(xp)
    # 1. Get index of left value
    xp_shift = np.nextafter(xp, -np.inf)

    x_new_indices = np.searchsorted(xp_shift, x, side="left")

    # 2. Clip x_new_indices so that they are within the range of x indices.
    x_new_indices = x_new_indices.clip(1, len(xp)).astype(np.intp)

    # 3. Calculate the actual value for each entry in x_new.
    y_new = fp[x_new_indices - 1]

    return y_new


def map_values_to_cmap(values, colormap=None, discrete=True):
    """
    colormap : Mx4 array-like

    If a Mx4 array-like, the rows define the values (x, r, g, b), where r/g/b are
    number from 0-255.

    The x values must start with x=0, end with x=1.

    discrete true =-> map to discrete colorbar
    """

    if colormap is None:
        colormap = np.asarray(
            [
                [0.0, 239, 85, 59],  # red
                [0.5, 99, 110, 250],  # blue
                [0.75, 0, 204, 150],  # green
                [1.0, 0, 204, 150],  # green
            ]
        )

        # 0.0-0.5 red!
        # 0.5-0.75 - blue
        # 0.75-1 - green
    try:
        colormap = np.asarray(colormap, dtype=float)
        values = np.asarray(values, dtype=float)
    except Exception:
        raise TypeError(
            "colormap and interpolation values must be convertible to an array."
        )

    shape = colormap.shape
    if len(shape) != 2 or shape[1] != 4:
        raise ValueError("colormap must be Mx4 format")

    if len(values) == 0 or values.ndim != 1:
        raise ValueError("Mapping values must be 1D.")

    if discrete:
        interpolator = interp_to_previous
    else:
        interpolator = np.interp

    red = interpolator(
        values, colormap[:, 0], colormap[:, 1], left=0, right=255
    ).astype(int)
    green = interpolator(values, colormap[:, 0], colormap[:, 2]).astype(int)
    blue = interpolator(values, colormap[:, 0], colormap[:, 3]).astype(int)

    colorscale = [
        f"#{red[i]:02x}{green[i]:02x}{blue[i]:02x}" for i in range(values.size)
    ]
    return colorscale
