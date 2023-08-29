import numpy as np
from numba import jit


def to_squareform(X):
    """Convert a condensed distance vector to a square-form distance matrix.

    Parameters
    ----------
    X :         array_like
                A condensed distance matrix.

    Returns
    -------
    Y : ndarray
        A redundant distance matrix.

    """
    if not X.data.c_contiguous:
        X = np.ascontiguousarray(X)

    s = X.shape
    if len(s) != 1:
        raise ValueError(
            (
                "The first argument must be one dimensional "
                "array. A %d-dimensional array is not "
                "permitted"
            )
            % len(s)
        )

    if s[0] == 0:
        return np.zeros((1, 1), dtype=X.dtype)

    # Grab the closest value to the square root of the number
    # of elements times 2 to see if the number of elements
    # is indeed a binomial coefficient.
    d = int(np.ceil(np.sqrt(s[0] * 2)))

    # Check that v is of valid dimensions.
    if d * (d - 1) != s[0] * 2:
        raise ValueError(
            "Incompatible vector size. It must be a binomial "
            "coefficient n choose 2 for some integer n >= 2."
        )

    # Allocate memory for the distance matrix.
    M = np.zeros((d, d), dtype=X.dtype)

    # Fill in the values of the distance matrix.
    _to_squareform_from_vector(M, X)

    # Return the distance matrix.
    return M


@jit(nopython=True)
def _to_squareform_from_vector(M, X):
    """Fill squareform matrix from condensed distance vector."""
    # Initial row length (keep in mind diagonal is not present)
    row_length = M.shape[0] - 1
    next_row_at = row_length
    row = 0
    col = 0
    for i in range(len(X)):
        col += 1
        if i == next_row_at:
            row += 1
            col = row + 1
            row_length -= 1
            next_row_at += row_length
        M[row, col] = X[i]
        M[col, row] = X[i]


def extract_row(X, row):
    """Extract column from condensed distance vector."""
    if not X.data.c_contiguous:
        X = np.ascontiguousarray(X)

    d = int(np.ceil(np.sqrt(len(X) * 2)))

    M = np.zeros(d, dtype=X.dtype)

    _extract_row(M, X, row)

    return M


@jit(nopython=True)
def _extract_row(M, X, r):
    """Fill vector for given column from condensed distance vector."""
    # Initial row length (keep in mind diagonal is not present)
    row_length = M.shape[0] - 1
    next_row_at = row_length
    row = 0
    col = 0
    n_written = 0
    for i in range(len(X)):
        col += 1

        if i == next_row_at:
            col = row + 1
            row += 1
            row_length -= 1
            next_row_at += row_length

        if row == r:
            M[col + 1] = X[i]
            n_written += 1
        elif col == r:
            M[row] = X[i]
            n_written += 1

        # Stop early once we collected all values
        if n_written > row_length:
            break



