import numpy as np
from numba import jit


def to_squareform(X, diagonal=0):
    """Convert a condensed distance vector to a square-form distance matrix.

    Parameters
    ----------
    X :         array_like
                A condensed matrix.
    diagonal :  float | int
                The value along the diagonal. Defaults to 0 which implies
                distance matrix.

    Returns
    -------
    M : ndarray
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
    M = np.full((d, d), fill_value=diagonal, dtype=X.dtype)

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


def extract_row(X, row_ix, diagonal=0):
    """Extract column from condensed distance vector.

    Parameters
    ----------
    X :         np.ndarray
                The condensed matrix.
    row_ix :    int
                Index of the row to extract.
    diagonal :  float | int
                The value along the diagonal. Defaults to 0 which implies
                distance matrix.

    Returns
    -------
    row :       np.ndarray

    """
    if not X.data.c_contiguous:
        X = np.ascontiguousarray(X)

    # Get the size of the matrix
    d = int(np.ceil(np.sqrt(len(X) * 2)))

    # Generate an empty frame for us to fill
    M = np.full(d, fill_value=diagonal, dtype=X.dtype)

    # Fill row
    _extract_row(M, X, row_ix)

    return M


@jit(nopython=True)
def _extract_row(M, X, row_ix):
    """Fill vector for given row from condensed distance vector."""
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

        # We have to fill from both row and column to get the full vector
        if row == row_ix:
            M[col + 1] = X[i]
            n_written += 1
        elif col == row_ix:
            M[row] = X[i]
            n_written += 1

        # Stop early once if we already collected all values
        # (this probably needs a bit of testing to see if it actually
        #  speeds things up)
        if n_written > row_length:
            break


def get_indices(X):
    """Extract row and column indices for condensed distance vector.

    Parameters
    ----------
    X :         np.ndarray
                The condensed matrix.

    Returns
    -------
    row_ix :    np.ndarray
    col_ix :    np.ndarray
                For each entry in `X` get the row and column index.

    """
    if not X.data.c_contiguous:
        X = np.ascontiguousarray(X)

    # Get the size of the matrix
    d = int(np.ceil(np.sqrt(len(X) * 2)))

    # Prepare empty vectors
    row_ix = np.zeros(len(X), dtype=np.int64)
    col_ix = np.zeros(len(X), dtype=np.int64)

    # Fill
    _get_indices(X, row_ix, col_ix, d)

    return row_ix, col_ix


@jit(nopython=True)
def _get_indices(X, row_ix, col_ix, d):
    """Row and column indices from condensed distance vector."""
    # Initial row length (keep in mind diagonal is not present)
    row_length = d - 1
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

        row_ix[i] = row
        col_ix[i] = col


def get_max_axis(X, diagonal=0):
    """Extract max values along axis.

    Parameters
    ----------
    X :         np.ndarray
                The condensed matrix.
    diagonal :  float | int
                The value along the diagonal. Defaults to 0 which implies
                distance matrix.

    Returns
    -------
    max :       np.ndarray
                Max values along axis.

    """
    if not X.data.c_contiguous:
        X = np.ascontiguousarray(X)

    # Get the size of the matrix
    d = int(np.ceil(np.sqrt(len(X) * 2)))

    # Prepare empty vectors
    M = np.full(d, fill_value=diagonal, dtype=X.dtype)

    # Fill
    _get_max_axis(M, X, d)

    return M


@jit(nopython=True)
def _get_max_axis(M, X, d):
    """Extract max values along axis from condensed distance vector."""
    # Initial row length (keep in mind diagonal is not present)
    row_length = d - 1
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

        if M[col] < X[i]:
            M[col] = X[i]
        if M[row] < X[i]:
            M[row] = X[i]


def get_min_axis(X, diagonal=0):
    """Extract min values along axis.

    Parameters
    ----------
    X :         np.ndarray
                The condensed matrix.
    diagonal :  float | int
                The value along the diagonal. Defaults to 0 which implies
                distance matrix.

    Returns
    -------
    min :       np.ndarray
                Min values along axis.

    """
    if not X.data.c_contiguous:
        X = np.ascontiguousarray(X)

    # Get the size of the matrix
    d = int(np.ceil(np.sqrt(len(X) * 2)))

    # Prepare empty vectors
    M = np.full(d, fill_value=diagonal, dtype=X.dtype)

    # Fill
    _get_min_axis(M, X, d)

    return M


@jit(nopython=True)
def _get_min_axis(M, X, d):
    """Extract min values along axis from condensed distance vector."""
    # Initial row length (keep in mind diagonal is not present)
    row_length = d - 1
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

        if M[col] > X[i]:
            M[col] = X[i]
        if M[row] > X[i]:
            M[row] = X[i]