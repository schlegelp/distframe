import pandas as pd
import numpy as np

from .utils import to_squareform, get_max_axis, get_min_axis


class VectFrame:
    """DataFrame-like wrapper around condensed matrix.

    This is the generic implementation with arbitrary diagonal (i.e. could also
    be used for similarity matrices).

    Parameters
    ----------
    data :      array-like
                Condensed (N, ) matrix.
    diagonal :  int | float
                The value of along the diagonal. For example 0 for condensed
                distance matrices. Note that this will be cast to the same
                data type as `data`.
    index :     array-like
                Index to use for resulting frame. Will default to pandas.RangeIndex
                if no indexing information part of input data and no index provided.
    columns :   array-like
                Column labels to use for resulting frame when data does not have
                them, defaulting to RangeIndex(0, 1, 2, ..., n).

    """

    def __init__(self, data, diagonal, index=None, columns=None, copy=False):
        data = np.asarray(data)
        test_condensed_matrix(data)

        self.values = data
        if copy:
            self.values = self.values.copy()

        self.diagonal = diagonal

        if columns is None:
            self.columns = pd.RangeIndex(0, len(self))
        else:
            self.columns = pd.RangeIndex(0, len(self))

        if index is None:
            self.index = pd.RangeIndex(0, len(self))
        else:
            self.index = pd.RangeIndex(0, len(self))

        self.iloc = iLocIndexer(self)
        self.loc = LocIndexer(self)

    def __str__(self):
        return f"<VectFrame {self.shape}>"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        return VectFrame(
            self.values + other,
            diagonal=self.diagonal + other,
            index=self.index.copy(),
            columns=self.columns.copy(),
        )

    def __subtract___(self, other):
        return VectFrame(
            self.values - other,
            diagonal=self.diagonal - other,
            index=self.index.copy(),
            columns=self.columns.copy(),
        )

    def __getattr__(self, name):
        if name not in self.columns:
            raise AttributeError(f"'DistFrame' object has no attribute '{name}'")
        return self[name]

    def __len__(self):
        return self.shape[0]

    @property
    def size(self):
        return self.values.shape[0]

    @property
    def shape(self):
        n = int(np.ceil(np.sqrt(self.size * 2)))
        return (n, n)

    def copy(self):
        """Make copy of self."""
        return VectFrame(
            self.values.copy(),
            diagonal=self.diagonal,
            columns=self.columns.copy(),
            index=self.index.copy(),
        )

    def max(self, axis=0):
        """Return the maximum of the values over the requested axis.

        Parameters
        ----------
        axis :  0 (index) | 1 (columns) | None
                Over which axis to calculate the maximum values. Set to ``None``
                to get the overall maximum.

        """
        assert axis in (None, 0, 1)

        if axis is None:
            mx = self.values.max()
            if mx < self.diagonal:
                mx = self.diagonal
        else:
            mx = pd.Series(
                get_max_axis(self.values, diagonal=self.diagonal),
                index=self.index if axis == 0 else self.columns,
            )
        return mx

    def min(self, axis=0):
        """Return the minimum of the values over the requested axis.

        Parameters
        ----------
        axis :  0 (index) | 1 (columns) | None
                Over which axis to calculate the minimum values. Set to ``None``
                to get the overall minimum.

        """
        assert axis in (None, 0, 1)

        if axis is None:
            mn = self.values.min()
            if mn < self.diagonal:
                mn = self.diagonal
        else:
            mn = pd.Series(
                get_min_axis(self.values, diagonal=self.diagonal),
                index=self.index if axis == 0 else self.columns,
            )
        return mn

    def to_squareform(self):
        """Turn condensed (vector-form) into square-form matrix.

        Returns
        -------
        M :     pandas.DataFrame
                Redundant matrix.

        """
        return pd.DataFrame(
            to_squareform(self.values, diagonal=self.diagonal),
            index=self.index.copy(),
            columns=self.columns.copy(),
        )


class DistFrame(VectFrame):
    """DataFrame-like wrapper around condensed distance matrix.

    Parameters
    ----------
    data :      array-like
                Condensed (N, ) distance matrix.
    index :     array-like
                Index to use for resulting frame. Will default to pandas.RangeIndex
                if no indexing information part of input data and no index provided.
    columns :   array-like
                Column labels to use for resulting frame when data does not have
                them, defaulting to RangeIndex(0, 1, 2, ..., n).

    """

    def __init__(self, data, index=None, columns=None, copy=False):
        super().__init__(data, diagonal=0, index=index, columns=columns, copy=copy)

    def __str__(self):
        return f"<DistFrame {self.shape}>"


def test_condensed_matrix(data):
    """Test condensed matrix and raise if issues."""
    if not data.ndim == 1:
        raise ValueError("Condensed distance matrix must be one-dimensional")

    s = data.shape
    d = int(np.ceil(np.sqrt(s[0] * 2)))
    if d * (d - 1) != s[0] * 2:
        raise ValueError(
            "Incompatible vector size. It must be a binomial "
            "coefficient n choose 2 for some integer n >= 2."
        )


class LocIndexer:
    """Access a group of rows and columns by label(s) or a boolean array.

    Analogous to pandas `.loc` indexer.
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, key):
        ix = self.key_to_indices(key)

    def __setitem__(self, key, value):
        return NotImplementedError

    def key_to_indices(self, key):
        print(key)
        if isinstance(key, tuple):
            if len(tuple) == 1:
                pass


class iLocIndexer(LocIndexer):
    """Purely integer-location based indexing for selection by position.

    Analogous to pandas `.iloc` indexer.
    """
