import pandas as pd
import numpy as np

from .utils import to_squareform


class DistFrame:
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
        data = np.asarray(data)

        test_condensed_matrix(data)

        self.values = data
        if copy:
            self.values = self.values.copy()

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
        return f'<DistFrame {self.shape}>'

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        return NotImplementedError

    def __subtract___(self, other):
        return NotImplementedError

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
        return DistFrame(
            self.values.copy(), columns=self.columns.copy(), index=self.index.copy()
        )

    def max(self, axis=0):
        return NotImplementedError

    def min(self, axis=1):
        return NotImplementedError

    def to_squareform(self):
        """Turn condensed (vector-form) into square-form distance matrix."""
        return pd.DataFrame(
            to_squareform(self.values),
            index=self.index.copy(),
            columns=self.columns.copy(),
        )


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
