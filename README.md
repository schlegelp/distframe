# distframe
A DataFrame-like wrapper around condensed distance matrices.

## Rationale 
Symmetric pairwise distance matrix (e.g. from scipy's `pdist`) are typically represented as condensed matrixes that cover only the upper triangle.
This saves >50% memory which may not sound like much but can make the difference between being able to load a matrix into memory on your laptop or not.
Unfortunately, it makes exploring the data a bit more complicated. `distframe` seeks to provide an intuitive DataFrame-like wrapper around the 
condensed matrix to facilitate basic operations.

## Install
TODO

## Examples 
TODO

