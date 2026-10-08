"""Compatibility settings for writing AnnData files."""

import anndata as ad
import pandas as pd


def _cast_string_columns_to_object(frame):
    if frame is None:
        return
    for column in frame.columns:
        values = frame[column]
        if pd.api.types.is_string_dtype(values) and values.dtype != object:
            frame[column] = values.astype(object)
    if pd.api.types.is_string_dtype(frame.index) and frame.index.dtype != object:
        frame.index = frame.index.astype(object)


def enable_anndata_write_compatibility():
    """Use object-backed strings for portable h5ad serialization."""
    try:
        pd.options.future.infer_string = False
    except Exception:
        pass

    original_write = ad.AnnData.write_h5ad
    if getattr(original_write, "_posse_compatible", False):
        return

    def compatible_write(self, *args, **kwargs):
        _cast_string_columns_to_object(self.obs)
        _cast_string_columns_to_object(self.var)
        return original_write(self, *args, **kwargs)

    compatible_write._posse_compatible = True
    ad.AnnData.write_h5ad = compatible_write
    ad.AnnData.write = compatible_write
