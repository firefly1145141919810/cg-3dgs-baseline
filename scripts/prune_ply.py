"""CPU-only exact row selection for the pinned SH3 Gaussian PLY schema."""
import math
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement


def read_vertices(path):
    ply = PlyData.read(path)
    if len(ply.elements) != 1 or ply.elements[0].name != 'vertex':
        raise ValueError('Expected one vertex element, not a mesh or generic point cloud')
    vertices = ply['vertex'].data
    expected = (['x', 'y', 'z', 'nx', 'ny', 'nz'] +
                [f'f_dc_{i}' for i in range(3)] +
                [f'f_rest_{i}' for i in range(45)] + ['opacity'] +
                [f'scale_{i}' for i in range(3)] +
                [f'rot_{i}' for i in range(4)])
    if set(vertices.dtype.names or ()) != set(expected):
        raise ValueError('Expected official degree-3 Gaussian fields (62 float32 columns)')
    if not len(vertices):
        raise ValueError('Empty Gaussian model')
    for name in expected:
        dtype = vertices.dtype.fields[name][0]
        if dtype.kind != 'f' or dtype.itemsize != 4:
            raise ValueError('Expected float32 field: ' + name)
        if not np.isfinite(vertices[name]).all():
            raise ValueError('Nonfinite field: ' + name)
    return vertices


def select_indices(logits, keep_fraction):
    if not math.isfinite(keep_fraction) or not 0 < keep_fraction <= 1:
        raise ValueError('keep_fraction must be in (0, 1]')
    logits = np.asarray(logits)
    if logits.ndim != 1 or not len(logits) or not np.isfinite(logits).all():
        raise ValueError('Expected finite nonempty 1D opacity logits')
    count = max(1, math.floor(len(logits) * keep_fraction))
    # sigmoid is strictly increasing: rank logits, avoiding sigmoid saturation.
    # Stable sorting breaks equal-score ties by original row index.
    order = np.argsort(-logits.astype(np.float64), kind='stable')[:count]
    # Retain source row order: sorting all rows by score can alter depth ties.
    return np.sort(order)


def write_pruned(source, destination, keep_fraction):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Refusing to overwrite model: ' + str(destination))
    vertices = read_vertices(source)
    indices = select_indices(vertices['opacity'], keep_fraction)
    selected = vertices[indices].copy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(selected, 'vertex')], text=False, byte_order='<').write(destination)
    loaded = read_vertices(destination)
    equal = len(loaded) == len(selected) and all(
        loaded[n].tobytes() == selected[n].tobytes() for n in selected.dtype.names)
    if not equal:
        raise ValueError('Saved PLY differs from selected source parameters')
    return indices, {'source_count': len(vertices), 'kept_count': len(indices),
                     'keep_fraction_requested': keep_fraction,
                     'keep_fraction_actual': len(indices) / len(vertices),
                     'all_selected_fields_bitwise_equal': equal,
                     'order': 'original rows; ties ranked by ascending original index',
                     'score': 'raw opacity logit, monotonic with sigmoid opacity',
                     'source_ply_bytes': Path(source).stat().st_size,
                     'output_ply_bytes': destination.stat().st_size}
