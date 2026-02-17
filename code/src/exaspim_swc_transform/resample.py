"""SWC resampling utilities backed by SNT."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scyjava
from scipy.interpolate import splprep, splev

_TREE_CLS = None
_POINT_IN_IMAGE_CLS = None


def _java_classes():
    global _TREE_CLS
    global _POINT_IN_IMAGE_CLS
    if _TREE_CLS is None or _POINT_IN_IMAGE_CLS is None:
        _TREE_CLS = scyjava.jimport("sc.fiji.snt.Tree")
        _POINT_IN_IMAGE_CLS = scyjava.jimport("sc.fiji.snt.util.PointInImage")
    return _TREE_CLS, _POINT_IN_IMAGE_CLS


def _path_to_ndarray(path) -> np.ndarray:
    nodes = []
    for i in range(path.size()):
        n = path.getNode(i)
        nodes.append([n.x, n.y, n.z])
    return np.array(nodes)


def _resample(points: np.ndarray, node_spacing: float, degree: int = 1) -> np.ndarray:
    _, ind = np.unique(points, axis=0, return_index=True)
    points = points[np.sort(ind)]

    diff = np.diff(points, axis=0)
    length = np.sqrt(np.power(diff, 2).sum(axis=1)).sum()
    quo, rem = divmod(length, node_spacing)

    samples = np.linspace(0, node_spacing * quo, int(quo + 1), endpoint=True)
    if rem != 0:
        samples = np.append(samples, samples[-1] + rem)

    query_points = np.clip(samples / max(samples), a_min=0.0, a_max=1.0)
    tck, _ = splprep(points.T, k=degree)
    return np.array(splev(query_points, tck)).T


def resample_path(path, node_spacing: float, degree: int = 1, start_joins=None):
    _, point_in_image = _java_classes()
    path_points = _path_to_ndarray(path)
    if start_joins is not None:
        sp = path.getStartJoinsPoint()
        path_points = np.vstack((np.array([sp.getX(), sp.getY(), sp.getZ()]), path_points))

    if len(path_points) < 2:
        return path

    resampled = _resample(path_points, node_spacing, degree)
    respath = path.createPath()
    for p in resampled:
        respath.addNode(point_in_image(float(p[0]), float(p[1]), float(p[2])))
    return respath


def resample_tree(tree, node_spacing: float, degree: int = 1) -> None:
    paths = list(tree.list())
    for path in paths:
        start_joins = path.getStartJoins()
        resampled = resample_path(path, node_spacing, degree, start_joins)
        tree.add(resampled)

        if start_joins is not None:
            start_joins_point = path.getStartJoinsPoint()
            path.unsetStartJoin()
            resampled.setStartJoin(start_joins, start_joins_point)

        for child in list(path.getChildren()):
            start_joins_point = child.getStartJoinsPoint()
            closest_idx = resampled.indexNearestTo(
                start_joins_point.getX(),
                start_joins_point.getY(),
                start_joins_point.getZ(),
                float("inf"),
            )
            closest_point = resampled.getNode(closest_idx)
            child.unsetStartJoin()
            child.setStartJoin(resampled, closest_point)

        tree.remove(path)


def resample_swc(in_swc: Path, out_swc: Path, node_spacing: float) -> None:
    tree_cls, _ = _java_classes()
    out_swc.parent.mkdir(parents=True, exist_ok=True)
    tree = tree_cls(str(in_swc))
    resample_tree(tree, node_spacing)
    tree.setRadii(1.0)
    tree.saveAsSWC(str(out_swc))
