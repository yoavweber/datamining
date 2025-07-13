from typing import Sequence, Union
import math

# Cache for point-to-point distances
_point_distance_cache = {}
_euclidean_distance_cache = {}


def manhattan_distance(
    point1: Sequence[Union[int, float]], point2: Sequence[Union[int, float]]
) -> Union[int, float]:
    """
    Calculate the Manhattan distance between two points and show the calculation process.
    Uses a cache to avoid redundant calculations.
    Args:
        point1 (iterable): The first point (e.g., tuple, list, or numpy array).
        point2 (iterable): The second point (e.g., tuple, list, or numpy array).
    Returns:
        int or float: The Manhattan distance between the two points.
    """
    key = tuple(sorted((tuple(point1), tuple(point2))))
    if key in _point_distance_cache:
        print(
            f"[CACHED] Manhattan distance between {point1} and {point2}: {_point_distance_cache[key]}"
        )
        return _point_distance_cache[key]
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimension.")
    print(f"Calculating Manhattan distance between {point1} and {point2}:")
    abs_diffs = []
    for i, (a, b) in enumerate(zip(point1, point2)):
        diff = abs(a - b)
        abs_diffs.append(diff)
        print(f"  |{a} - {b}| = {diff}")
    total = sum(abs_diffs)
    print(f"Sum of absolute differences: {abs_diffs} => {total}")
    _point_distance_cache[key] = total
    return total


def euclidean_distance(
    point1: Sequence[Union[int, float]], point2: Sequence[Union[int, float]]
) -> float:
    """
    Calculate the Euclidean distance between two points and show the calculation process.
    Uses a cache to avoid redundant calculations.
    Args:
        point1 (iterable): The first point (e.g., tuple, list, or numpy array).
        point2 (iterable): The second point (e.g., tuple, list, or numpy array).
    Returns:
        float: The Euclidean distance between the two points.
    """
    key = tuple(sorted((tuple(point1), tuple(point2))))
    if key in _euclidean_distance_cache:
        print(
            f"[CACHED] Euclidean distance between {point1} and {point2}: {_euclidean_distance_cache[key]}"
        )
        return _euclidean_distance_cache[key]
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimension.")
    print(f"Calculating Euclidean distance between {point1} and {point2}:")
    sq_diffs = []
    for i, (a, b) in enumerate(zip(point1, point2)):
        diff = (a - b) ** 2
        sq_diffs.append(diff)
        print(f"  ({a} - {b})^2 = {diff}")
    total = sum(sq_diffs)
    print(f"Sum of squared differences: {sq_diffs} => {total}")
    result = math.sqrt(total)
    print(f"Square root of sum: sqrt({total}) = {result}")
    _euclidean_distance_cache[key] = result
    return result
