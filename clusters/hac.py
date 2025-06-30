from typing import Sequence, Union

from .distance import manhattan_distance


def print_distance_matrix(step, cluster_names, distance_table):
    print(f"\n# Distance matrix for Step {step} (rows/cols = clusters):")
    header = "|   |" + "|".join([f" {name} " for name in cluster_names]) + "|"
    separator = "|---" * (len(cluster_names) + 1) + "|"
    print(header)
    print(separator)
    for i, row in enumerate(distance_table):
        row_str = (
            f"| {cluster_names[i]} |"
            + "|".join([f" {d:.2f} " if d is not None else "  -  " for d in row])
            + "|"
        )
        print(row_str)


def single_linkage(
    cluster1: Sequence[Sequence[Union[int, float]]],
    cluster2: Sequence[Sequence[Union[int, float]]],
    distance_func=manhattan_distance,
) -> Union[int, float]:
    """
    Calculate the single linkage (minimum distance) between two clusters and show the calculation process.
    Args:
        cluster1: Sequence of points (each a sequence of numbers).
        cluster2: Sequence of points (each a sequence of numbers).
        distance_func: Function to calculate distance between two points.
    Returns:
        The minimum distance between any pair of points (one from each cluster).
    Raises:
        ValueError: If either cluster is empty.
    """
    print(
        f"Calculating single linkage between clusters:\n  Cluster 1: {cluster1}\n  Cluster 2: {cluster2}"
    )
    min_distance = None
    min_pair = None
    for i, p1 in enumerate(cluster1):
        for j, p2 in enumerate(cluster2):
            print(
                f"\nDistance between point {i} in cluster1 and point {j} in cluster2:"
            )
            dist = distance_func(p1, p2)
            if (min_distance is None) or (dist < min_distance):
                min_distance = dist
                min_pair = (p1, p2)
    if min_distance is None or min_pair is None:
        raise ValueError("Both clusters must contain at least one point.")
    print(
        f"\nSingle linkage (minimum distance) is {min_distance} between points {min_pair[0]} and {min_pair[1]}"
    )
    return min_distance


def hac(
    data: Sequence[Sequence[Union[int, float]]],
    linkage_func,
    distance_func=manhattan_distance,
):
    """
    Perform Hierarchical Agglomerative Clustering (HAC) and print the process step by step.
    Args:
        data: Sequence of points (each a sequence of numbers).
        linkage_func: Function to calculate linkage between two clusters (should accept distance_func).
        distance_func: Function to calculate distance between two points.
    """
    from clusters.distance import _point_distance_cache

    clusters = [[point] for point in data]
    cluster_names = [f"C{i+1}" for i in range(len(data))]
    step = 1
    prev_new_cluster = None
    while len(clusters) > 1:
        print(f"\n{'='*40}\nStep {step}")
        print("Current clusters:")
        for name, clust in zip(cluster_names, clusters):
            print(f"  {name}: {clust}")
        n = len(clusters)
        distance_table = [[None] * n for _ in range(n)]
        min_dist = None
        min_pair = None
        new_cluster_idx = n - 1 if step > 1 else None
        for i in range(n):
            for j in range(i + 1, n):
                all_cached = True
                cached_distances = []
                if new_cluster_idx is not None and (
                    i == new_cluster_idx or j == new_cluster_idx
                ):
                    if prev_new_cluster != (i, j):
                        print(
                            f"\n--- Calculating distances involving new cluster {cluster_names[new_cluster_idx]} ---"
                        )
                        prev_new_cluster = (i, j)
                for p1 in clusters[i]:
                    for p2 in clusters[j]:
                        key = tuple(sorted((tuple(p1), tuple(p2))))
                        if key in _point_distance_cache:
                            print(
                                f"[CACHED] Manhattan distance between {p1} and {p2}: {_point_distance_cache[key]}"
                            )
                            cached_distances.append(_point_distance_cache[key])
                        else:
                            all_cached = False
                if all_cached:
                    dist = min(cached_distances) if cached_distances else None
                    distance_table[i][j] = dist
                    distance_table[j][i] = dist
                    if dist is not None and (min_dist is None or dist < min_dist):
                        min_dist = dist
                        min_pair = (i, j)
        for i in range(n):
            for j in range(i + 1, n):
                if distance_table[i][j] is not None:
                    continue
                if new_cluster_idx is not None and (
                    i == new_cluster_idx or j == new_cluster_idx
                ):
                    if prev_new_cluster != (i, j):
                        print(
                            f"\n--- Calculating distances involving new cluster {cluster_names[new_cluster_idx]} ---"
                        )
                        prev_new_cluster = (i, j)
                print(
                    f"\n--- New single linkage calculation for {cluster_names[i]} and {cluster_names[j]} ---"
                )
                dist = linkage_func(clusters[i], clusters[j], distance_func)
                distance_table[i][j] = dist
                distance_table[j][i] = dist
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    min_pair = (i, j)
        print(f"\n# Distance matrix for Step {step} (rows/cols = clusters):")
        header = "|   |" + "|".join([f" {name} " for name in cluster_names]) + "|"
        separator = "|---" * (n + 1) + "|"
        print(header)
        print(separator)
        for i, row in enumerate(distance_table):
            row_str = (
                f"| {cluster_names[i]} |"
                + "|".join([f" {d:.2f} " if d is not None else "  -  " for d in row])
                + "|"
            )
            print(row_str)
        if min_pair is None:
            raise ValueError("No clusters can be joined. Check input data.")
        i, j = min_pair
        print(
            f"\n--- New single linkage calculation for {cluster_names[i]} and {cluster_names[j]} (MERGE) ---"
        )
        linkage_func(clusters[i], clusters[j], distance_func)
        prev_nums = "".join([c[1:] for c in [cluster_names[i], cluster_names[j]]])
        new_name = f"C{prev_nums}"
        print(
            f"\nJoining clusters {cluster_names[i]} and {cluster_names[j]} (distance = {min_dist:.2f})"
        )
        new_cluster = clusters[i] + clusters[j]
        new_clusters = [clusters[k] for k in range(n) if k not in (i, j)]
        new_names = [cluster_names[k] for k in range(n) if k not in (i, j)]
        new_clusters.append(new_cluster)
        new_names.append(new_name)
        clusters = new_clusters
        cluster_names = new_names
        step += 1
    print(f"\n{'='*40}\nClustering complete. Final cluster: {clusters[0]}")
