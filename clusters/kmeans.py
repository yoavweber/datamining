import random
from typing import List, Tuple, Optional, Sequence, Union
from clusters.distance import manhattan_distance


def print_centroids(step: int, centroids: List[Tuple[Union[int, float], ...]]):
    print(f"\n# Centroids at Step {step}:")
    for idx, c in enumerate(centroids):
        print(f"  C{idx+1}: {c}")


def print_assignments(assignments: List[int], points: Sequence[Tuple], k: int):
    print("\n# Cluster assignments:")
    for cluster_idx in range(k):
        members = [
            f"P{i+1}{points[i]}" for i, a in enumerate(assignments) if a == cluster_idx
        ]
        print(
            f"  Cluster {cluster_idx+1}: {', '.join(members) if members else 'EMPTY'}"
        )


def compute_centroid(cluster: List[Tuple[Union[int, float], ...]]) -> Tuple[float, ...]:
    if not cluster:
        raise ValueError("Cannot compute centroid of empty cluster.")
    dim = len(cluster[0])
    centroid = []
    for d in range(dim):
        vals = [point[d] for point in cluster]
        mean = sum(vals) / len(vals)
        centroid.append(mean)
    return tuple(centroid)


def print_centroid_calculation(cluster: List[Tuple[Union[int, float], ...]], idx: int):
    if not cluster:
        print(f"  Cluster {idx+1} is empty. Centroid undefined.")
        return
    print(f"  Cluster {idx+1} members: {cluster}")
    dim = len(cluster[0])
    for d in range(dim):
        vals = [point[d] for point in cluster]
        print(
            f"    Dimension {d+1}: mean of {vals} = {sum(vals)} / {len(vals)} = {sum(vals)/len(vals):.2f}"
        )


def kmeans_verbose(
    data: Sequence[Tuple[Union[int, float], ...]],
    k: int = 2,
    initial_centroids: Optional[Sequence[Tuple[Union[int, float], ...]]] = None,
    distance_func=manhattan_distance,
    max_iter: int = 100,
    random_seed: Optional[int] = 42,
):
    """
    Perform k-means clustering and print all calculation steps.
    Args:
        data: List of points (tuples).
        k: Number of clusters.
        initial_centroids: Optional list of initial centroids.
        distance_func: Distance function.
        max_iter: Maximum number of iterations.
        random_seed: Seed for reproducibility if centroids are random.
    """
    if not data:
        print("No data provided.")
        return
    n = len(data)
    if initial_centroids is not None:
        centroids = [tuple(c) for c in initial_centroids]
        print("Initial centroids provided:")
    else:
        random.seed(random_seed)
        centroids = random.sample(data, k)
        print(f"Randomly selected initial centroids (seed={random_seed}):")
    print_centroids(0, centroids)
    assignments = [-1] * n
    for step in range(1, max_iter + 1):
        print(f"\n{'='*40}\nStep {step}")
        # Assignment step
        new_assignments = []
        all_dists = []  # For the distance table
        for i, point in enumerate(data):
            print(f"\nPoint P{i+1}{point} distances to centroids:")
            dists = []
            for j, centroid in enumerate(centroids):
                print(f"  To centroid C{j+1} {centroid}:")
                dist = distance_func(point, centroid)
                print(f"    Manhattan distance = {dist}")
                dists.append(dist)
            min_dist = min(dists)
            assigned = dists.index(min_dist)
            print(f"  => Assigned to cluster {assigned+1} (distance = {min_dist})")
            new_assignments.append(assigned)
            all_dists.append(dists)
        print_assignments(new_assignments, data, k)
        # Print markdown table of distances
        print("\n# Distance Table (Step {}):".format(step))
        header = "| Point |" + "|".join([f" C{j+1} " for j in range(k)]) + "|"
        separator = "|---" * (k + 1) + "|"
        print(header)
        print(separator)
        for i, dists in enumerate(all_dists):
            row = f"| P{i+1} " + "|".join([f" {d:.2f} " for d in dists]) + "|"
            print(row)
        # Check for convergence
        if new_assignments == assignments:
            print("\nNo change in assignments. Clustering converged.")
            print(f"{'='*40} END OF STEP {step} {'='*40}\n")
            break
        assignments = new_assignments
        # Update step
        new_centroids = []
        print("\n# Centroid update:")
        for cluster_idx in range(k):
            cluster_points = [
                data[i] for i, a in enumerate(assignments) if a == cluster_idx
            ]
            print_centroid_calculation(cluster_points, cluster_idx)
            if cluster_points:
                new_centroid = compute_centroid(cluster_points)
                print(f"    => New centroid: {new_centroid}")
            else:
                # If a cluster is empty, reinitialize its centroid randomly
                new_centroid = random.choice(data)
                print(
                    f"    => Cluster empty. Reinitialized centroid to random point: {new_centroid}"
                )
            new_centroids.append(new_centroid)
        print_centroids(step, new_centroids)
        print(f"{'='*40} END OF STEP {step} {'='*40}\n")
        centroids = new_centroids
    print("\n" + "=" * 40)
    print("Final clusters:")
    print_assignments(assignments, data, k)
    for cluster_idx in range(k):
        members = [
            f"P{i+1}{data[i]}" for i, a in enumerate(assignments) if a == cluster_idx
        ]
        print(
            f"  Cluster {cluster_idx+1}: {', '.join(members) if members else 'EMPTY'}"
        )
