import sys
import os
from datetime import datetime
from clusters.hac import hac, single_linkage
from clusters.distance import manhattan_distance

# currently only supports clustering


class OutputLogger:
    def __init__(self, to_file=None):
        self.to_file = to_file
        self._stdout = sys.stdout

    def write(self, msg):
        self._stdout.write(msg)
        if self.to_file:
            self.to_file.write(msg)

    def flush(self):
        self._stdout.flush()
        if self.to_file:
            self.to_file.flush()


def main():
    data = [
        (1, 2),
        (4, 8),
        (3, 9),
        (7, 3),
        (4, 3),
        (2, 4),
        (5, 2),
        (3, 5),
        (2, 5),
        (6, 6),
    ]
    report = "--report" in sys.argv
    file_handle = None
    linkage_name = "single_linkage"
    distance_name = "manhattan_distance"
    if report:
        os.makedirs("report", exist_ok=True)
        filename = f"report/HAC_{linkage_name}_{distance_name}.txt"
        file_handle = open(filename, "w", encoding="utf-8")
        sys.stdout = OutputLogger(file_handle)
    try:
        hac(data, single_linkage, manhattan_distance)
    finally:
        if file_handle:
            file_handle.close()
            sys.stdout = sys.__stdout__
            print(f"Report written to {filename}")


if __name__ == "__main__":
    main()
