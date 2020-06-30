import numpy as np
from pathlib import Path


def writeClusters(f: Path, data: np.ndarray, labels: np.ndarray, separator=";"):
    with f.open("w") as fid:
        for dp, label in zip(data, labels):
            lineData = np.array2string(dp, separator=separator, max_line_width=0x80000000)[1:-1]
            fid.write(f"{label}{separator}{lineData}\n")


def loadCsvWithIntLabelsAsSeq(f: Path, labelOnFirst=True, separator=";"):
    dpSeq = []
    labelSeq = []

    with f.open() as fid:
        for l in fid.readlines():
            if l.isspace():
                continue
            parts = l.strip().replace("\n", "").split(separator)
            if labelOnFirst:
                try:
                    labelSeq.append(int(float(parts[0])))
                    dpSeq.append([float(x) for x in parts[1:]])
                except ValueError:
                    print(parts)
            else:
                labelSeq.append(int(parts[-1]))
                dpSeq.append([float(x) for x in parts[:-1]])

    return np.array(dpSeq), np.array(labelSeq)
