"""Summarize evaluation metrics from independent training runs."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('metrics', nargs='+', type=Path)
    args = parser.parse_args()
    if len(args.metrics) < 2 or len(set(p.resolve() for p in args.metrics)) != len(args.metrics):
        parser.error('Provide at least two distinct metrics.json files')
    rows = [json.loads(path.read_text()) for path in args.metrics]
    result = {}
    for key in ('precision', 'recall', 'ap50', 'ap', 'f1'):
        values = [100 * float(row[key]) for row in rows]
        result[key] = dict(mean_percent=statistics.mean(values),
                           sample_std_percent=statistics.stdev(values), runs=len(values))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
