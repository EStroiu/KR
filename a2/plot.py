import argparse
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot data.")
    grouping = ap.add_mutually_exclusive_group(required=True)
    grouping.add_argument("--runs", action="store_true") #Plot the same heuristics in different runs together
    grouping.add_argument("--heuristics", action="store_true") #Plot the different heuristics within a run together
    data_type = ap.add_mutually_exclusive_group(required=True)
    data_type.add_argument("--baseline", action="store_true")
    data_type.add_argument("--clues", action="store_true")
    ap.add_argument("path")
    return ap.parse_args()

def get_data(path: str, dir: str, data: dict, extra: str = "") -> dict:
    raw = pd.read_csv(path)
    size = "N9" if "n9" in dir else "N25"
    if "d20" not in dir:
        title = "random" if "random" in dir else "unsat" if "unsat" in dir else "sat"
        chance = "d10" if "d10" in dir else "d30" if "d30" in dir else "d50"
        if extra != "":
            data[f"{size} {title} {chance} {extra}"] = raw["wall_time_s"]
        else:
            data[f"{size} {title} {chance}"] = raw["wall_time_s"]
    else:
        title = "random" if "random" in dir else "MRV"
        if extra != "":
            data[f"{size} {title} {extra}"] = raw["wall_time_s"]
        else:
            data[f"{size} {title}"] = raw["wall_time_s"]
    return data

def get_path(path: pathlib.Path, suffix: str = "") -> str:
    return path / f"solver-solver{suffix}" / "metrics.csv"

def plot(original_data: dict, file: pathlib.Path, size: str):
    new_data = {}
    for key in original_data.keys():
        if size in key:
            new_data[key] = original_data[key]
    if len(new_data.keys()) == 0:
        return
    raw = list(sorted(zip(new_data.keys(), new_data.values()), key=lambda x: x[0]))
    data = list(map(lambda x: x[1], raw))
    labels = list(map(lambda x: x[0], raw))

    plt.boxplot(data)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.savefig(file)
    plt.clf()

def plot_both(original_data: dict, file_name: str):
    plots = pathlib.Path.cwd() / "plots"
    plot(original_data, plots / f"N9_{file_name}", "N9")
    plot(original_data, plots / f"N25_{file_name}", "N25")

def process_runs(dirs: list[pathlib.Path], prefix: str):
    random = {}
    DLIS = {}
    JW = {}
    MOM = {}
    for dir in dirs:
        random = get_data(get_path(dir), dir.name, random)
        DLIS = get_data(get_path(dir, "_DLIS"), dir.name, DLIS)
        JW = get_data(get_path(dir, "_JW"), dir.name, JW)
        MOM = get_data(get_path(dir, "_MOM"), dir.name, MOM)
    plots = pathlib.Path.cwd() / "plots"
    if not plots.exists():
        os.mkdir(plots)
    plot_both(random, f"{prefix}_runs_random.png")
    plot_both(DLIS, f"{prefix}_runs_DLIS.png")
    plot_both(JW, f"{prefix}_runs_JW.png")
    plot_both(MOM, f"{prefix}_runs_MOM.png")

def runs(path: str, baseline: bool, clues: bool):
    folder = pathlib.Path(path)
    dirs = [dir for dir in folder.iterdir() if dir.is_dir()]
    if baseline:
        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
        current_dirs = [dir for dir in dirs if "d20" in dir.name]
        process_runs(current_dirs, "baseline")
        
    if clues:
        plt.rcParams["figure.figsize"] = (13,4.8)
        current_dirs = [dir for dir in dirs if "clues" in dir.name]
        process_runs(current_dirs, "clues")

def process_heuristic(dirs: list[pathlib.Path]):
    for dir in dirs:
        data = {}
        data = get_data(get_path(dir), dir.name, data, "random")
        data = get_data(get_path(dir, "_DLIS"), dir.name, data, "DLIS")
        data = get_data(get_path(dir, "_JW"), dir.name, data, "JW")
        data = get_data(get_path(dir, "_MOM"), dir.name, data, "MOM")
        type = "random" if "random" in dir.name else "unsat" if "unsat" in dir.name else "sat"
        if "clues" in dir.name:
            chance = "d10" if "d10" in dir.name else "d30" if "d30" in dir.name else "d50"
            if "random" in dir.name:
                plot_both(data, f"{type}_{chance}_random_clues_heuristics.png")
            else:
                plot_both(data, f"{type}_{chance}_clues_heuristics.png")
        else:
            if "random" in dir.name:
                plot_both(data, f"random_baseline_heuristics.png")
            else:
                plot_both(data, f"baseline_heuristics.png")


def heuristics(path: str, baseline: bool, clues: bool):
    folder = pathlib.Path(path)
    dirs = [dir for dir in folder.iterdir() if dir.is_dir()]
    if baseline:
        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
        current_dirs = [dir for dir in dirs if "d20" in dir.name]
        process_heuristic(current_dirs)

    if clues:
        plt.rcParams["figure.figsize"] = (13,4.8)
        current_dirs = [dir for dir in dirs if "clues" in dir.name]
        process_heuristic(current_dirs)

def main():
    args = parse_args()
    if args.runs:
        runs(args.path, args.baseline, args.clues)
    if args.heuristics:
        heuristics(args.path, args.baseline, args.clues)

if __name__ == "__main__":
    main()