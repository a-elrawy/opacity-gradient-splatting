import argparse
import json
import os
import re
from collections import defaultdict
from rich.console import Console
from rich.table import Table

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def get_dataset_for_dir(dir_name):
    if "mipnerf" in dir_name:
        return "MIPNERF360"
    else:
        return "LLFF"

def is_float(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def format_value(value):
    if isinstance(value, float) or (isinstance(value, str) and is_float(value)):
        return f"{float(value):.4f}"
    return str(value)

def find_metrics(data, metrics_dict):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (int, float)):
                metrics_dict[key] = value
            elif isinstance(value, dict):
                find_metrics(value, metrics_dict)

def merge_results(directories):
    all_results = defaultdict(lambda: defaultdict(dict))
    all_metrics = set()
    
    dataset_shapes = defaultdict(set)

    for dir_path in directories:
        if not os.path.isdir(dir_path):
            print(f"Directory not found: {dir_path}")
            continue

        dir_name = os.path.basename(dir_path)
        dataset_name = get_dataset_for_dir(dir_name)
        
        for shape_name in sorted(os.listdir(dir_path)):
            shape_path = os.path.join(dir_path, shape_name)
            if not os.path.isdir(shape_path):
                continue

            dataset_shapes[dataset_name].add(shape_name)
            
            results_file = os.path.join(shape_path, 'results.json')
            points_file = os.path.join(shape_path, 'total_points.txt')
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    try:
                        data = json.load(f)
                        metrics_found = {}
                        find_metrics(data, metrics_found)
                        for metric, value in metrics_found.items():
                            all_metrics.add(metric)
                            all_results[(dir_name, shape_name)][metric] = format_value(value)
                    except json.JSONDecodeError:
                        print(f"Could not parse {results_file}")

            if os.path.exists(points_file):
                with open(points_file, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    is_new_format = any(line.startswith("FPS:") for line in lines)

                    if is_new_format:
                        data_points = {}
                        i = 0
                        while i < len(lines) - 1:
                            points_line = lines[i]
                            fps_line = lines[i+1]
                            if ' ' in points_line and not points_line.startswith("FPS:") and fps_line.startswith("FPS:"):
                                try:
                                    points_parts = points_line.split()
                                    key = int(points_parts[0])
                                    total_points = points_parts[1]

                                    fps_parts = fps_line.split()
                                    fps = fps_parts[1]
                                    data_points[key] = {'total_points': total_points, 'fps': fps}
                                    
                                    i += 2
                                except (IndexError, ValueError):
                                    print(f"Could not parse lines in {points_file}: '{points_line}' and '{fps_line}'")
                                    i += 1
                            else:
                                i += 1
                        
                        if data_points:
                            final_key = max(data_points.keys())
                            final_points = data_points[final_key]['total_points']
                            final_fps = data_points[final_key]['fps']

                            all_metrics.add("total_points")
                            all_results[(dir_name, shape_name)]["total_points"] = format_value(final_points)
                            
                            all_metrics.add("FPS")
                            all_results[(dir_name, shape_name)]["FPS"] = format_value(final_fps)
                    elif lines:
                        try:
                            points = lines[-1].strip().split()[-1]
                            metric_name = "total_points"
                            all_metrics.add(metric_name)
                            all_results[(dir_name, shape_name)][metric_name] = format_value(points)
                        except IndexError:
                             print(f"Could not parse points from {points_file}")

    return all_results, sorted(list(all_metrics), key=natural_sort_key), dataset_shapes

def calculate_averages(all_results, shapes, dir_names, all_metrics):
    averages = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))

    for shape in shapes:
        for dir_name in dir_names:
            for metric in all_metrics:
                value_str = all_results.get((dir_name, shape), {}).get(metric)
                if value_str and is_float(value_str):
                    averages[dir_name][metric] += float(value_str)
                    counts[dir_name][metric] += 1
    
    for dir_name in dir_names:
        for metric in all_metrics:
            if counts[dir_name][metric] > 0:
                averages[dir_name][metric] /= counts[dir_name][metric]

    return averages

def print_results(all_results, all_metrics, dataset_shapes, directories):
    console = Console()
    dir_names = [os.path.basename(d) for d in directories]

    for dataset_name, shapes in sorted(dataset_shapes.items()):
        if not shapes:
            continue
            
        console.print(f"[bold green]Dataset: {dataset_name}[/bold green]")
        
        filtered_dir_names = [d for d in dir_names if get_dataset_for_dir(d) == dataset_name]
        
        averages = calculate_averages(all_results, shapes, filtered_dir_names, all_metrics)

        for metric in all_metrics:
            table = Table(title=f"Results for {metric}")
            table.add_column("Shape", justify="right", style="cyan", no_wrap=True)
            for dir_name in filtered_dir_names:
                table.add_column(dir_name, justify="right", style="magenta")

            for shape in sorted(list(shapes)):
                row = [shape]
                for dir_name in filtered_dir_names:
                    value = all_results.get((dir_name, shape), {}).get(metric, "N/A")
                    row.append(value)
                table.add_row(*row)
            
            avg_row = ["Average"]
            for dir_name in filtered_dir_names:
                avg_value = averages.get(dir_name, {}).get(metric)
                if avg_value is not None:
                    avg_row.append(format_value(avg_value))
                else:
                    avg_row.append("N/A")
            table.add_row(*avg_row, style="bold yellow")

            console.print(table)

        summary_table = Table(title=f"[bold blue]Average Metrics for {dataset_name}[/bold blue]")
        summary_table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
        for dir_name in filtered_dir_names:
            summary_table.add_column(dir_name, justify="right", style="magenta")

        for metric in all_metrics:
            row = [metric]
            for dir_name in filtered_dir_names:
                avg_value = averages.get(dir_name, {}).get(metric)
                if avg_value is not None:
                    row.append(format_value(avg_value))
                else:
                    row.append("N/A")
            summary_table.add_row(*row, style="bold yellow")
        
        console.print(summary_table)
        console.print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and display experiment results.")
    parser.add_argument('directories', nargs='+', help='List of directories to process (e.g., output_ours output_theirs)')
    args = parser.parse_args()

    results, metrics, dataset_shapes = merge_results(args.directories)
    if results:
        print_results(results, metrics, dataset_shapes, args.directories) 