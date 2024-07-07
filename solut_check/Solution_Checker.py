import os
import re
import sys
import json

TIMEOUT = 300
# OPT[i] = Optimal value for instance i.
OPT = [None, 14, 226, 12, 220, 206]


def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON from file '{file_path}'.")
        return None


def main(args):
    '''
    check_solution.py <input folder> <results folder>
    '''
    # FIXME: Input folder contains the input files (in the format instXY.dat).
    #       The results folder contains the .json file of each approach.
    #       No other file should appear in these folders.
    errors = []
    warnings = []
    results_folder = args[2]
    for subfolder in os.listdir(results_folder):
        if subfolder.startswith('.'):
            # Skip hidden folders.
            continue
        folder = results_folder + subfolder
        print(f'\nChecking results in {folder} folder')
        for results_file in sorted(os.listdir(folder)):
            if results_file.startswith('.'):
                # Skip hidden folders.
                continue
            results = read_json_file(folder + '/' + results_file)
            print(f'\tChecking results for instance {results_file}')
            inst_number = re.search('\d+', results_file).group()
            if len(inst_number) == 1:
                inst_number = '0' + inst_number
            inst_path = args[1] + '/inst' + inst_number + '.dat'
            print(f'\tLoading input instance {inst_path}')
            with open(inst_path) as inst_file:
                i = 0
                for line in inst_file:
                    if i == 0:
                        n_couriers = int(line)
                    elif i == 1:
                        n_items = int(line)
                        dist_matrix = [None] * (n_items + 1)
                    elif i == 2:
                        capacity = [int(x) for x in line.split()]
                        assert len(capacity) == n_couriers
                    elif i == 3:
                        sizes = [int(x) for x in line.split()]
                        assert len(sizes) == n_items
                    else:
                        row = [int(x) for x in line.split()]
                        assert len(row) == n_items + 1
                        dist_matrix[i - 4] = [int(x) for x in row]
                    i += 1
            for i in range(len(dist_matrix)):
                assert dist_matrix[i][i] == 0
            for solver, result in results.items():
                print(f'\t\tChecking solver {solver}')
                header = f'Solver {solver}, instance {inst_number}'
                if result['time'] < 0 or result['time'] > TIMEOUT:
                    errors += [f"{header}: runtime unsound ({result['time']} sec.)"]
                if 'sol' not in result or not result['sol'] or result['sol'] == 'N/A':
                    continue
                max_dist = 0
                n_collected = sum(len(p) for p in result['sol'])
                if n_collected != n_items:
                    errors += [f"{header}: solution {result['sol']} collects {n_collected} instead of {n_items} items"]
                courier_id = 0
                for path in result['sol']:
                    dist = 0
                    path_size = 0
                    # Adjusting with origin point.
                    path = [n_items + 1] + path + [n_items + 1]
                    for i in range(1, len(path)):
                        curr_item = path[i] - 1
                        prev_item = path[i - 1] - 1
                        dist += dist_matrix[prev_item][curr_item]
                        if i < len(path) - 1:
                            path_size += sizes[curr_item]
                    if path_size > capacity[courier_id]:
                        errors += [
                            f"{header}: path {path} of courier {courier_id} has total size {path_size}, exceeding its capacity {capacity[courier_id]}"]
                    if dist > max_dist:
                        max_dist = dist
                        max_path = path
                        max_cour = courier_id
                    courier_id += 1
                if max_dist != result['obj']:
                    errors += [
                        f"{header}: objective value {result['obj']} inconsistent with max. distance {max_dist} of path {max_path}, courier {max_cour})"]
                i = int(inst_number)
                if i < 6:
                    if result['optimal']:
                        if result['obj'] != OPT[i]:
                            errors += [
                                f"{header}: claimed optimal value {result['obj']} inconsistent with actual optimal value {OPT[i]})"]
                    else:
                        warnings += [f"{header}: instance {inst_number} not solved to optimality"]
    print('\nCheck terminated.')
    if warnings:
        print('Warnings:')
        for w in warnings:
            print(f'\t{w}')
    if errors:
        print('Errors detected:')
        for e in errors:
            print(f'\t{e}')
    else:
        print('No errors detected!')


if __name__ == "__main__":
    main(sys.argv)