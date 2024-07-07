from z3 import *
from itertools import combinations
import numpy as np
import time
from math import floor
import json
import argparse
import os

def at_least_one(bool_vars):
    return Or(bool_vars)


def at_most_one(bool_vars, name):
    if len(bool_vars) <= 4:
        return And(And([Not(And(pair[0], pair[1])) for pair in combinations(bool_vars, 2)]))
    y = Bool(f"y_{name}")
    return And(And(at_most_one(bool_vars[:3] + [y],name)), And(at_most_one(bool_vars[3:] + [Not(y)], name+"_")))


def exactly_one(bool_vars, name):
    return And(at_most_one(bool_vars, name), at_least_one(bool_vars))

def inputFile(num):
    # Change the working directory to the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # Instantiate variables from file
    if num < 10:
        instances_path = "instances/inst0" + str(num) + ".dat"  # inserire nome del file
    else:
        instances_path = "instances/inst" + str(num) + ".dat"  # inserire nome del file

    data_file = open(instances_path)
    lines = [line for line in data_file]
    data_file.close()
    n_couriers = int(lines[0].rstrip('\n'))
    n_items = int(lines[1].rstrip('\n'))
    max_load = list(map(int, lines[2].rstrip('\n').split()))
    size_item = list(map(int, lines[3].rstrip('\n').split()))
    for i in range(4, len(lines)):
        lines[i] = lines[i].rstrip('\n').split()

    dist = np.array([[lines[j][i] for i in range(len(lines[j]))] for j in range(4, len(lines))])
    dist = dist.astype(int)
    return n_couriers, n_items, max_load, size_item, dist





def build_standard_model(n_couriers, n_cities, courier_capacity, item_size, D,sym=False):
    dimension = (n_cities // n_couriers) + 3

    smt_solver = Optimize()

    ass = [[Int(f"x_{i}_{j}") for j in range(n_couriers)] for i in range(dimension)]

    weights = [Int(f"weight_{c}") for c in range(n_couriers)]

    distances = [Int(f"distance_{c}") for c in range(n_couriers)]

    n_obj = [Int(f"n_obj_{c}") for c in range(n_couriers)]

    max_dist = Int(f"max_dist")


    lower_bound = max([(D[n_cities, i] + D[i, n_cities]) for i in range(0, n_cities)])

    upper_bound = sum([D[i, i + 1] for i in range(0, n_cities)]) + lower_bound
    for i in range(dimension):
        for j in range(n_couriers):
            smt_solver.add(ass[i][j] >= 1)
            smt_solver.add(ass[i][j] <= n_cities + 1)

    for courier in range(n_couriers):
        smt_solver.add(ass[1][courier] != n_cities + 1)


    for courier in range(n_couriers):
        smt_solver.add(ass[0][courier] == n_cities + 1)
        smt_solver.add(ass[-1][courier] == n_cities + 1)

    for courier in range(n_couriers):
        smt_solver.add(weights[courier] == Sum(
            [If(ass[i][courier] == city, item_size[city - 1], 0) for i in range(dimension) for city in
             range(n_cities + 1)]))

    # Now i set the limit weights given the capacity of a courier

    for courier in range(n_couriers):
        smt_solver.add(weights[courier] <= courier_capacity[courier])

    # Now i set the distance
    for courier in range(n_couriers):
        smt_solver.add(distances[courier] == Sum(
            [If(And(ass[i][courier] == city1, ass[i + 1][courier] == city2), int(D[city1 - 1, city2 - 1]), 0) for i in
             range(dimension - 1)
             for city1 in range(1, n_cities + 2) for city2 in range(1, n_cities + 2)]))
    for courier in range(n_couriers):
        for i in range(1, dimension - 1):
            smt_solver.add(Implies(ass[i][courier] == n_cities + 1, ass[i + 1][courier] == n_cities + 1))

    for courier in range(n_couriers):
        smt_solver.add(max_dist >= distances[courier])

    for city in range(1, n_cities + 1):
        smt_solver.add(exactly_one([ass[i][j] == city for i in range(dimension) for j in range(n_couriers)],f"visit {city}"))

    if sym:
        for courier1, courier2 in zip(range(n_couriers), range(n_couriers)):
            smt_solver.add(Implies(
                And(weights[courier1] <= courier_capacity[courier2], weights[courier2] <= courier_capacity[courier1]),
                ass[1][courier1] < ass[1][courier2]))
            smt_solver.add(Implies(courier_capacity[courier1] > courier_capacity[courier2],
                                   weights[courier1] >= weights[courier2]))


    smt_solver.add(max_dist >= lower_bound)
    smt_solver.add(max_dist <= upper_bound)

    return smt_solver,ass,max_dist


def solve_smt(model_config, instance_number, time_to_solve):

    n_couriers, n_cities, courier_capacity, item_size, D = inputFile(instance_number)

    dimension = (n_cities // n_couriers) + 3

    if model_config == "standard":
        smt_solver,ass,max_dist = build_standard_model(n_couriers, n_cities, courier_capacity, item_size, D,False)

    if model_config == "standard_sym_heu":
        smt_solver,ass,max_dist = build_standard_model(n_couriers, n_cities, courier_capacity, item_size, D,True)



    smt_solver.set("timeout",time_to_solve * 1000)

    start_time = time.time()

    smt_solver.minimize(max_dist)

    result = smt_solver.check()

    end_time = time.time()

    time_taken = end_time - start_time


    sol = []
    for courier in range(n_couriers):


        solution_courier = []

        for j in range(dimension):

            if smt_solver.model().evaluate(ass[j][courier] != n_cities + 1):
                solution_courier.append( smt_solver.model().evaluate(ass[j][courier]).as_long())

        sol.append(solution_courier)
    if result == sat and time_taken < 300:
        optimal = True
    else:
        optimal = False

    return time_taken, optimal, smt_solver.model().evaluate(max_dist).as_long(), sol


def main(instance_number,model_config):
    n_instances = 21
    time_to_solve = 300
    configurations = ["standard", "standard_sym_heu"]
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)

    if instance_number == 0:
        for instance_number in range(1, n_instances + 1):
            json_final = {}

            for model_config in configurations:
                print("---Solving Instance Number --- : ", instance_number, " with ", model_config, " model---")

                json_instance = {}
                time = 0
                optimal = False
                obj = None
                sol = []

                time, optimal, obj, sol = solve_smt(model_config, instance_number, time_to_solve)

                print(f"Time taken ===> {floor(time)}s optimality===> {optimal} objective found ===> {obj}")
                print(f"Best route found is ====>{str(sol)} <======")
                json_instance["time"] = floor(time)
                json_instance["optimal"] = optimal
                json_instance["obj"] = int(obj)
                json_instance["sol"] = sol

                json_final[model_config] = json_instance

            with open(parent_directory + "/res/SMT/" + str(instance_number) + ".json", 'w') as file:
                json.dump(json_final, file, indent=3)

        pass
    if instance_number != 0:
        if model_config != 0:
            json_final = {}
            print("---Solving Instance Number --- : ", instance_number, " with ", model_config, " model---")

            json_instance = {}
            time = 0
            optimal = False
            obj = None
            sol = []

            time, optimal, obj, sol = solve_smt(model_config, instance_number, time_to_solve)
            print(f"Time taken ===> {floor(time)}s optimality===> {optimal} objective found ===> {obj}")
            print(f"Best route found is ====>{str(sol)} <======")
            time = 300 if time > 300 else time
            json_instance["time"] = floor(time)
            json_instance["optimal"] = optimal
            json_instance["obj"] = obj
            json_instance["sol"] = sol
            json_final[model_config] = json_instance

        with open(parent_directory + "/res/SMT/" + str(instance_number) + ".json", 'w') as file:
            json.dump(json_final, file, indent=3)
        if model_config ==0:
            json_final = {}

            for model_config in configurations:
                print("---Solving Instance Number --- : ", instance_number, " with ", model_config, " model---")

                json_instance = {}
                time = 0
                optimal = False
                obj = None
                sol = []

                time, optimal, obj, sol = solve_smt(model_config, instance_number, time_to_solve)

                print(f"Time taken ===> {floor(time)}s optimality===> {optimal} objective found ===> {obj}")
                print(f"Best route found is ====>{str(sol)} <======")
                json_instance["time"] = floor(time)
                json_instance["optimal"] = optimal
                json_instance["obj"] = obj
                json_instance["sol"] = sol

                json_final[model_config] = json_instance

            with open(parent_directory + "/res/SMT/" + str(instance_number) + ".json", 'w') as file:
                json.dump(json_final, file, indent=3)


if __name__ == "__main__":
    main(instance,configuration)