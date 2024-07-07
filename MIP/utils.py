import numpy as np
import os
from pulp import *

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

def pathFormatter(x,n_cities, n_couriers):
    sol=[]
    for c in range(n_couriers):
        solution_courier=[]
        num_assigned_to_courier =len([1 for i in range(n_cities) for j in range(n_cities) if value(x[i][j][c])>=0.9])
        for i in range(n_cities-1):
            if value(x[n_cities-1][i][c])>=0.9:
                solution_courier.append(i+1)
                city = i
                break
        for j in range(num_assigned_to_courier):
            for i in range(n_cities-1):
                if value(x[city][i][c])>0.9:
                    solution_courier.append(i+1)
                    city = i
        sol.append(solution_courier)
    return sol

def jsonizer(x,n_cities,n_couriers,time,optimal,obj):
    if obj < 0:
        return {"time": time, "optimal": optimal, "obj": "N/A", "sol": []}
    else:
        res = pathFormatter(x,n_cities, n_couriers)
        return {"time": time, "optimal": optimal, "obj": round(obj), "sol": res}


def format_and_store(instance,json_dict):
    # Get the directory of the current script
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)

    # Define the file name and path
    file_name = f"{instance}.json"
    
    # Save the dictionary to a JSON file
    with open(parent_directory+"/res/MIP/" + file_name, 'w') as file:
            json.dump(json_dict, file, indent=3)
    print(f"File saved at {parent_directory+"/res/MIP/" + file_name}")