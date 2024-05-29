import matplotlib.pyplot as plt
import numpy as np
import os
from pulp import *
from z3 import *
from math import log

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

def cityPlot(D):
    plt.figure(figsize=(5, 5))
    last = D.shape[0]-1
    for c in range(last):
        plt.scatter(c+1, D[-1,c],  c='r')
    plt.scatter(D[-1,-1], D[-1,-1], c='b')
    plt.grid(True)
    plt.show()

def routePlot(paths,D):
    cityPlot(D)
    for path in paths:
        plt.plot([path[0]-1,path[1]-1],[D[-1,path[0]-1],D[-1,path[1]-1]])
    plt.grid(True)
    plt.show()    

def pathFormatter(x,n_cities, n_couriers):
    sol=[]

    for c in range(n_couriers):
        solution_courier=[]
        num_assigned_to_courier =len([1 for i in range(n_cities) for j in range(n_cities) if value(x[c][i][j])==1])
        for i in range(n_cities-1):
            if value(x[c][n_cities-1][i])==1:
                solution_courier.append(i+1)
                city = i
                break
        for j in range(num_assigned_to_courier):
            for i in range(n_cities-1):
                if value(x[c][city][i])==1:
                    solution_courier.append(i+1)
                    city = i
        sol.append(solution_courier)
    return sol

def jsonizer(x,n_cities,n_couriers,time,optimal,obj):
    res = pathFormatter(x,n_cities, n_couriers)
    return {"time": time, "optimal": optimal, "obj": round(obj), "sol": str(res)}


def format_and_store(instance,json_dict):
    # Get the directory of the current script
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)

    # Define the file name and path
    file_name = f"{instance}.json"
    
    # Save the dictionary to a JSON file
    with open(parent_directory+"/res/SAT/" + file_name, 'w') as file:
            json.dump(json_dict, file, indent=3)
    
    print(f"File saved at {parent_directory+"/res/SAT/" + file_name}")

def instance_format(n_couriers, n_items,courier_cap,item_size, D):
    load_bit = bit_length(max(sum(item_size),max(courier_cap)))
    dist_bit = bit_length(sum( [max(i) for i in D] ))
    
    courier_size_bin = [int_to_bool(courier_cap[i], load_bit) 
                            for i in range(n_couriers)]
    item_size_bin =  [int_to_bool(item_size[i], load_bit) 
                        for i in range(n_items)]
    distances_bin = [[int_to_bool(D[i][j], dist_bit) 
                        for j in range(n_items+1)]
                            for i in range(n_items+1)]
    
    n_obj = n_items // n_couriers + 1
    min_dist = max([(D[n_items,i] + D[i,n_items]) for i in range(n_items)])
    low_cour = min([(D[n_items,i] + D[i,n_items]) for i in range(n_items)])
    max_dist = min_dist + 2*round(log(sum([D[i,i+1] for i in range(n_obj+1)]))) + low_cour
    return courier_size_bin,item_size_bin,distances_bin,load_bit,dist_bit,int_to_bool(min_dist,dist_bit),int_to_bool(low_cour,dist_bit),int_to_bool(max_dist,dist_bit)

def int_to_bool(n, k):
    binary_str = bin(n)[2:]
    # Padding with zeros
    padded_binary_str = binary_str.zfill(k)
    # Convert each character in the padded binary string to a boolean
    return [char == '1' for char in padded_binary_str]

def bit_length(n):
    if n < 0:
        raise ValueError("Input should be a non-negative integer")
    # Handle the special case where n is 0
    if n == 0:
        return 1
    return int(n).bit_length()
