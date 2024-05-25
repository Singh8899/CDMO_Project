import matplotlib.pyplot as plt
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
        print("ok")
        plt.plot([path[0]-1,path[1]-1],[D[-1,path[0]-1],D[-1,path[1]-1]])
    plt.grid(True)
    plt.show()    

def pathFormatter(x,origin, n_couriers):
    sol=[]
    for c in range(1,n_couriers+1):
        solution_courier=[]
        num_assigned_to_courier =len([1 for i in range(1,origin+1) for j in range(1,origin+1) if value(x[i][j][c])==1])
        for i in range(1,origin):
            if value(x[origin][i][c])==1:
                solution_courier.append(i)
                city = i
                break
        for j in range(num_assigned_to_courier):
            for i in range(1,origin):
                if value(x[city][i][c])==1:
                    solution_courier.append(i)
                    city = i
        sol.append(solution_courier)
    return sol

def format_and_store(x,origin,n_couriers,time,optimal,obj,instance):
    res = pathFormatter(x,origin, n_couriers)
    # Get the directory of the current script
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)

    script_dir = os.path.dirname(os.path.abspath(__file__))


    json_dict = {}
    key_dict="solvername"
    json_dict[key_dict] = {"time": time, "optimal": optimal, "obj": int(obj), "sol": str(res)}
    # Define the file name and path
    file_name = f"{instance}.json"
    print(json_dict)
    # Save the dictionary to a JSON file
    with open(parent_directory+"/res/MIP/" + file_name, 'w') as file:
            json.dump(json_dict, file, indent=3)
    
    print(f"File saved at {parent_directory+"/res/MIP/" + file_name}")