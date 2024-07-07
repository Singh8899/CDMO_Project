import matplotlib.pyplot as plt
import numpy as np
import os
from pulp import *
from z3 import *
from math import log
from itertools import combinations

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
    
    if obj < 0:
        return {"time": time, "optimal": optimal, "obj": "N/A", "sol": []}
    else:
        return {"time": time, "optimal": optimal, "obj": round(obj), "sol": pathFormatter(x,n_cities, n_couriers)}
    
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

def greater_eq(vec1, vec2):
    # Ensure the two vectors have the same length
    assert len(vec1) == len(vec2)
    # Initialize constraints list
    b1 = vec1[-1] 
    b2 = vec2[-1] 
    borrow = And(Not(b1),b2)
    # Subtract vec2 from vec1
    for i in range(2,len(vec1)):
        b1 = vec1[-i] 
        b2 = vec2[-i] 
        borrow = Or( And(borrow,Not(b1)), And(borrow,b2) , And(b2,Not(b1))  )
    b1 = vec1[0] 
    b2 = vec2[0] 
    borrow = Or( And(borrow,Not(b1)), And(borrow,b2) , And(b2,Not(b1))  )
    return Not(borrow)

def binary_prod(boolVar, num):
    return [And(boolVar,i) for i in num]

def at_least_one_np(bool_vars):
    return Or(bool_vars)

def at_most_one_np(bool_vars, name = ""):
    return And([Not(And(pair[0], pair[1])) for pair in combinations(bool_vars, 2)])

def at_least_one_he(bool_vars):
    return at_least_one_np(bool_vars)

def at_most_one_he(bool_vars, name):
    if len(bool_vars) <= 4:
        return And(at_most_one_np(bool_vars))
    y = Bool(f"y_{name}")
    return And(And(at_most_one_np(bool_vars[:3] + [y])), And(at_most_one_he(bool_vars[3:] + [Not(y)], name+"_")))

def exactly_one_he(bool_vars, name):
    return And(at_most_one_he(bool_vars, name), at_least_one_he(bool_vars))

def bool_vars_to_int(bool_vars):
    return Sum([If(b, 2**i, 0) for i, b in enumerate(list(reversed(bool_vars)))])

def bin_sum(vec1, vec2):
    # Ensure the two vectors have the same length
    assert len(vec1) == len(vec2)
    # Initialize constraints list
    res = []
    b1 = vec1[-1] 
    b2 = vec2[-1] 
    res.append(Xor(b1,b2)) 
    borrow = And(b1,b2)
    # Add vec2 from vec1
    for i in range(2,len(vec1)):
        b1 = vec1[-i]
        b2 = vec2[-i]
        res.append(Or( And( Not(borrow),Xor(b1,b2)), And(borrow,b1==b2) ) ) 
        borrow = Or( And(borrow,b1), And(borrow,b2) , And(b2,b1)  )
    b1 = vec1[0] 
    b2 = vec2[0] 
    res.append(Or( And( Not(borrow),Xor(b1,b2)), And(borrow,b1==b2) ) ) 
    return list(reversed(res))

def binary_sum(bin_numbers):
    sum = bin_numbers[0]
    for i in bin_numbers[1:]:
        sum = bin_sum(sum,i)
    return sum

def lesseq(a, b):
    constraints = []
    constraints.append(Or(Not(a[0]),b[0]))
    for i in range(1,len(a)):
        constraints.append(Implies(And([a[k] == b[k] for k in range(i)]), Or(Not(a[i]),b[i])))
    return And(constraints)

def toInteger(bool_list):
    binary_string = ''.join('1' if b else '0' for b in bool_list)
    return int(binary_string, 2)

def toBinary(num, length = None, dtype = int):
    num_bin = bin(num).split("b")[-1]
    if length:
        num_bin = "0"*(length - len(num_bin)) + num_bin
    num_bin = [dtype(int(s)) for s in num_bin]
    return num_bin

def maxim(vec, maxi, name= ""):
    if len(vec) == 1:
        return equals(vec[0], maxi)
    elif len(vec) == 2:
        constr1 = Implies(lesseq(vec[0], vec[1]), equals(vec[1], maxi))
        constr2 = Implies(Not(lesseq(vec[0], vec[1])), equals(vec[0], maxi))
        return And(constr1, constr2)
  
    par = [[Bool(f"maxpar_{name}_{i}_{b}") for b in range(len(maxi))] for i in range(len(vec)-2)]
    constr = []

    constr.append(Implies(lesseq(vec[0], vec[1]), equals(vec[1], par[0])))
    constr.append(Implies(Not(lesseq(vec[0], vec[1])), equals(vec[0], par[0])))

    for i in range(1, len(vec)-2):
        constr.append(Implies(lesseq(vec[i+1], par[i-1]), equals(par[i-1], par[i])))
        constr.append(Implies(Not(lesseq(vec[i+1], par[i-1])), equals(vec[i+1], par[i])))

    constr.append(Implies(lesseq(vec[-1], par[-1]), equals(par[-1], maxi)))
    constr.append(Implies(Not(lesseq(vec[-1], par[-1])), equals(vec[-1], maxi)))
  
    return And(constr)

def equals(a, b):
    return And([a[i] == b[i] for i in range(len(a))])

def greater_eq(vec1, vec2):
    # Ensure the two vectors have the same length
    assert len(vec1) == len(vec2)
    # Initialize constraints list
    b1 = vec1[-1] 
    b2 = vec2[-1] 
    borrow = And(Not(b1),b2)
    # Subtract vec2 from vec1
    for i in range(2,len(vec1)):
        b1 = vec1[-i] 
        b2 = vec2[-i] 
        borrow = Or( And(borrow,Not(b1)), And(borrow,b2) , And(b2,Not(b1))  )
    b1 = vec1[0] 
    b2 = vec2[0] 
    borrow = Or( And(borrow,Not(b1)), And(borrow,b2) , And(b2,Not(b1))  )
    return Not(borrow)

def binary_prod(boolVar, num):
    return [And(boolVar,i) for i in num]

def at_least_one_np(bool_vars):
    return Or(bool_vars)

def at_most_one_np(bool_vars, name = ""):
    return And([Not(And(pair[0], pair[1])) for pair in combinations(bool_vars, 2)])

def at_least_one_he(bool_vars):
    return at_least_one_np(bool_vars)

def at_most_one_he(bool_vars, name):
    if len(bool_vars) <= 4:
        return And(at_most_one_np(bool_vars))
    y = Bool(f"y_{name}")
    return And(And(at_most_one_np(bool_vars[:3] + [y])), And(at_most_one_he(bool_vars[3:] + [Not(y)], name+"_")))

def exactly_one_he(bool_vars, name):
    return And(at_most_one_he(bool_vars, name), at_least_one_he(bool_vars))

def bool_vars_to_int(bool_vars):
    return Sum([If(b, 2**i, 0) for i, b in enumerate(list(reversed(bool_vars)))])

def bin_sum(vec1, vec2):
    # Ensure the two vectors have the same length
    assert len(vec1) == len(vec2)
    # Initialize constraints list
    res = []
    b1 = vec1[-1] 
    b2 = vec2[-1] 
    res.append(Xor(b1,b2)) 
    borrow = And(b1,b2)
    # Add vec2 from vec1
    for i in range(2,len(vec1)):
        b1 = vec1[-i]
        b2 = vec2[-i]
        res.append(Or( And( Not(borrow),Xor(b1,b2)), And(borrow,b1==b2) ) ) 
        borrow = Or( And(borrow,b1), And(borrow,b2) , And(b2,b1)  )
    b1 = vec1[0] 
    b2 = vec2[0] 
    res.append(Or( And( Not(borrow),Xor(b1,b2)), And(borrow,b1==b2) ) ) 
    return list(reversed(res))

def binary_sum(bin_numbers):
    sum = bin_numbers[0]
    for i in bin_numbers[1:]:
        sum = bin_sum(sum,i)
    return sum