from z3 import *
from itertools import combinations
import numpy as np
import time
from math import floor
import signal

import threading
delay_time = 3   # delay time in seconds
def watchdog():
  print('Taken too long time, going on next instance/model...')
  os._exit(1)

def exactly_one(variables):
    # At least one of the variables must be true
    at_least_one = Or(variables)

    # At most one of the variables must be true
    at_most_one = And(
        [Implies(variables[i], And([Not(variables[j]) for j in range(len(variables)) if j != i])) for i in
         range(len(variables))])

    return And(at_least_one, at_most_one)
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
n_couriers, n_cities, courier_capacity, item_size, D=inputFile(11)
dimension=(n_cities//n_couriers)+3


smt_solver=Optimize()


ass= [[Int(f"x_{i}_{j}") for j in range(n_couriers) ]for i in range(dimension)]

weights= [ Int(f"weight_{c}") for c in range(n_couriers)]

distances= [ Int(f"distance_{c}") for c in range(n_couriers)]

n_obj= [Int(f"n_obj_{c}") for c in range(n_couriers)]

max_dist=Int(f"max_dist")

# for courier in range(n_couriers):
#     smt_solver.add( n_obj[courier] == Sum([If(ass[i][courier] != n_cities+1, 1, 0) for i in range(dimension) ]))
#
lower_bound = max([(D[n_cities,i] + D[i,n_cities]) for i in range(0,n_cities)])


upper_bound = sum([D[i,i+1] for i in range(0,n_cities)]) + lower_bound
for i in range(dimension):
    for j in range(n_couriers):
        smt_solver.add(ass[i][j]>=1)
        smt_solver.add(ass[i][j]<=n_cities+1)

for courier in range(n_couriers):
    smt_solver.add(ass[1][courier]!=n_cities+1)

for courier in range(n_couriers):
    smt_solver.add(ass[0][courier] == n_cities+1)
    smt_solver.add(ass[-1][courier] == n_cities+1)

for courier in range(n_couriers):
    smt_solver.add(weights[courier] == Sum([If(ass[i][courier] == city, item_size[city-1], 0) for i in range(dimension) for city in range(n_cities+1)]))

# Now i set the limit weights given the capacity of a courier

for courier in range(n_couriers):
    smt_solver.add(weights[courier]<=courier_capacity[courier])

#Now i set the distance
for courier in range(n_couriers):
     smt_solver.add( distances[courier] == Sum([If(And(ass[i][courier] == city1,ass[i+1][courier]==city2),int(D[city1-1,city2-1]), 0) for i in range(dimension-1)
                                                for city1 in range(1,n_cities+2) for city2 in range(1,n_cities+2) ])   )
for courier in range(n_couriers):
    for i in range(1,dimension-1):
        smt_solver.add(Implies( ass[i][courier]==n_cities+1, ass[i+1][courier]==n_cities+1))

for courier in range(n_couriers):
    smt_solver.add(max_dist>=distances[courier])



for city in range(1,n_cities + 1):
            smt_solver.add(exactly_one([ass[i][j] == city for i in range(dimension) for j in range(n_couriers)]))

smt_solver.add(max_dist>=lower_bound)
smt_solver.add(max_dist<=upper_bound)






# if smt_solver.check() == sat:
#     print(smt_solver.model())
# else:
#     print("Failed to solve")



print("sto iniziando")
start_time = time.time()


smt_solver.minimize(max_dist)
result=smt_solver.check()

end_time = time.time()
print("ha finito")
time_taken = end_time - start_time

model=smt_solver.model()
print(model)
print(f"Minimum max distance found is ===>>> : ",{model[max_dist]})
print("Total time taken ====>: ",floor(time_taken),"s")



def solution_builder(model):
    solution=[]
    for courier in range(n_couriers):

        solution_courier=[]

        for j in range(dimension):

            if model.evaluate(ass[j][courier] != n_cities+1):

                solution_courier.append(model.evaluate(ass[j][courier]))

        solution.append(solution_courier)

    return solution


print(solution_builder(smt_solver.model()))
