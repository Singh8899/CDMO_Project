from pulp import *
import numpy as np
from utils import *
import time
from math import floor
import gurobipy as gp


def main():
    time_limit = 60
    solvers = {
                "CBC":PULP_CBC_CMD(msg=False, timeLimit=time_limit),
                "HiGHS":getSolver('HiGHS',msg=False, timeLimit=time_limit)
              }
    for instance in range(7,8):
        json_dict = {}
        for solver in solvers:
            n_couriers, n_items, courier_capacity,item_size, D = inputFile(instance)

            m_TSP,x = set_const(n_couriers, n_items, courier_capacity,item_size, D)
            m_TSP.solve(solvers[solver])
            # print([(i, j, c) for i in range(n_items+1) for j in range(n_items+1) for c in range(n_couriers) if pulp.value(x[i][j][c]) == 1])
            solve_time = floor(m_TSP.solutionTime)
            opt = (time_limit > solve_time)
            json_dict[solver] = jsonizer(x,n_items+1,n_couriers,solve_time,opt,value(m_TSP.objective))
        format_and_store(instance,json_dict)

def set_const(n_couriers, n_items, courier_capacity,item_size, D):
    m_TSP = LpProblem("Minimize_m_TSP",LpMinimize)

    n_cities=D.shape[0]-1
    origin=n_cities+1
    n_obj = n_items // n_couriers + 1
    min_dist = max([(D[n_cities,i] + D[i,n_cities]) for i in range(0,n_cities)])
    max_dist = 200 #sum([D[i,i+1] for i in range(0,n_obj+1)]) + min_dist
    low_cour = min([(D[n_cities,i] + D[i,n_cities]) for i in range(0,n_cities)])
    print(low_cour,min_dist,max_dist)
    #We define a 3d matrix of variables x[i][j][c] means that courier c has used node (i,j) to leave city i and reach city j
    x = LpVariable.dicts("x", (range(origin), range(origin), range(n_couriers)), cat="Binary")
    u = LpVariable.dicts("u", (range(n_cities), range(n_couriers)), lowBound=0, upBound=origin-1, cat="Integer")
    maximum = LpVariable("max_dist",lowBound=min_dist, upBound=max_dist, cat="Integer")
    weigths = [LpVariable(name=f'weigth_{i}', lowBound=0, upBound=courier_capacity[i], cat="Integer")
                   for i in range(n_couriers)]
    cour_dist = [
            LpVariable(name=f'obj_dist{i}', cat="Integer", lowBound=low_cour, upBound=max_dist)
            for i in range(n_couriers)]
    
    m_TSP += maximum

    # Set the weight carried by each courier
    for k in range(n_couriers):
            m_TSP += weigths[k] == LpAffineExpression([
                (x[i][j][k], item_size[j])
                for i in range(n_items+1)
                for j in range(n_items)])
            
    # Ensure that we dont use useless arcs 
    m_TSP += lpSum(x[i][i][c] for i in range(origin) for c in range(n_couriers)) == 0

    # Ensure that every city is reached by one and only one courier
    for j in range(n_cities) :
        m_TSP += lpSum(x[i][j][c] for i in range(origin) for c in range(n_couriers)) == 1

    # Ensure that every courier leaves the depot 
    for c in range(n_couriers) :
        m_TSP += lpSum(x[n_cities][j][c] for j in range(n_cities)) == 1

    # Ensure that every courier reach again the depot
        for c in range(n_couriers) :
            m_TSP += lpSum(x[i][n_cities][c] for i in range(n_cities)) == 1

    # Ensure that each courier doesnt exceed its max capacity
    # for c in range(n_couriers) :
    #     m_TSP += lpSum(x[i][j][c]*item_size[j] for i in range(origin) for j in range(n_cities)) <= courier_capacity[c]

    # Ensure that each courier path it's connected
    for j in range(origin):
        for c in range(n_couriers):
            m_TSP += lpSum(x[i][j][c] for i in range(origin) ) ==   lpSum(x[j][i][c] for i in range(origin) )

    for c in range(n_couriers):
        for i in range(n_cities):
            for j in range(n_cities):
                m_TSP += (x[i][j][c]   + x[j][i][c])  <=1

    for j in range(n_couriers):
        m_TSP += lpSum(x[i][j][c] for i in range(origin) for c in range(n_couriers)) ==1

    for k in range(n_couriers):
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    m_TSP += u[i][k] - u[j][k] + n_cities * x[i][j][k] <= n_cities - 1

    for c in range(n_couriers):
        m_TSP += lpSum( x[i][j][c] * D[i][j] for i in range(origin) for j in range(origin)) == cour_dist[c]
    
    for d in cour_dist:
            m_TSP += maximum >= d

    return m_TSP,x



def linear_prod(model,binary_var, countinuos_var, ub, name):
    """
    res = binary_var * countinuos_var
    :param binary_var: binary variable
    :param countinuos_var: countinuos variable
    :param ub: upper bound of the countinuos variable
    :param name: name of the product
    :return: the result of the product
    """
    res = LpVariable(cat=LpInteger, name=name)
    model += ub * binary_var >= res
    model += countinuos_var >= res
    model += countinuos_var - (1 - binary_var) * ub <= res
    model += res >= 0
    return res
def If(model,a, b, M, name):
    """
    if(a>b)
    :param a: left side condition
    :param b: right side condition
    :param M: big M
    :param name: name of the if
    :return: binary var: 1 if a>b 0 otherwise
    """
    delta = LpVariable(cat=LpInteger, name=name)
    model += a >= b + (0.001) - M * (1 - delta)
    model += a <= b + M * delta
    return delta

def And(model,a, b, name):
    """
    And(a,b)
    :param a: first parameter of And condition
    :param b: second parameter of And condition
    :param name: name of the And
    :return: 1 if a and b is true, false otherwise
    """
    delta = LpVariable(cat=LpInteger, name=name)
    model += delta <= a
    model += delta >= a + b - 1
    model += delta >= 0
    model += delta <= b
    return delta

if __name__ == "__main__":
    main()