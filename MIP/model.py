from pulp import *
import numpy as np
from utils import *


def main():
    instance = 11
    n_couriers, n_items, courier_capacity,item_size, D = inputFile(instance)
    n_cities=D.shape[0]-1
    origin=n_cities+1

    m_TSP = LpProblem("Minimize_m_TSP",LpMinimize)

    #We define a 3d matrix of variables x[i][j][c] means that courier c has used node (i,j) to leave city i and reach city j
    x = LpVariable.dicts("x", (range(1,origin+1), range(1,origin+1), range(1,n_couriers+1)), cat="Binary")
    u = LpVariable.dicts("u", (range(1,n_cities+1), range(1,n_couriers+1)), lowBound=0, upBound=origin-1, cat="Integer")
    n_obj = n_items // n_couriers + 1
    min_dist = max([(D[n_cities,i] + D[i,n_cities]) for i in range(0,n_cities)])
    max_dist = sum([D[i,i+1] for i in range(0,n_obj+1)]) + min_dist
    maximum = LpVariable("max_dist",lowBound=min_dist, upBound=max_dist, cat="Integer")

    #Ensure that we dont use useless arcs 
    m_TSP += lpSum(x[i][i][c] for i in range(1,origin+1) for c in range(1,n_couriers+1)) == 0

    # Ensure that every city is reached by one and only one courier
    for j in range(1,n_cities+1) :
        m_TSP += lpSum(x[i][j][c] for i in range(1,origin+1) for c in range(1,n_couriers+1)) == 1

    # Ensure that every courier leaves the depot 
    for c in range(1,n_couriers+1) :
        m_TSP += lpSum(x[origin][j][c] for j in range(1,origin)) == 1

    # Ensure that every courier reach again the depot
        for c in range(1,n_couriers+1) :
            m_TSP += lpSum(x[i][origin][c] for i in range(1,origin)) == 1

    # Ensure that each courier doesnt exceed its max capacity
    for c in range(1,n_couriers+1) :
        m_TSP += lpSum(x[i][j][c]*item_size[j-1] for i in range(1,origin+1) for j in range(1,n_cities+1)) <= courier_capacity[c-1]
        
    # Ensure that each courier path it's connected
    for j in range(1,origin+1):
        for c in range(1,n_couriers+1):
            m_TSP += lpSum(x[i][j][c] for i in range(1,origin+1) ) ==   lpSum(x[j][i][c] for i in range(1,origin+1) )

    for c in range(1,n_couriers+1):
        for i in range(1,n_cities+1):
            for j in range(1,n_cities+1):
                m_TSP += (x[i][j][c]   + x[j][i][c])  <=1

    for j in range(1,origin):
        m_TSP += lpSum(x[i][j][c] for i in range(1,origin+1) for c in range(1,n_couriers+1)) ==1

    for k in range(1,n_couriers+1):
        for i in range(1, n_cities+1):
            for j in range(1, n_cities+1):
                if i != j:
                    m_TSP += u[i][k] - u[j][k] + n_cities * x[i][j][k] <= n_cities - 1

    for c in range(1,n_couriers+1):
        m_TSP += lpSum( x[i][j][c] * D[i-1][j-1] for i in range(1,origin+1) for j in range(1,origin+1)) <= maximum
    m_TSP +=  maximum
    print(min_dist,max_dist)
    status = m_TSP.solve()
    
    format_and_store(x,origin,n_couriers,m_TSP.solutionTime,LpStatus[status],value(m_TSP.objective),instance)
    
if __name__ == "__main__":
    main()