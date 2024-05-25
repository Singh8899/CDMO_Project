from pulp import *
import numpy as np
from utils import *
import time
from math import floor


def main():
    instance = 7
    n_couriers, n_items, courier_capacity,item_size, D = inputFile(instance)

    m_TSP,x,origin = set_const(n_couriers, n_items, courier_capacity,item_size, D)
    
    time_limit = 30
    solver = PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    m_TSP.solve(solver)
    solve_time = floor(m_TSP.solutionTime)
    opt = (time_limit > solve_time)
    format_and_store(x,origin,n_couriers,solve_time,opt,value(m_TSP.objective),instance)

def set_const(n_couriers, n_items, courier_capacity,item_size, D):
    m_TSP = LpProblem("Minimize_m_TSP",LpMinimize)

    n_cities=D.shape[0]-1
    origin=n_cities+1
    n_obj = n_items // n_couriers + 1
    min_dist = max([(D[n_cities,i] + D[i,n_cities]) for i in range(0,n_cities)])
    max_dist = sum([D[i,i+1] for i in range(0,n_obj+1)]) + min_dist
    #rows_max = [max(row) for row in D]  # List to store the sums of maximum values in each row
    low_cour = min([(D[n_cities,i] + D[i,n_cities]) for i in range(0,n_cities)])

    #We define a 3d matrix of variables x[i][j][c] means that courier c has used node (i,j) to leave city i and reach city j
    x = LpVariable.dicts("x", (range(1,origin+1), range(1,origin+1), range(1,n_couriers+1)), cat="Binary")
    u = LpVariable.dicts("u", (range(1,n_cities+1), range(1,n_couriers+1)), lowBound=0, upBound=origin-1, cat="Integer")
    maximum = LpVariable("max_dist",lowBound=min_dist, upBound=max_dist, cat="Integer")
    weigths = [LpVariable(name=f'weigth_{i}', lowBound=0, upBound=courier_capacity[i-1], cat="Integer")
                   for i in range(1,n_couriers+1)]
    cour_dist = [
            LpVariable(name=f'obj_dist{i}', cat="Integer", lowBound=low_cour, upBound=max_dist)
            for i in range(n_couriers)]
    
    m_TSP +=  maximum

    # Set the weight carried by each courier
    for k in range(1,n_couriers+1):
            m_TSP += weigths[k-1] == LpAffineExpression([
                (x[i][j][k], item_size[j-1])
                for i in range(1,n_items+1)
                for j in range(1,n_items+1)])
            
    # Ensure that we dont use useless arcs 
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
    # for c in range(1,n_couriers+1) :
    #     m_TSP += lpSum(x[i][j][c]*item_size[j-1] for i in range(1,origin+1) for j in range(1,n_cities+1)) <= courier_capacity[c-1]
        
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

    for c in range(n_couriers):
        m_TSP += lpSum( x[i][j][c+1] * D[i-1][j-1] for i in range(1,origin+1) for j in range(1,origin+1)) == cour_dist[c]
    
    for d in cour_dist:
            m_TSP += maximum >= d

    return m_TSP,x,origin


def set_constraints_model0(couriers, items, courier_size, item_size, distances):
        """
        :param data: data of the instance
        :return: model structures
        """
        print('setting constraints...')
        model = LpProblem("Minimize_m_TSP",LpMinimize)
        corr_dict = couriers

        courier_size = np.sort(courier_size)[::-1]

        all_travel = (True if max(item_size) <= courier_size[-1] else False)

        lower_bound = max([(distances[items,i] + distances[i,items]) for i in range(0,items)])

        row_sums = []  # List to store the sums of maximum values in each row

        for row in distances:
            max_value = max(row)  # Find the maximum value in the current row
            row_sums.append(max_value)  # Add the maximum value to the row_sums list

        upper_bound = sum([distances[i,i+1] for i in range(0,items)]) + lower_bound

        print('lower bound: ', lower_bound)
        print('upper_bound:', upper_bound)

        # General variables
        asg = [[[LpVariable(name=f'asg_{k}-{i}-{j}', cat=LpBinary)
                 for i in range(items)]
                for j in range(items)]
               for k in range(couriers)]

        weigths = [LpVariable(name=f'weigth_{i}', lowBound=0, upBound=courier_size[i], cat=LpInteger)
                   for i in range(couriers)]

        # Distances computation variables
        dist1 = [
            LpVariable(name=f'or_to_first_{i}', lowBound=0, upBound=np.max(distances), cat=LpInteger)
            for i in range(couriers)]

        dist2 = [LpVariable(name=f'last_to_or_{i}', lowBound=0, upBound=np.max(distances), cat=LpInteger)
                 for i in range(couriers)]

        dist2_bool = [[LpVariable(name=f'dist2_bool_{k}-{i}', cat=LpBinary)
                       for i in range(items)]
                      for k in range(couriers)]

        dist3 = [LpVariable(name=f'other_dist_{i}', lowBound=0, upBound=np.sum(distances), cat=LpInteger)
                 for i in range(couriers)]

        obj_dist = [
            LpVariable(name=f'obj_dist{i}', cat=LpInteger, lowBound=np.min(distances), upBound=np.sum(distances))
            for i in range(couriers)]

        maximum = LpVariable(name=f'_maximum', lowBound=lower_bound, upBound=upper_bound,
                             cat=LpInteger)

        model += maximum

        # Distances constraints

        # 1 ) Origin to first distribution center distance
        for k in range(couriers):
            model += dist1[k] == LpAffineExpression([
                (asg[k][0][i], distances[items][i])
                for i in range(items)])

        # 2 ) Last distribution center to Origin distance
        for k in range(couriers):
            model += lpSum(dist2_bool[k]) <= 1
            for i in range(items - 1):
                model += dist2_bool[k][i] - (lpSum(asg[k][i]) - lpSum(asg[k][i + 1])) == 0
            model += dist2_bool[k][-1] - lpSum(asg[k][-1]) == 0

        for k in range(couriers):
            model += LpAffineExpression([
                (linear_prod(model,dist2_bool[k][i], asg[k][i][j], 3, f'second_dist{k}-{j}-{i}'), distances[j][items])
                for i in range(items)
                for j in range(items)]) == dist2[k]

        # 3 ) Distances between all the distribution centers
        for k in range(couriers):
            model += LpAffineExpression([
                (linear_prod(model,asg[k][i][j], asg[k][i + 1][l], 3, f'items_dist{i}-{j}-{k}-{l}'), distances[j][l])
                for j in range(items)
                for l in range(items)
                for i in range(items - 1)]) == dist3[k]

        for k in range(couriers):
            model += obj_dist[k] == dist1[k] + dist2[k] + dist3[k]

        # for el in obj_dist:
        #     model += maximum >= el

        # Each item must be carried once
        # for i in range(items):
        #     model += lpSum(asg[k][j][i] for j in range(items) for k in range(couriers)) == 1

        # Each row must contain at most one value different from zero
        for k in range(couriers):
            for i in range(items):
                model += lpSum(asg[k][i]) <= 1

        # Force all the items to be in the first lines
        for k in range(couriers):
            for i in range(items - 1):
                model += lpSum(asg[k][i]) >= lpSum(asg[k][i + 1])

        # Weigths constraints
        for k in range(couriers):
            model += weigths[k] == LpAffineExpression([
                (asg[k][i][j], item_size[j])
                for i in range(items)
                for j in range(items)])
        # #return model,x,origin
        # return (asg, 
        #         weigths, 
        #         obj_dist, 
        #         couriers, 
        #         items, 
        #         distances,value(maximum)), corr_dict

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