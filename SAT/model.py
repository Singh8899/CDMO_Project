from z3 import *
from utils import *
from itertools import combinations
from math import floor
import time

time_limit = 300

def main():
    for instance in range(2,3):

        json_dict = {}
        n_couriers, n_items, courier_capacity,item_size, D = inputFile(instance)
        courier_capacity,item_size,D,load_bit,dist_bit,min_dist,low_cour,max_dist = instance_format(n_couriers, n_items,courier_capacity,item_size, D)
        start_time = time.time()
        x,maximum = set_const(n_couriers, n_items, courier_capacity,item_size, D,load_bit,dist_bit,min_dist,low_cour,max_dist)
        if x != None:
            solve_time = round(time.time() - start_time)
            opt = (time_limit > solve_time)
            json_dict["Z3"] = jsonizer(x,n_items+1,n_couriers,solve_time,opt,maximum.as_long())
            format_and_store(instance,json_dict)

def set_const(n_couriers, n_items, courier_capacity,item_size, D,bit_weight,bit_dist,min_dist,low_cour,max_dist):
    # Create all the variables
    origin = n_items + 1
    p = [[[Bool(f"x_{k}_{i}_{j}") 
            for j in range(origin)] 
                for i in range(origin)] 
                    for k in range(n_couriers)]
    maximum =   [Bool(f"max_dist_{i}") 
                    for i in range(bit_dist)]
    weights  =  [[Bool(f"weigh_cour_{i}_{j}") 
                    for i in range(bit_weight)] 
                        for j in range(n_couriers)]
    cour_dist = [[Bool(f"cour_dist_{i}_{j}") 
                    for i in range(bit_dist)]
                        for j in range(n_couriers)]
    u = [[[Bool(f"ul_{i}_{j}_{k}") 
            for i in range(bit_length(n_items*2))]
                for j in range(n_items)]
                    for k in range(n_couriers)]    
    n_cities_bin = int_to_bool(n_items,bit_length(n_items*2))
    # Create the solver instance

    s = Optimize()
    s.set("timeout", time_limit*1000,)
    s.add(greater_eq(maximum,min_dist))
    s.add(greater_eq(max_dist,maximum))
    s.add([greater_eq(n_cities_bin,u[k][i]) for i in range(n_items) for k in range(n_couriers) ])
    for i in cour_dist:
        s.add(greater_eq(max_dist,i))
        s.add(greater_eq(i,low_cour))

    # Set the weight carried by each courier
    for c in range(n_couriers):
        # Compute the weight sum for courier k
        weight_sum = binary_sum([binary_prod(p[c][i][j] ,item_size[j]) 
                                    for i in range(origin) 
                                    for j in range(n_items)])
        s.add([weights[c][i] == weight_sum[i] for i in range(bit_weight)])
        s.add(greater_eq(courier_capacity[c],weight_sum))
        

    # Ensure that we dont use useless arcs 
    for i in range(origin):
        for c in range(n_couriers):
            s.add(p[c][i][i] == False)

    # # Ensure that every city is reached by one and only one courier
    for j in range(n_items) :
        s.add(exactly_one_he([p[c][i][j] for i in range(origin) for c in range(n_couriers)] ,f"codlr_{j}"))

    # Ensure that every courier leaves the depot 
    for c in range(n_couriers) :
        s.add(exactly_one_he([p[c][n_items][j] for j in range(n_items)],f"depLev_{c}"))

    # # Ensure that every courier reach again the depot
    for c in range(n_couriers) :
        s.add(exactly_one_he([p[c][i][n_items] for i in range(n_items)],f"arrLev_{c}"))
    
    # # Ensure that each courier path it's connected
    for j in range(origin):
        for c in range(n_couriers):
            s.add(Or([p[c][i][j] for i in range(origin)]) == 
                  Or([p[c][j][i] for i in range(origin)]) )

    for c in range(n_couriers):
        for i in range(n_items):
            for j in range(n_items):
                if i != j:
                    s.add(Or(Not(p[c][i][j]),Not(p[c][j][i])))

    for j in range(n_couriers):
        s.add(exactly_one_he([p[c][i][j] for i in range(origin) for c in range(n_couriers)],f"rand_{j}"))

    for c in range(n_couriers):
        dist = binary_sum([binary_prod(p[c][i][j], D[i][j]) 
                            for i in range(origin) 
                            for j in range(origin)])
        for i in range(bit_dist):
            s.add(dist[i] == cour_dist[c][i])
    # To eliminate subroutes
    for k in range(n_couriers):
        for i in range(n_items):
            for j in range(n_items):
                if i != j:
                    s.add(greater_eq
                            (
                                binary_sum([n_cities_bin,
                                            u[k][j]
                                            ]),
                                binary_sum([u[k][i],
                                            binary_prod(
                                                            p[k][i][j],
                                                            n_cities_bin
                                                        ),
                                            int_to_bool(1,bit_length(n_items*2))
                                            ])
                            )
                         )
                    # m_TSP += u[i][k] +1  + n_cities * x[i][j][k] <= n_cities  + u[j][k]
    
    for d in cour_dist:
            s.add(greater_eq(maximum,d))
    opt = s.minimize(bool_vars_to_int(maximum))

    # Check for satisfiability
    status = s.check()

    if status == unknown or status == sat:
        model = s.model()
        print(maximum,type(maximum))
        print(model.evaluate(bool_vars_to_int(maximum)),type(model.evaluate(bool_vars_to_int(maximum))))
        x = [[[int(is_true(model[p[k][i][j]]))
                for j in range(origin)] 
                for i in range(origin)] 
                    for k in range(n_couriers)]
        return x,model.evaluate(bool_vars_to_int(maximum))
    else:
        return None,None

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
if __name__ == "__main__":
    main()