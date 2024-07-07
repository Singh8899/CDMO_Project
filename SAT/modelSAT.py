import argparse
from z3 import *
from SAT.utils import *
from math import floor
import time
import SAT.modelBS as modelBS

time_limit = 300
def main(instance,configuration):
    assert configuration >= 0 and configuration < 3
    assert instance >= 0 and instance < 22
    if instance == 0:
        li = 1
        lu = 22
    else:
        li = instance
        lu = li+1
    
    if configuration == 0:
        cfgs = [1,2]
    else:
        cfgs = [configuration]
            
    for instance in range(li,lu):
        json_dict = {}
        for cfg in cfgs:
            if cfg == 1:
                start_time = time.time()
                n_couriers, n_items, courier_capacity,item_size, D = inputFile(instance)
                courier_capacity,item_size,D,w_bit,d_bit,min_dist,low_cour,max_dist = instance_format(n_couriers, n_items,courier_capacity,item_size, D)
                s = Optimize()
                p,maximum = set_const(s, n_couriers, n_items, courier_capacity,item_size, D,w_bit,d_bit,min_dist,low_cour,max_dist)
                    
                status = s.check()
                if  status == sat:
                    print("SAT")
                    model = s.model()
                    x = [[[int(is_true(model[p[k][i][j]]))
                            for j in range(n_items+1)] 
                            for i in range(n_items+1)] 
                            for k in range(n_couriers)]
                    maximum = model.evaluate(bool_vars_to_int(maximum))
                    solve_time = round(time.time() - start_time)
                    opt = True
                elif status == unknown:
                    maximum = -1
                    x = []
                    solve_time = 300
                    opt = False
                else:
                    print("ERROR")
                    maximum = -1
                    x = []
                    solve_time = 300
                    opt = False
                
                json_dict["standard"] = jsonizer(x,n_items+1,n_couriers,solve_time,opt,maximum.as_long() if type(maximum)!=int else maximum)
          
            if cfg == 2:
                json_dict["BS"] = modelBS.main(instance,cfg)
                
        format_and_store(instance,json_dict)

def set_const(s, n_couriers, n_items, courier_capacity,item_size, D,bit_weight,bit_dist,min_dist,low_cour,max_dist):
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
    s.set("timeout", time_limit*1000)
    # Upper/lower bounds on the objective function to minimize
    s.add(greater_eq(maximum,min_dist))
    s.add(greater_eq(max_dist,maximum))
    # 
    s.add([greater_eq(n_cities_bin,u[k][i]) for i in range(n_items) for k in range(n_couriers) ])
    # Upper/lower bounds on the distance travelled by each courier
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

    # Ensure that every city is reached by one and only one courier
    for j in range(n_items) :
        s.add(exactly_one_he([p[c][i][j] for i in range(origin) for c in range(n_couriers)] ,f"codlr_{j}"))

    # Ensure that every courier leaves the depot 
    for c in range(n_couriers) :
        s.add(exactly_one_he([p[c][n_items][j] for j in range(n_items)],f"depLev_{c}"))

    # Ensure that every courier reach again the depot
    for c in range(n_couriers) :
        s.add(exactly_one_he([p[c][i][n_items] for i in range(n_items)],f"arrLev_{c}"))
    
    # Ensure that each courier path it's connected
    for j in range(origin):
        for c in range(n_couriers):
            s.add(Or([p[c][i][j] for i in range(origin)]) == 
                  Or([p[c][j][i] for i in range(origin)]) )

    # To avoid round-routes internally
    # (not keeping in consideration the last column and last row, because that case it's allowed)
    # This constraint it's extra ( covered by the subroutes-elim contraints)
    # Just decreases computational time
    for c in range(n_couriers):
        for i in range(n_items):
            for j in range(n_items):
                if i != j:
                    s.add(Or(Not(p[c][i][j]),Not(p[c][j][i])))

    # computing dist array with distance travelled by every courier
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
    # Contraining objective function "maximum" to be greater than the distance 
    # travelled by each courier
    for d in cour_dist:
            s.add(greater_eq(maximum,d))
    opt = s.minimize(bool_vars_to_int(maximum))
    return p,maximum

def add_sb(s):
    pass


if __name__ == "__main__":
    main(instance,configuration)
