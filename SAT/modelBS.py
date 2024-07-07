from z3 import *
from SAT.utils_BS_model import *
from itertools import combinations
from math import floor
import time

time_limit = 300
s = Solver()
def main(instance,cfg):
    global s
    s = Solver()
    s.set('timeout', time_limit * 1000)
    n_couriers, n_items, courier_capacity,item_size, D_int = inputFile(instance)
    courier_capacity,item_size,D,load_bit,dist_bit,min_dist,low_cour,max_dist = instance_format(n_couriers, n_items,courier_capacity,item_size, D_int)
    maximum, x, cour_dist, weights = set_const(n_couriers, n_items, courier_capacity,item_size, D,load_bit,dist_bit,min_dist,low_cour,max_dist)
    past_time,optimal,obj,x = binary_search(n_couriers, n_items,D_int,max_dist,min_dist,(maximum, x, cour_dist, weights))
    print(past_time, optimal, obj)

    return jsonizer(x,n_items+1,n_couriers,past_time,optimal,obj)

def binary_search(m,n,D,courier_dist_ub_bool,courier_dist_lb_bool, variables,timeout=time_limit):

    res, X, D_tot, _ = variables
    maxDistBin= int(np.ceil(np.log2(sum( [max(i) for i in D] ))))
    
    UB = toInteger(courier_dist_ub_bool)
    LB = toInteger(courier_dist_lb_bool)
    
    s.set('timeout', timeout * 1000)

    start_time = time.time()
    iter = 0
    
    satisf = True
    optimal = True
    prevModel = None
    
    while(satisf):
        if (UB - LB) <= 1:
            satisf = False
        if UB - LB == 1:
            MB = LB
        else:
            MB = int(np.ceil((UB + LB) / 2))
        middle_bits = toBinary(MB, maxDistBin, BoolVal) 
        s.add(lesseq(res,middle_bits))  
        past_time = int(time.time() - start_time)
        print(past_time)
        s.set('timeout', (timeout - past_time)*1000)
        status = s.check()
    
        if status == sat:
            iter += 1
            model = s.model()
            prevModel = model
            dist = toInteger([model.evaluate(res[b]) for b in range(maxDistBin)])
            UB = dist

        elif status == unsat:
            iter += 1
            s.pop()
            s.push()
            LB = MB
        
        elif status == unknown:
            if iter == 0:
                return timeout, False, -1, []
            satisf = False
            optimal = False
        
    past_time = time.time() - start_time

    model = prevModel
    x = [[[int(is_true(model[X[k][i][j]]))
                            for j in range(n+1)] 
                            for i in range(n+1)] 
                            for k in range(m)]
    xDist = [model.evaluate(bool_vars_to_int(b)).as_long() for b in D_tot]
    obj = max(xDist)
    return (int(past_time),optimal,obj,x)


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
    s.set("timeout", time_limit*1000)
    # Upper/lower bounds on the function to minimize
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
    s.add(maxim(cour_dist,maximum,"cl"))

    # for d in cour_dist:
    #         s.add(greater_eq(maximum,d))
    s.push()
    return maximum, p, cour_dist, weights

if __name__ == "__main__":
    main(instance,cfg)