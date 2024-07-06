from z3 import *
from utils import *
from utils import *
from itertools import combinations
from math import floor
import time

time_limit = 300
s = Solver()
def main():
    for instance in range(8,11):

        json_dict = {}
        global s
        s = Solver()
        s.set('timeout', time_limit * 1000)
        n_couriers, n_items, courier_capacity,item_size, D_int = inputFile(instance)
        courier_capacity,item_size,D,load_bit,dist_bit,min_dist,low_cour,max_dist = instance_format(n_couriers, n_items,courier_capacity,item_size, D_int)
        maximum, x, cour_dist, weights = set_const(n_couriers, n_items, courier_capacity,item_size, D,load_bit,dist_bit,min_dist,low_cour,max_dist)
        
        past_time,optimal,obj = binary_search(n_couriers, n_items,D_int,max_dist,min_dist,(maximum, x, cour_dist, weights))
        print(past_time, optimal, obj)
        # if maximum != None:
        #     solve_time = round(time.time() - start_time)
        #     opt = (time_limit > solve_time)
        #     json_dict["Z3"] = jsonizer(x,n_items+1,n_couriers,solve_time,opt,maximum.as_long())
        #     format_and_store(instance,json_dict)
            
def binary_search(m,n,D,courier_dist_ub_bool,courier_dist_lb_bool, variables,timeout=time_limit):

    rho, X, D_tot, _ = variables
    maxDistBin= int(np.ceil(np.log2(sum( [max(i) for i in D] ))))
    
    UPPER_BOUND = toInteger(courier_dist_ub_bool)
    LOWER_BOUND = toInteger(courier_dist_lb_bool)
    
    s.set('timeout', timeout * 1000)

    start_time = time.time()
    iter = 0
    
    satisfiable = True
    optimal = True
    previousModel = None
    
    while(satisfiable):
        print("UPPER_BOUND",UPPER_BOUND,type(UPPER_BOUND))
        print("LOWER_BOUND",LOWER_BOUND,type(LOWER_BOUND))
        
        if (UPPER_BOUND - LOWER_BOUND) <= 1:
            satisfiable = False
        
        if UPPER_BOUND - LOWER_BOUND == 1:
            MIDDLE_BOUND = LOWER_BOUND
        else:
            MIDDLE_BOUND = int(np.ceil((UPPER_BOUND + LOWER_BOUND) / 2))
            
        middle_bits = toBinary(MIDDLE_BOUND, maxDistBin, BoolVal) 
        s.add(lesseq(rho,middle_bits)) 
        current_time = time.time()
        past_time = int(current_time - start_time)
        s.set('timeout', (timeout - past_time)*1000)
        status = s.check()
    
        if status == sat:
            # print("SAT")
            iter += 1
            model = s.model()
            previousModel = model
            # dist = [model.evaluate(rho[b]) for b in range(maxDistBin)]
            dist = toInteger([model.evaluate(rho[b]) for b in range(maxDistBin)])
            # print(dist)
            UPPER_BOUND = dist

        elif status == unsat:
            # if iter == 0:
            #     print("UNSAT")
            #     past_time = int((current_time - start_time))
            #     return past_time, False, "N/A"
            # print("UNSAT")
            iter += 1
            s.pop()
            s.push()
            LOWER_BOUND = MIDDLE_BOUND
        
        elif status == unknown:
            if iter == 0:
                # print("UNKNOWN RESULT for insufficient time")
                return timeout, False, "N/A", []
            satisfiable = False
            optimal = False
        
    
    current_time = time.time()
    past_time = current_time - start_time

    model = previousModel
    # x = [[[ model.evaluate(X[i][j][k]) for k in range(0,n+1) ] for j in range(n) ] for i in range(m)]
    xDist = [model.evaluate(bool_vars_to_int(b)).as_long() for b in D_tot]
    print(xDist)
    obj = max(xDist)
    # output  
    # tot_s = []
    # for i in range(m):
    #     sol = []
    #     for j in range(n):
    #         for k in range(1,n+1):
    #             if x[i][j][k] == True:
    #                 sol.append(k)
    #     tot_s.append(sol)

    # distances,tot_s = instance.post_process_instance(distances, tot_s)

    return (int(past_time),optimal,obj)

def set_const(n_couriers, n_items, courier_capacity,item_size, D,bit_weight,bit_dist,min_dist,low_cour,max_dist):

def bool_vars_to_int(bool_vars):
    return Sum([If(b, 2**i, 0) for i, b in enumerate(list(reversed(bool_vars)))])

if __name__ == "__main__":
    main()