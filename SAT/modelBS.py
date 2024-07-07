from z3 import *
from utils import *
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
    past_time,optimal,obj,model = binary_search(n_couriers, n_items,D_int,max_dist,min_dist,(maximum, x, cour_dist, weights))
    print(past_time, optimal, obj)
    
    x = [[[int(is_true(model[x[k][i][j]]))
                            for j in range(n_items+1)] 
                            for i in range(n_items+1)] 
                            for k in range(n_couriers)]
    json_f = jsonizer(x,n_items+1,n_couriers,past_time,optimal,obj)
    return json_f

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
            iter += 1
            model = s.model()
            previousModel = model
            dist = toInteger([model.evaluate(rho[b]) for b in range(maxDistBin)])
            UPPER_BOUND = dist

        elif status == unsat:
            iter += 1
            s.pop()
            s.push()
            LOWER_BOUND = MIDDLE_BOUND
        
        elif status == unknown:
            if iter == 0:
                return timeout, False, "N/A", []
            satisfiable = False
            optimal = False
        
    
    current_time = time.time()
    past_time = current_time - start_time

    model = previousModel
    xDist = [model.evaluate(bool_vars_to_int(b)).as_long() for b in D_tot]
    obj = max(xDist)
    return (int(past_time),optimal,obj,model)

def lesseq(a, b):
  constraints = []
  constraints.append(Or(Not(a[0]),b[0]))
  for i in range(1,len(a)):
    constraints.append(Implies(And([a[k] == b[k] for k in range(i)]), Or(Not(a[i]),b[i])))
  return And(constraints)
def toInteger(bool_list):
  """
      Decodes a number from binary form
      :bool_list: a list containing BoolVal variables
  """
  binary_string = ''.join('1' if b else '0' for b in bool_list)
  return int(binary_string, 2)

def toBinary(num, length = None, dtype = int):
  """
      Encodes a number in binary form in the desired type
      :param num: the int number to convert
      :param length: the output length
      :param dtype: the output data type. It supports integers, booleans or z3 booleans
  """
  num_bin = bin(num).split("b")[-1]
  if length:
      num_bin = "0"*(length - len(num_bin)) + num_bin
  num_bin = [dtype(int(s)) for s in num_bin]
  return num_bin

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
def maxim(vec, maxi, name= ""):
  """
      The constraints needed to find the maximum number inside a vector
      :param vec:   list of binary encoded numbers
      :param maxi:  binary encoded maximum
      :param name:  name to disambiguate the slack variables
  """
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
  """
      The constraint a == b
  """
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
if __name__ == "__main__":
    main(instance,cfg)