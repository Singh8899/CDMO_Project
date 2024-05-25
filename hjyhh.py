import minizinc as mzn
import os
import datetime
import time as tm
import numpy as np
def get_model(implied, sym):
    if implied and sym:
        print("ffd")
        model = mzn.Model("CP/Solvers/implied_sym.mzn")
    if sym and not implied:
        model = mzn.Model("CP/Solvers/sym.mzn")
    if not sym and implied:
        model = mzn.Model("CP/Solvers/implied.mzn")
    if not sym and not implied:
        model = mzn.Model("CP/Solvers/standard.mzn")
    # return mzn.Model("CP/Solvers/sym.mzn")
    return model
configurations=[""]
for i in configurations:
    print(i)
    model = get_model(i[0],i[1])






def solve_CP(instance_number, model, solver):
    if instance_number>10:
        model.add_file("CP/Instances/inst"+str(instance_number)+".dzn")
    else:
        model.add_file("CP/Instances/inst0" + str(instance_number) + ".dzn")
    solver = mzn.Solver.lookup(solver)

    instance_to_solve = mzn.Instance(solver, model)


    start_time = tm.time()
    time_limit =datetime.timedelta(seconds=1)
    res = instance_to_solve.solve(timeout=time_limit)
    end_time = tm.time()
    time= end_time - start_time


    if res.status == mzn.Status.UNSATISFIABLE or res.status==mzn.Status.UNKNOWN:

        optimal=False
        obj=None
        sol=[]
    else:

        obj = res["max_cap_dist"]
        ass= res["ass"]
        if time < 300 :
            optimal=True
            sol = res_CP(ass)
        else:
            optimal=False
            sol = res_CP(ass)

    return time,optimal,obj,sol

def res_CP(ass): # We have a matrix with n_couriers columns, when we get a zero we reached the final destination
    ass = np.array(ass)
    n_couriers = ass.shape[1]
    res=[]
    for i in range(n_couriers):

        j=0
        res_courier=[]
        while j<ass.shape[0] and ass[j,i]!=0 :
            res_courier.append(ass[j,i])
            j+=1
        res.append(res_courier)
    return res



print(solve_CP(1,model,"gecode"))