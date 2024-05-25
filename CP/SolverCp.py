import minizinc as mzn
import numpy as np
import datetime
import os
import json
import time as tm

time_limit=1
def get_model(model_config):

     if model_config=="implied_sym":

         model=mzn.Model("Solvers/implied_sym.mzn")
     if model_config=="sym":

         model=mzn.Model("Solvers/sym.mzn")
     if model_config=="implied":

         model=mzn.Model("Solvers/implied.mzn")
     if model_config=="standard":
         model=mzn.Model("Solvers/standard.mzn")

     return model

def solve_CP(instance_number, model, solver):
    time_limit=1
    if instance_number >= 10:
        model.add_file("Instances/inst" + str(instance_number) + ".dzn")
    else:
        model.add_file("Instances/inst0" + str(instance_number) + ".dzn")
    solver = mzn.Solver.lookup(solver)

    instance_to_solve = mzn.Instance(solver, model)

    start_time = tm.time()
    time_limit = datetime.timedelta(seconds=time_limit)
    res = instance_to_solve.solve(timeout=time_limit)
    end_time = tm.time()
    time = end_time - start_time

    if res.status == mzn.Status.UNSATISFIABLE or res.status == mzn.Status.UNKNOWN:

        optimal = False
        obj = None
        sol = []
    else:

        obj = res["max_cap_dist"]
        ass = res["ass"]
        if time < 300:
            optimal = True
            sol = res_CP(ass)
        else:
            optimal = False
            sol = res_CP(ass)

    return time, optimal, obj, sol


def res_CP(ass):  # We have a matrix with n_couriers columns, when we get a zero we reached the final destination
    ass = np.array(ass)
    n_couriers = ass.shape[1]
    res = []
    for i in range(n_couriers):

        j = 0
        res_courier = []
        while j < ass.shape[0] and ass[j, i] != 0:
            res_courier.append(ass[j, i])
            j += 1
        res.append(res_courier)
    return res





def main():
    n_instances = 21
    configurations = ["sym","implied_sym","implied","standard"]
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    solvers = ["gecode"]
    #Solve for every instance
    path_mip_res="Res/"
    for instance_number in range(1,n_instances+1):
        #now for every configuration
        json_final = {}
        for model_config in configurations:

            json_solver = {}
            for solver in solvers:
                #model=mzn.Model("Solvers/implied.mzn")
                model=get_model(model_config)

                time,optimal,obj,sol=solve_CP(instance_number, model, solver)
                json_instance={}
                json_instance["time"] = time
                json_instance["optimal"] = optimal
                json_instance["obj"] = obj
                json_instance["sol"] = str(sol)

            json_solver[solver+"_"+model_config] =json_instance

        #json_final[] = json_solver
        with open(parent_directory+"/res/CP/" + str(instance_number) +".json", 'w') as file:
            json.dump(json_solver, file, indent=4)

if __name__ == "__main__":
    main()






