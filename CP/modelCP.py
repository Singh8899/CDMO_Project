import minizinc as mzn
import numpy as np
import datetime
import os
import json
import time as tm
import argparse

mzn.warnings.simplefilter("ignore")


def get_model(model_config):

     script_dir = os.path.dirname(os.path.abspath(__file__))


     if model_config=="optimized":
         file_path = os.path.join(script_dir, "Solvers/optimized.mzn")

         model=mzn.Model(file_path)
     if model_config=="optimized_no_heu":
         file_path = os.path.join(script_dir, "Solvers/optimized_no_heu.mzn")

         model=mzn.Model(file_path)
     if model_config=="sym":
         file_path = os.path.join(script_dir, "Solvers/sym.mzn")
         model=mzn.Model(file_path)
     if model_config=="sym_no_heu":
         file_path = os.path.join(script_dir, "Solvers/sym_no_heu.mzn")
         model=mzn.Model(file_path)
     if model_config=="standard":
         file_path = os.path.join(script_dir, "Solvers/standard.mzn")

         model=mzn.Model(file_path)
     if model_config=="standard_no_heu":
         file_path = os.path.join(script_dir, "Solvers/standard_no_heu.mzn")

         model=mzn.Model(file_path)
     if model_config=="single":
         file_path = os.path.join(script_dir, "Solvers/single.mzn")

         model=mzn.Model(file_path)

     return model

def solve_CP(instance_number, model, solver):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    time_to_solve=300
    if instance_number >= 10:
        file_path = os.path.join(script_dir, "Instances/inst" + str(instance_number) + ".dzn")
        model.add_file(file_path)
    else:
        file_path = os.path.join(script_dir, "Instances/inst0" + str(instance_number) + ".dzn")
        model.add_file(file_path)
    solver = mzn.Solver.lookup(solver)

    instance_to_solve = mzn.Instance(solver, model)

    start_time = tm.time()
    time_limit = datetime.timedelta(seconds=time_to_solve)
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
        while j < ass.shape[0] :
            if ass[j, i] != np.max(ass):
                res_courier.append(ass[j, i])
            j += 1
        res.append(res_courier)
    res = [
        [int(item) for item in sublist]
        for sublist in res
    ]
    return res





def main(instance,configuration,solver):
    n_instances = 21
    configurations = ["standard", "standard_no_heu", "sym", "sym_no_heu", "optimized","single"]
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    solvers = ["gecode", "com.google.ortools.sat", "chuffed"]

    if instance !=0:
        json_solver={}
        instance_number = instance
        if configuration ==0 and solver ==0:

            for model_config in configurations:

                for solver in solvers:
                    if (solver == "chuffed" or solver == "com.google.ortools.sat") and "_" not in model_config:
                        continue
                    print(
                        f"Solving Instance number ===> {instance_number} with solver ===> {solver} and model_config ===> {model_config}")

                    model = get_model(model_config)

                    time, optimal, obj, sol = solve_CP(instance_number, model, solver)
                    print(
                        f"SOLVING STATHISTICS ===> time=>{round(time, 1)} optimal ==>{optimal} objective found ==>{obj} ")
                    print(f"best couriers route found ====> {str(sol)}")
                    json_instance = {}
                    if time>300:
                        time=300
                    json_instance["time"] = int(time)
                    json_instance["optimal"] = optimal
                    json_instance["obj"] = obj
                    json_instance["sol"] = sol
                    if solver == "com.google.ortools.sat":
                        json_solver["or-tools" + "_" + model_config] = json_instance
                    else:
                        json_solver[solver + "_" + model_config] = json_instance

            with open(parent_directory + "/res/CP/" + str(instance_number) + ".json", 'w') as file:

                json.dump(json_solver, file, indent=3)
        if configuration == 0 and solver !=0:

            for model_config in configurations:
                if (solver == "chuffed" or solver == "com.google.ortools.sat") and "_" not in model_config:
                    continue
                print(
                    f"Solving Instance number ===> {instance_number} with solver ===> {solver} and model_config ===> {model_config}")

                model = get_model(model_config)

                time, optimal, obj, sol = solve_CP(instance_number, model, solver)
                print(
                    f"SOLVING STATHISTICS ===> time=>{round(time, 1)} optimal ==>{optimal} objective found ==>{obj} ")
                print(f"best couriers route found ====> {str(sol)}")
                json_instance = {}
                if time>300:
                    time=300
                json_instance["time"] = int(time)
                json_instance["optimal"] = optimal
                json_instance["obj"] = obj
                json_instance["sol"] = sol
                if solver == "com.google.ortools.sat":
                    json_solver["or-tools" + "_" + model_config] = json_instance
                else:
                    json_solver[solver + "_" + model_config] = json_instance

            with open(parent_directory + "/res/CP/" + str(instance_number) + ".json", 'w') as file:

                json.dump(json_solver, file, indent=3)
        if configuration !=0 and solver ==0:

            for solver in solvers:
                if (solver == "chuffed" or solver == "com.google.ortools.sat") and "_" not in model_config:
                    continue
                print(
                    f"Solving Instance number ===> {instance_number} with solver ===> {solver} and model_config ===> {model_config}")

                model = get_model(model_config)

                time, optimal, obj, sol = solve_CP(instance_number, model, solver)
                print(
                    f"SOLVING STATHISTICS ===> time=>{round(time, 1)} optimal ==>{optimal} objective found ==>{obj} ")
                print(f"best couriers route found ====> {str(sol)}")
                json_instance = {}
                if time>300:
                    time=300
                json_instance["time"] = int(time)
                json_instance["optimal"] = optimal
                json_instance["obj"] = obj
                json_instance["sol"] = sol
                if solver == "com.google.ortools.sat":
                    json_solver["or-tools" + "_" + model_config] = json_instance
                else:
                    json_solver[solver + "_" + model_config] = json_instance

            with open(parent_directory + "/res/CP/" + str(instance_number) + ".json", 'w') as file:

               json.dump(json_solver, file, indent=3)
        if configuration !=0 and solver !=0:
            model_config = configuration
            if (solver == "chuffed" or solver == "com.google.ortools.sat") and "_" not in model_config:
                print("Can't solve the instance with this solver and specific model")
                return
            print(
                f"Solving Instance number ===> {instance_number} with solver ===> {solver} and model_config ===> {model_config}")

            model = get_model(model_config)

            time, optimal, obj, sol = solve_CP(instance_number, model, solver)
            print(
                f"SOLVING STATHISTICS ===> time=>{round(time, 1)} optimal ==>{optimal} objective found ==>{obj} ")
            print(f"best couriers route found ====> {str(sol)}")
            json_instance = {}
            if time>300:
                time=300
            json_instance["time"] = int(time)
            json_instance["optimal"] = optimal
            json_instance["obj"] = obj
            json_instance["sol"] = sol
            if solver == "com.google.ortools.sat":
                json_solver["or-tools" + "_" + model_config] = json_instance
            else:
                json_solver[solver + "_" + model_config] = json_instance

        with open(parent_directory + "/res/CP/" + str(instance_number) + ".json", 'w') as file:

           json.dump(json_solver, file, indent=3)

    if instance ==0:

        # Solve for every instance
        for instance_number in range(1, n_instances + 1):
            json_solver = {}
            # now for every configuration
            for model_config in configurations:

                for solver in solvers:
                    if (solver == "chuffed" or solver == "com.google.ortools.sat") and "_" not in model_config:
                        continue
                    print(
                        f"Solving Instance number ===> {instance_number} with solver ===> {solver} and model_config ===> {model_config}")

                    model = get_model(model_config)

                    time, optimal, obj, sol = solve_CP(instance_number, model, solver)
                    print(
                        f"SOLVING STATHISTICS ===> time=>{round(time, 1)} optimal ==>{optimal} objective found ==>{obj} ")
                    print(f"best couriers route found ====> {str(sol)}")
                    json_instance = {}
                    if time>300:
                        time=300
                    json_instance["time"] = int(time)
                    json_instance["optimal"] = optimal
                    json_instance["obj"] = obj
                    json_instance["sol"] = sol
                    if solver == "com.google.ortools.sat":
                        json_solver["or-tools" + "_" + model_config] = json_instance
                    else:
                        json_solver[solver + "_" + model_config] = json_instance

            with open(parent_directory + "/res/CP/" + str(instance_number) + ".json", 'w') as file:

                json.dump(json_solver, file, indent=3)

if __name__ == "__main__":
    main(instance,configuration,solver)