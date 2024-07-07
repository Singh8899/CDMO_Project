import json
from time import time
import os
import numpy as np
from CP import modelCP
from MIP import modelMIP
from SAT import modelSAT
from SMT import modelSMT
import argparse

def main(args):
    instance = args.instance
    approach=args.approach
    configuration = args.configuration
    solver = args.solver
    if approach.lower() == "cp":
        modelCP.main(instance,configuration,solver)
    if approach.lower() == "mip":
        modelMIP.main(instance,configuration,solver)
    if approach.lower() == "sat":
        modelSAT.main(instance,configuration,solver)
    if approach.lower() == "smt":
        modelSMT.main(instance,configuration,solver)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser.")
    parser.add_argument('--instance', type=int, required=True, help="Instance Number 0 if all")
    parser.add_argument('--approach', type=str, required=False, help="Approach, choose between CP, MIP, SMT or SAT")
    parser.add_argument('--configuration', type=str, required=False, help="Depend on the approach, see configuration file")
    parser.add_argument('--solver', type=str, required=False, help="Depend on the approach, see configuration file")
    args = parser.parse_args()
    main(args)