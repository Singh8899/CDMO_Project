#!/bin/bash

solver_type="${1:-cp}"

python_script="/app/All_solver.py"

case "$solver_type" in
  "cp"|"mip"|"sat"|"smt")
    echo "Running $solver_type Solver with specified arguments..."
    # Pass all arguments except the first (solver type) to the Python script
    python $python_script --approach="$solver_type" "${@:2}"
    ;;
  *)
    echo "Error: Unknown solver type '$solver_type'"
    exit 1
  ;;
esac
