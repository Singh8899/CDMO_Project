# Multiple courier satisfaction problem
## Project description
This is the 2023/2024 project for Combinatorial Decision Making and Optimization(CDMO) course of University of Bologna.
Different strategies have been used:

Constraint Programming with MiniZinc with gecode/or-tools/chuffed solvers
SAT-Solvers with Z3 
SMT-Solvers with Z3 
Mixed Integer Programming with Pulp Python library with CBC/Highs solvers

This repository contains a Docker setup for running MiniZinc models across multiple data instances. It uses the official MiniZinc Docker image as a base and includes a script to run the model against all `.dzn` and `.dat` instance files located in the `instances` directory.

## Requirements

- Docker installed on your machine. Visit [Docker's official website](https://docs.docker.com/get-docker/) for installation instructions.

## Project Structure
- `Dockerfile`: Defines the Docker container setup.
- `instance`: specify instance to equal an instance number. For all use instances=0 

## CP arguments
- `configurations` = ["standard", "standard_no_heu", "sym", "sym_no_heu", "optimized"]
- `solvers` = ["gecode", "com.google.ortools.sat", "chuffed"]
## MIP arguments
- `configurations` = [0,1,2] 0=all 1=CBC 2=HiGHS
## SAT arguments
- `configurations` = [0,1,2] 0=all 1=standard 2=BS
## SMT arguments
- `configurations` = ["0",standard", "standard_sym_heu",]

### Building the Docker Image

To build the Docker image, navigate to the root of this project directory where the Dockerfile is located and run the following command:

```bash
cd ./project_directory
docker build -t solver .
docker run --td solver 

```

# CP run image
```bash
python3 All_solver.py --instance 0..21 --approach cp --configuration configurations --solver solvers_type
```

# MIP run image
```bash
python3 All_solver.py --instance 0..21 --approach mip --configuration configurations
```
# SAT  run image
```bash
python3 All_solver.py --instance 0..21 --approach sat --configuration configurations
```

# SMT  run image
```bash
python3 All_solver.py --instance 0..21 --approach smt --configuration configurations
```

# TO EXTRACT THE FOLDER WITH THE SOLUTIONS(run in a new CMD prompt)
```bash
docker cp docker_image_id:/CDMO_Project/res project_directory
Example docker_image_id = 47674d6b939a
        project_directory = C:\Users\singh\Documents\GitHub\CDMO_Project
```

# The solution checker folder contains the inputs output just have to run a comand with your local directory
```bash
python C:\Users\singh\Documents\GitHub\CDMO_Project\solut_check\Solution_Checker.py C:\Users\singh\Documents\GitHub\CDMO_Project\solut_check\input C:\Users\singh\Documents\GitHub\CDMO_Project\solut_check\output
```
the three above are the directory where are present the checker file, the input folder and the output folder
