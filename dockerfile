# We use the minizinc image as a base
FROM minizinc/minizinc:latest

# Setting the working directory
WORKDIR ./CDMO_Project

# Coping all the content of this folder into the container
COPY . .

# Installing python
RUN apt-get update \
  && apt-get install -y python3 \
  && apt-get install -y python3-pip

# Install required libraries
RUN python3 -m pip install -r requirements.txt --break-system-packages