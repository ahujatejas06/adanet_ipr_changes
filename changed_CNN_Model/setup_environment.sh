#!/bin/bash
ENV_NAME="adanet"

PYTHON_VERSION="3.6"

REQUIREMENTS_FILE="requirements.txt"

echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

if [ $? -eq 0 ]; then
  echo "Environment '$ENV_NAME' created successfully!"
else
  echo "Error in creating environment '$ENV_NAME'. Exiting..."
  exit 1
fi

echo "Activating conda environment '$ENV_NAME'"
source activate $ENV_NAME

if [ $? -eq 0 ]; then
  echo "Environment '$ENV_NAME' activated successfully!"
else
  echo "Error in activating environment '$ENV_NAME'. Exiting..."
  exit 1
fi

if [ -f "$REQUIREMENTS_FILE" ]; then
  echo "Installing packages from '$REQUIREMENTS_FILE'"
  pip install -r $REQUIREMENTS_FILE

  if [ $? -eq 0 ]; then
    echo "Packages from '$REQUIREMENTS_FILE' installed successfully!"
  else
    echo "Error in installing packages. Exiting..."
    exit 1
  fi
else
  echo "No requirements.txt file found. Skipping package installation."
fi
