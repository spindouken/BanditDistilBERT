# Use the official TensorFlow Docker image with GPU support
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory
WORKDIR /workspace

# Copy the current directory contents into the container at /workspace
COPY . /workspace

# Install dependencies
RUN pip install --upgrade pip
RUN pip install pandas scikit-learn transformers wandb

# Set the default command to bash (optional)
CMD ["/bin/bash"]
