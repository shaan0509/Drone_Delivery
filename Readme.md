# Drone Delivery Optimization RL Project

This project implements a reinforcement learning (RL) environment and training pipeline for optimizing drone delivery routes. It uses [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) to train agents in a custom drone delivery environment.

## Project Structure

- [`requirements.txt`](requirements.txt): Python dependencies.
- [`src/`](src/)
  - [`train.py`](src/train.py): Main training script for RL agent.
  - [`dataset/make_dataset.py`](src/dataset/make_dataset.py): Script to generate synthetic drone delivery datasets.
  - [`envs/drone_delivery_env.py`](src/envs/drone_delivery_env.py): Custom Gymnasium environment for drone delivery.
  - [`envs/__init__.py`](src/envs/__init__.py): Environment module init.
  - [`__init__.py`](src/__init__.py): Package init.
- [`data/`](data/): Generated datasets (ignored by git).
- [`models/`](models/): Saved RL models (ignored by git).
- [`logs/`](logs/): Training logs and TensorBoard files (ignored by git).

## Setup

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Generate dataset:**
   ```sh
   python src/dataset/make_dataset.py
   ```
   This creates a compressed dataset in the `data/` directory.

3. **Train the RL agent:**
   ```sh
   python src/train.py
   ```
   Models and logs will be saved in `models/` and `logs/` respectively.

 4. **Tensorboard :**
   ```sh
    %load_ext tensorboard
    %tensorboard --logdir /your/log/dir/here #<------ add ur log directory path here.
   ```
  This loads the Tensoboard.


## Custom Environment

The core environment is [`DroneDeliveryEnv`](src/envs/drone_delivery_env.py), which simulates the assignment of drones to deliver packages to customers under constraints like capacity, range, and delivery sequence.

## Notes

- All generated data, models, and logs are git-ignored by default (see [`.gitignore`](.gitignore)).
- Training progress and evaluation metrics are logged for monitoring and analysis.

## License

This project is for academic and research purposes and made as a project during my Internship in Indian Institute of Technology (ISM) Dhanbad.
