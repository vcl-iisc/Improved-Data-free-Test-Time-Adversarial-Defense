import json

# Store the Python command in a variable
python_command = "python combined_mahalnobis.py --dataset cifar10 --batch_size 64 --model_name vgg16 --model_path data/source/cifar10/vgg/net.pt --detector_path data/source/fmnist/resnet18/fmnist_pgd_seed_0_source_detector.pt --attacks pgd --method vanila --gpu 0 --droprate 0.005 --seed 0 --lr 0.005 --epochs 10 --s_dataset fmnist  --ent_par 0.8 --cls_par 0.3 --recreate_adv_data"

# Split the command into a list of individual words
command_parts = python_command.split()

# Get the path to the Python executable
python_executable = command_parts[0]

# Get the path to the script
script_path = command_parts[1]

# Create the configuration object
config = {
    "name": "My Python Script",
    "type": "python",
    "request": "launch",
    "program": script_path,
    "console": "integratedTerminal",
    "env": {},
    "args": [],
    "cwd": "${workspaceFolder}",
    "envFile": "${workspaceFolder}/.env",
    "pythonPath": python_executable,
    "debugOptions": [
        "RedirectOutput"
    ]
}

# Add any additional arguments to the args list
for i in range(2, len(command_parts)):
    config["args"].append(command_parts[i])

# Print the configuration object as a JSON string
print(json.dumps(config ,  indent=2))
