import subprocess
import multiprocessing
import sys

# Define the datasets and augmentations
datasets = ["musae_Twitch_DE", "grand_Brain", "grand_LungCancer", "benchmark_Cora_Cite", "musae_Github"]
augmentations =  ['NodeDrop', 'EdgeDrop', 'NodeMixUp', 'NodeFeatureMasking','NULL']

# Retrieve the model from the command line arguments
model = sys.argv[1]

# Function to run the subprocess with output redirected to a file
def run_process(dataset, augmentation, model):
    data_dir = f"data_reformatted/{dataset.lower()}"
    raw_data_dir = f"data_reformatted/raw/"

    # Construct the command
    command = f"python train.py --method {model} --dname {dataset} " \
              f"--All_num_layers 2 --MLP_num_layers 2 --MLP2_num_layers 2 --MLP3_num_layers 2 " \
              f"--Classifier_num_layers 2 --MLP_hidden 256 --Classifier_hidden 64 --aggregate mean " \
              f"--restart_alpha 0.5 --lr 0.001 --wd 0 --epochs 500 --runs 2 --feature_noise 1.0 --cuda -1 " \
              f"--data_dir {data_dir} --raw_data_dir {raw_data_dir} " \
              f"--Augment {augmentation}"

    # Run the command and redirect output to a file
    with open(f'outputs/output_{dataset}_{augmentation}.txt', 'w') as outfile:
        subprocess.run(command, shell=True, check=True, stdout=outfile, stderr=subprocess.STDOUT)

# Limit for the number of processes
process_limit = 64  # Adjust this number based on your system's capabilities

# Create a pool of workers with the specified limit and run processes
if __name__ == "__main__":
    pool = multiprocessing.Pool(process_limit)
    for dataset in datasets:
        for aug in augmentations:
            pool.apply_async(run_process, args=(dataset, aug, model))

    pool.close()
    pool.join()
