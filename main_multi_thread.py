import multiprocessing
import subprocess
import sys

# Define the datasets and augmentations
# datasets = ["musae_Facebook", "musae_Github",
#             "musae_Twitch_FR", "musae_Twitch_EN", "musae_Twitch_ES",
#             "musae_Twitch_PT", "musae_Twitch_RU", "musae_Twitch_DE",
#             "musae_Wiki_chameleon", "musae_Wiki_crocodile", "musae_Wiki_squirrel",
#             "grand_ArteryAorta", "grand_ArteryCoronary", "grand_Breast", "grand_Brain", "grand_Leukemia",
#             "grand_Lung", "grand_Stomach", "grand_LungCancer", "grand_StomachCancer", "grand_KidneyCancer",
#             "amazon_Photo", "amazon_Computer", "benchmark_Cora_Author", "benchmark_Cora_Cite", "benchmark_Pubmed"]
datasets = ["musae_Github", "musae_Twitch_DE", "grand_Brain", "grand_LungCancer", "benchmark_Cora_Cite",
            "amazon_Computer"]
augmentations = ['NULL', 'NodeDrop', 'EdgeDrop', 'NodeMixUp', 'NodeFeatureMasking']

# Retrieve the model from the command line arguments
model = sys.argv[1]


# Function to run the subprocess with output redirected to a file
def run_process(dataset, augmentation, model):
    # Cirrus HPC
    data_dir = f"/work/ec249/ec249/xz9118/Projects/graph-cross-attention/data_reformatted/{dataset.lower()}"
    raw_data_dir = f"/work/ec249/ec249/xz9118/Projects/graph-cross-attention/data_reformatted/raw/"

    # Local
    # data_dir = f"data_reformatted/{dataset.lower()}"
    # raw_data_dir = f"data_reformatted/"

    # Construct the command
    command = f"python train.py --method {model} --dname {dataset} --augment {augmentation} " \
              f"--All_num_layers 2 --MLP_num_layers 2 --MLP2_num_layers 2 --MLP3_num_layers 2 " \
              f"--Classifier_num_layers 2 --MLP_hidden 256 --Classifier_hidden 64 --aggregate mean " \
              f"--restart_alpha 0.5 --lr 0.001 --wd 0 --epochs 500 --runs 2 --feature_noise 1.0 --cuda 0 " \
              f"--data_dir {data_dir} --raw_data_dir {raw_data_dir}"

    # Run the command and redirect output to a file
    with open(f'out/{dataset}-{model}-{augmentation}.out', 'w') as outfile:
        subprocess.run(command, shell=True, check=True, stdout=outfile, stderr=subprocess.STDOUT)


# Limit for the number of processes
NUM_PROCESSES_LIMIT = 64  # Adjust this number based on your system's capabilities

# Create a pool of workers with the specified limit and run processes
if __name__ == "__main__":
    pool = multiprocessing.Pool(NUM_PROCESSES_LIMIT)
    for dataset in datasets:
        for aug in augmentations:
            pool.apply_async(run_process, args=(dataset, aug, model))

    pool.close()
    pool.join()
