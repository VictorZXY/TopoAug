import subprocess
import sys

# datasets = ["musae_Facebook", "musae_Github",
#             "musae_Twitch_FR", "musae_Twitch_EN", "musae_Twitch_ES",
#             "musae_Twitch_PT", "musae_Twitch_RU", "musae_Twitch_DE",
#             "musae_Wiki_chameleon", "musae_Wiki_crocodile", "musae_Wiki_squirrel",
#             "grand_ArteryAorta", "grand_ArteryCoronary", "grand_Breast", "grand_Brain", "grand_Leukemia",
#             "grand_Lung", "grand_Stomach", "grand_LungCancer", "grand_StomachCancer", "grand_KidneyCancer",
#             "amazon_Photo", "amazon_Computer", "benchmark_Cora_Author", "benchmark_Cora_Cite", "benchmark_Pubmed"]
# datasets = ["benchmark_Cora_Author", "benchmark_Cora_Cite", "benchmark_Pubmed"]
datasets = ["musae_Twitch_DE","grand_Brain","grand_LungCancer","benchmark_Cora_Cite","musae_Github"]
datasets = ["grand_LungCancer","benchmark_Cora_Cite","musae_Github"]
# datasets = ["grand_LungCancer"]

# augmentations  = ['NodeDrop', 'EdgeDrop', 'NodeMixUp', 'NodeFeatureMasking']
# augmentations  = ['NodeDrop']
augmentations  = ['NodeFeatureMasking','NULL']

model = sys.argv[1]

# Update the dataset names and directories as needed
for dataset in datasets:
    # Cirrus HPC
    data_dir = f"data_reformatted/{dataset.lower()}"
    raw_data_dir = f"data_reformatted/raw/"

    # Local
    # data_dir = f"data_reformatted/{dataset.lower()}"
    # raw_data_dir = f"data_reformatted/"
    for aug in augmentations:
        command = f"python train.py --method {model} --dname {dataset} " \
                f"--All_num_layers 2 --MLP_num_layers 2 --MLP2_num_layers 2 --MLP3_num_layers 2 " \
                f"--Classifier_num_layers 2 --MLP_hidden 256 --Classifier_hidden 64 --aggregate mean " \
                f"--restart_alpha 0.5 --lr 0.001 --wd 0 --epochs 500 --runs 2 --feature_noise 1.0 --cuda -1 " \
                f"--data_dir {data_dir} --raw_data_dir {raw_data_dir} " \
                f"--Augment {aug}"
        subprocess.run(command, shell=True, check=True)
