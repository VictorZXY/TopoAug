import subprocess

# HPC
# datasets = ["musae_Twitch_FR", "musae_Twitch_EN", "musae_Twitch_ES",
#             "musae_Twitch_PT", "musae_Twitch_RU", "musae_Twitch_DE",
#             "grand_ArteryAorta", "grand_ArteryCoronary", "grand_Breast", "grand_Brain",
#             "grand_Leukemia", "grand_Lung", "grand_Stomach", "grand_LungCancer", "grand_StomachCancer",
#             "grand_KidneyCancer", "amazon_Photo", "amazon_Computer",
#             "musae_Facebook", "musae_Github"]

# Local testing
datasets = ["musae_Github"]

# Update the dataset names and directories as needed
for dataset in datasets:
    data_dir = f"data_reformatted/{dataset.lower()}"
    raw_data_dir = f"data_raw/"
    command = f"CUDA_VISIBLE_DEVICES='2' python train.py --method LPGCNEDGNN --dname {dataset} " \
              f"--All_num_layers 2 --MLP_num_layers 2 --MLP2_num_layers 2 --MLP3_num_layers 2 " \
              f"--Classifier_num_layers 2 --MLP_hidden 256 --Classifier_hidden 64 --aggregate mean " \
              f"--restart_alpha 0.5 --lr 0.001 --wd 0 --epochs 500 --runs 2 --feature_noise 1.0 --cuda 0 " \
              f"--data_dir {data_dir} --raw_data_dir {raw_data_dir}"
    subprocess.run(command, shell=True, check=True)
