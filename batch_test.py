import subprocess
import pandas as pd
import os
from utils_gen_mia import create_dataset_with_metadata, infer_task_type
from process_dataset import process_data

def train_sample_tabsyn(dataset, N, dataset_name="test_dataset", save_path="./data/synthetic_data.csv"):
    """
    Runs the Tabsyn training and synthesis pipeline for a specified dataset.
    
    Args:
    - dataset (np.ndarray or pd.Dataframe): The input dataset to be used.
    - N (int): Unused parameter, could be removed or repurposed as needed.
    - dataset_name (str): The name of the dataset to be used for training and synthesis.
    - save_path (str): The path to save the synthesized data temporarily.

    Returns:
    - pd.DataFrame: The synthetic data as a DataFrame.
    """
    # Save the current working directory
    original_dir = os.getcwd()
    
    # Call the Tabsyn functions to create the dataset with metadata
    
    try:
        # Change to the TabSyn directory
        #os.chdir("TabSyn")

        task_type = infer_task_type(dataset)
        print(dataset)
        create_dataset_with_metadata(dataset, dataset_name, task_type)
        process_data(dataset_name)

        print("Current work dir:",os.getcwd())
        #import importlib
        #import sys
        #sys.path.append('~/TabSyn')
        # Step 1: Train the VAE model
        subprocess.run([
            "python", "main.py",
            "--dataname", dataset_name,
            "--method", "vae",
            "--mode", "train"
        ], check=True)
        
        # Step 2: Train the diffusion model
        subprocess.run([
            "python", "main.py",
            "--dataname", dataset_name,
            "--method", "tabsyn",
            "--mode", "train"
        ], check=True)

        print("Diffusion training completed!")
        
        # Step 3: Synthesize data using the trained model
        subprocess.run([
            "python", "main.py",
            "--dataname", dataset_name,
            "--method", "tabsyn",
            "--mode", "sample",
            "--save_path", save_path
        ], check=True)
        
        # Step 4: Load the synthetic data
        synthetic_df = pd.read_csv(save_path)
        
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during subprocess execution: {e}")
        return None
    
    finally:
        
        # Clean up by deleting the synthesized data file after loading
        if os.path.exists(save_path):
            os.remove(save_path)
            
        # Ensure returning to the original working directory
        os.chdir(original_dir)
    
    return synthetic_df

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def process_folder_and_generate_synthetic_data(folder_path, save_folder, test_idx):
    """
    Loops through all CSV files in a folder, performs train/test split, 
    and calls the train_sample_tabsyn function with the training data.

    Args:
    - folder_path (str): Path to the folder containing the CSV files.
    - save_folder (str): Path to the folder to save synthetic data files.
    - test_idx (int): Index to set the random_state for the train/test split.

    Returns:
    - None
    """
    # Ensure the save folder exists
    os.makedirs(save_folder, exist_ok=True)

    # Set up logging
    import logging
    logging.basicConfig(filename='process_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            dataset_name = os.path.splitext(file_name)[0]

            try:
                # Load the dataset
                data = pd.read_csv(file_path)

                save_path = os.path.join(save_folder, f"{dataset_name}_TabSyn_default_{test_idx}.csv")
                if os.path.exists(save_path):
                    logging.info(f"Synthetic data already exists for {file_name}. Skipping...")
                    continue

                # Perform train/test split
                if 'SOURCE_LABEL' in data.columns:
                    logging.info("Splitting based on source label!")
                    train = data[data['SOURCE_LABEL'] == 'train'].drop(columns=['SOURCE_LABEL'])
                    test = data[data['SOURCE_LABEL'] == 'test'].drop(columns=['SOURCE_LABEL'])
                else:
                    train, test = train_test_split(data, test_size=0.2, random_state=42+test_idx)

                # Save synthetic data for the train set
                

                logging.info(f"Processing {file_name}...")
                synthetic_data = train_sample_tabsyn(train, N=None, dataset_name=dataset_name, save_path=save_path)

                if synthetic_data is not None:
                    synthetic_data.to_csv(save_path, index=False)
                    logging.info(f"Synthetic data generated and saved to {save_path}")
                else:
                    logging.warning(f"Failed to generate synthetic data for {file_name}")

            except Exception as e:
                logging.error(f"An error occurred while processing {file_name}: {e}")


def main():
    """
    Main function to take user input for folder paths and call the processing function.
    """

    input_folder = '/mnt/c/Research/tabsyn/csv'
    output_folder = '/mnt/c/Research/tabsyn/synth_data'
    test_idx = 1

    # Call the processing function
    process_folder_and_generate_synthetic_data(input_folder, output_folder, test_idx)

if __name__ == "__main__":
    main()