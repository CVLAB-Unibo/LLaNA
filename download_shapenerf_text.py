'''

This script downloads the ShapeNeRF-Text dataset from the Huggingface Hub and prepares the correct directory structure for the dataset.

'''

import os

def download_data():
    os.makedirs('data', exist_ok=True)
    os.system('git clone https://huggingface.co/datasets/andreamaduzzi/ShapeNeRF-Text data/shapenerf_text')

def prepare_data():
    # move training nerf2vec embeddings to parent folder
    train_vecs_folder = 'data/shapenerf_text/train/vecs'
    for train_batch_folder in os.listdir(train_vecs_folder):
        print('processing ', train_batch_folder)
        train_batch_path = os.path.join(train_vecs_folder, train_batch_folder)
        for vec_file in os.listdir(train_batch_path):
            vec_path = os.path.join(train_batch_path, vec_file)
            os.rename(vec_path, os.path.join(train_vecs_folder, vec_file))
        os.rmdir(train_batch_path)

def main():
    download_data()
    prepare_data()


if __name__ == "__main__":
    main()