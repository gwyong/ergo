import os, pickle
import cv2

def get_filename(path):
    return os.path.basename(path)

def save_pickle_file(file, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(file, f)

def load_data_from_pickle(load_path):
    with open(load_path, 'rb') as file:
        data = pickle.load(file)
    return data

