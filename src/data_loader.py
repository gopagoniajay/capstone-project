import pickle

def load_wesad_subject(file_path):
    # latin1 encoding is mandatory for WESAD pkl files
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data