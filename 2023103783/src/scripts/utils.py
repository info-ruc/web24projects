import os

def check_file_in_directory(directory, filename):
    for file in os.listdir(directory):
        if file == filename:
            return True
    return False