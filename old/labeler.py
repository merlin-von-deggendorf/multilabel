import os
from StringIDMapper import StringIDMapper

def list_folders(data_dir):
    try:
        items = os.listdir(data_dir)
    except FileNotFoundError:
        print(f"Directory '{data_dir}' not found.")
        return []
    # Filter only directories
    folders = [item for item in items if os.path.isdir(os.path.join(data_dir, item))]
    return folders

def extract_label(folder) -> list[str]:
    labels = folder.split("___")
    for i in range(len(labels)):
        labels[i] = labels[i].replace("_", " ").lower()
    return labels


def relative_2_absolute_path(relative_path):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(current_dir, relative_path)
  
def get_unique_labels(folder)-> StringIDMapper:
    folders = list_folders(folder)
    string_id_mapper = StringIDMapper()
    for folder in folders:
        labels=extract_label(folder)
        for label in labels:
            string_id_mapper.add_word(label)
    return string_id_mapper

        
