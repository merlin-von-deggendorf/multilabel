import labeler
from StringIDMapper import StringIDMapper
import datasetsplitter

# unique_labels:StringIDMapper = labeler.get_unique_labels("D:/datasets/multilabel/unknownclass/")

# for id, label in unique_labels.id2voc.items():
#     print(f"ID: {id}, Label: {label}")

# folders = labeler.list_folders("D:/datasets/multilabel/unknownclass/")
# for folder in folders:
#     labels = labeler.extract_label(folder)
#     print(f"Folder: {folder}, Labels: {labels}")

datasetsplitter.split_dataset("D:/datasets/multilabel/missingclass/", ratio=0.8)