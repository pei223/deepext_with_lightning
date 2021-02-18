from typing import Tuple, List, Dict

from matplotlib import pyplot as plt


def create_label_list_and_dict(label_file_path: str) -> Tuple[List[str], Dict[str, int]]:
    label_names, label_dict = [], {}
    i = 0
    with open(label_file_path, "r") as file:
        for line in file:
            label_name = line.replace("\n", "")
            label_names.append(label_name)
            label_dict[label_name] = i
            i += 1
    return label_names, label_dict


def visualize_category_hist(title: str, val_dict: Dict[str, int], out_filepath: str):
    label = list(val_dict.keys())
    y = list(val_dict.values())
    x = [_ for _ in range(len(y))]
    plt.bar(x, y, tick_label=label, align="center")
    plt.title(title)
    plt.savefig(out_filepath)
    plt.close()


def label_names_to_dict(label_names: List[str]):
    class_index_dict = {}
    for i, label_name in enumerate(label_names):
        class_index_dict[label_name] = i
    return class_index_dict
