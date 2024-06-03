import os
from shutil import copy, rmtree
import random

def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)

#split the dataset into train set and val set in a 9:1 ratio
def main():
    random.seed(0)
    split_rate = 0.1

    cwd = os.getcwd()
    data_root = os.path.join(cwd, "PlantVillage")
    origin_plant_path = data_root
    assert os.path.exists(origin_plant_path)
    plant_class = [cla for cla in os.listdir(origin_plant_path)
                    if os.path.isdir(os.path.join(origin_plant_path, cla))]

    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cla in plant_class:
        mk_file(os.path.join(train_root, cla))

    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in plant_class:
        mk_file(os.path.join(val_root, cla))

    for cla in plant_class:
        cla_path = os.path.join(origin_plant_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        eval_index = random.sample(images, k=int(num*split_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")
        print()

    print("processing done!")


if __name__ == '__main__':
    main()

