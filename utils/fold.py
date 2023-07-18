import os, shutil
from sklearn.model_selection import KFold


def dataset_kfold(dataset_dir, save_path):
    data_list = os.listdir(dataset_dir)
    kf = KFold(5)

    for i, (tr, val) in enumerate(kf.split(data_list), 1):
        print(len(tr), len(val))
        if os.path.exists(os.path.join(save_path, 'train{}.txt'.format(i))):
            os.remove(os.path.join(save_path, 'train{}.txt'.format(i)))
            os.remove(os.path.join(save_path, 'val{}.txt'.format(i)))

        for item in tr:
            # file_name = data_list[item].split('.')[0]
            file_name = data_list[item]
            with open(os.path.join(save_path, 'train{}.txt'.format(i)), 'a') as f:
                f.write(file_name)
                f.write('\n')

        for item in val:
            # file_name = data_list[item].split('.')[0]
            file_name = data_list[item]
            with open(os.path.join(save_path, 'val{}.txt'.format(i)), 'a') as f:
                f.write(file_name)
                f.write('\n')

def train_val(file_name_path, file_path, save_path):
    for i in range(1, 6):
        train_names = []
        val_names = []

        with open(os.path.join(file_name_path, 'train{}.txt'.format(i)), 'r') as f:
            for line in f.readlines():
                train_names.append(line.strip())
        with open(os.path.join(file_name_path, 'val{}.txt'.format(i)), 'r') as f:
            for line in f.readlines():
                val_names.append(line.strip())

        train_set = [item for item in os.listdir(file_path) if item[:-6] in train_names]
        val_set = [item for item in os.listdir(file_path) if item[:-6] in val_names]

        if os.path.exists(os.path.join(save_path, 'train{}.txt'.format(i))):
            os.remove(os.path.join(save_path, 'train{}.txt'.format(i)))
            os.remove(os.path.join(save_path, 'val{}.txt'.format(i)))

        for e in train_set:
            with open(os.path.join(save_path, 'train{}.txt'.format(i)), 'a') as f:
                f.write(e)
                f.write('\n')

        for e in val_set:
            with open(os.path.join(save_path, 'val{}.txt'.format(i)), 'a') as f:
                f.write(e)
                f.write('\n')
        print('ok!')

