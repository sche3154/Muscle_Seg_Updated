import os

def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def list_dir(directory):
    visible_files = []
    for file in os.listdir(directory):
        if not file.startswith('.'):
            visible_files.append(file)
    return visible_files

def tms_sample_list(isTrain= True):
    root = '/home/sheng/datasets/muscle/generated_results/all_data'
    if isTrain:
        subs = ['3','4','5','6','7','9','12','14','15']
    else:
        subs = ['10','11','15']

    all_data = list_dir(os.path.join(root))
    for data in all_data:
        if data.startswith('.'):
            all_data.remove(data)
        if 'noisy' in data:
            all_data.remove(data)

    sample_list = []
    for sub in subs:
        prefix = sub + '_'
        for data in all_data:
            if prefix in data and 'mask' not in data:
                sample_list.append(data.split('.')[0])

    return sample_list