

from detectron2.structures import BoxMode



'''
    Here we use a dataset specific mapping function that is used in ./base.py
        for .json structure conversion and splitting into train, valid, eval according to the .yaml.
    Datasets are then automatically registered to detectron2 in ./libs.py
    For format, check the ones that are implemented here in this directory.
    Quick Description:
        1. create a module, it's name will be the dataset name.
            1.1. splits will automatically be created, e.g., for sb20.py it will create 3 datasets sb20_train, sb20_valid and sb20_eval
            1.2. those can easily be selected in your config
        2. define the mapping magic for paths, classes etc., add thing_classes to dataset_header
        3. add the dataset to the header in ./libs.py   
'''



def dataset_mapping():
    '''
        Use this to configure your dataset (bup20_single).
        This is called within the actual dataset registration fuctions, once for each subset
            (train, valid, eval), which need to have seperate registration functions.
    '''
    dataset_header = {
        'name': __name__.split('.')[-1], # do not change
        'dataset_yaml_dir': './datasets/CKA_sweet_pepper_2020_summer_subset05imgs/CKA_sweet_pepper_2020_summer_subset05imgs.yaml',
        'annotation_file_dir': './datasets/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json',
        'thing_classes': ['pepper'],
    }    
    ## file path manipulation
    def conversion_func_fpath( fpath ):
        fpath = fpath.replace( '/datasets/', './datasets/' )
        return fpath
    ## annotation manipulation
    def conversion_func_anno( k, seg_check, category_name, supercategory, category_color ):
        obj = {
            "id": k['id'],
            "image_id": k['image_id'],
            "area": k['area'],
            "bbox": k['bbox'],
            "bbox_mode": BoxMode.XYWH_ABS,
            "keypoints": k['keypoints'] if 'keypoints' in k.keys() else [0,0,0],
            "segmentation": seg_check,
            "iscrowd": k['iscrowd'],
            ## just 1 category (pepper)
            "category_id": 0,
            "category_name": 'pepper',
            "supercategory": 'pepper',
            "color": '#00FF00',
        }
        return obj
    return dataset_header, conversion_func_fpath, conversion_func_anno


