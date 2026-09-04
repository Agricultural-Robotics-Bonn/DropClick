from rich import print
import os, yaml, json, random
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from pathlib import Path
import yaml



def get_dataset_registration_funcs( dataset_mapping_func, clear_cache=True ):
    '''
    Returns the dataset registration functions for each subset train, valid, eval
        in the format needed for detectron2
    !!! In detectron2, each dataset and each subset needs a specific registration function.
    !!! These MUST NOT have arguments.
        When registering multiple datasets, dynamic arguments will introduce bugs !!!!
    '''
    dataset_header, conversion_func_fpath, conversion_func_anno = dataset_mapping_func()
    def _subset_handling( subset ):
        ## LOADING IF EXISTS
        path_data, dataset = check_dataset_exists( dataset_header['annotation_file_dir'], dataset_header['name'], subset, clear_cache )
        ## ELSE CONVERSION
        if dataset is None:
            dataset = convert_dataset( dataset_header['dataset_yaml_dir'], dataset_header['annotation_file_dir'], subset, path_data, conversion_func_fpath, conversion_func_anno )
        return dataset 
    def register_func_train():  # must not have arguments !!!
        return _subset_handling( 'train' )
    def register_func_valid():  # must not have arguments !!!
        return _subset_handling( 'valid' )
    def register_func_eval():  # must not have arguments !!!
        return _subset_handling( 'eval' )
    return dataset_header, {
        'train': register_func_train,
        'valid': register_func_valid,
        'eval': register_func_eval,
    }


def check_dataset_exists( annotation_file_dir, name, subset, clear_cache=True ):
    # create a filepath to load/save the dataset_dict
    p = Path( annotation_file_dir )
    path_data = Path( os.getcwd() ) / Path('custom_datasets') / Path( name + '_' + subset + p.suffix )
    # load and return if alredy exists
    if path_data.is_file():
        if not clear_cache:
            print( f'loading previously converted dataset: {name}_{subset}...' )
            with open( path_data, 'r' ) as fp:
                try:
                    dataset_dicts = json.load( fp )
                except Exception as e:
                    print(f'WARNING: handling exception: {e}')
                    print('deleting previously converted dataset due to Error during opening...')   
                    os.remove( path_data )
                    print( f'converting dataset: {name}_{subset}...' )
                    return path_data, None
            return path_data, dataset_dicts
        else:
            print( f'deleting previously converted dataset: {name}_{subset}...' )
            os.remove( path_data )
            print( f'converting dataset: {name}_{subset}...' )
            return path_data, None
    else:
        print( f'converting dataset: {name}_{subset}...' )
        return path_data, None


def _get_category_info(categories, category_id):
    for cat in categories:
        if cat['id'] == category_id:
            return cat['name'], cat['supercategory'], cat['color']


def _check_segmentation(seg_list):
    if len(seg_list) <= 0:
        return None
    valid_seg_list = []
    for i in range(len(seg_list)):
        if len(seg_list[i]) >= 6:
            valid_seg_list.append(seg_list[i])
    if len(valid_seg_list) <= 0:
        return None
    return  valid_seg_list


def convert_dataset( dataset_yaml_dir, annotation_file_dir, subset, path_data, conversion_func_fpath, conversion_func_anno ):
    # start conversion
    with open(dataset_yaml_dir) as fp:
        dataset_img_subset_list = yaml.load(fp, Loader=yaml.FullLoader)
    # get img ids for subsets
    image_sets = dataset_img_subset_list["image_sets"]
    with open(annotation_file_dir) as fp:
        json_data = json.load(fp)
        _categories = []
        for i, category in enumerate( json_data["categories"] ):
            cat = {
                "id": category['id'],
                "name": category['name'],
                "supercategory": category['supercategory'],
                "color": category['color']
            }
            _categories.append(cat)
    dataset_dicts = []
    if not Path(path_data).parent.is_dir():
        os.makedirs(Path(path_data).parent)
    with open( path_data, 'w') as fp:
        for i in range(len(image_sets[subset])):
            record = {}
            matched = False
            for j in json_data["images"]:
                if j['id'] == image_sets[subset][i]:
                    record['file_name'] = conversion_func_fpath( j['path'] )
                    record['image_id'] = j['id']
                    record['width'] = j['width']
                    record['height'] = j['height']
                    matched = True
                    break
            if not matched:
                raise ValueError(f"WARNING: image_id {image_sets[subset][i]} from yaml not found in annotation json, skipping!")
            annotations = []
            for _, k in enumerate( json_data["annotations"] ):
                if k['image_id'] == image_sets[subset][i]:
                    seg_check = _check_segmentation(k['segmentation'])
                    if seg_check is None:
                        continue
                    category_name, supercategory, category_color = _get_category_info(_categories, k['category_id'])
                    obj = conversion_func_anno( k, seg_check, category_name, supercategory, category_color )
                    annotations.append(obj)
            record["annotations"] = annotations
            # record["categories"] = _categories
            dataset_dicts.append(record)
        json.dump(dataset_dicts, fp)
    print( '...done!' )
    return dataset_dicts









