

import sys
import importlib
import pathlib

from detectron2.data import MetadataCatalog, DatasetCatalog

from .base import get_dataset_registration_funcs


datasets = [
    'bup20_single',
    'bup20_single_subset05imgs',
    'sb20_single',
    'sb20_single_subset05imgs',
]



def _register_dataset( dataset_name, register_func, thing_classes ):
    DatasetCatalog.register(dataset_name, register_func)
    MetadataCatalog.get(dataset_name).set(
        thing_classes=thing_classes,
        evaluator_type="coco",
        thing_dataset_id_to_contiguous_id = {i:i for i in range(len(thing_classes))},
    )
    # cat_id_map is necessary for thing_dataset_id_to_contiguous_id argument obligatory in some models.
    # mapping equal ids here cause we have contigous ids in the converted datasets already.


def register_custom_datasets(clear_cache=True):
    ## register all given datasets
    for dataset in datasets:
        module = importlib.import_module( f'.{dataset}', package=__package__ )
        dataset_mapping_func = getattr( module, f'dataset_mapping' )
        dataset_header, register_funcs = get_dataset_registration_funcs( dataset_mapping_func, clear_cache=clear_cache )
        thing_classes = dataset_header['thing_classes']
        for subset in [ 'train', 'valid', 'eval' ]:
            dataset_name = f'{dataset}_{subset}'
            register_func = register_funcs[subset]
            _register_dataset( dataset_name, register_func, thing_classes )
    return DatasetCatalog, MetadataCatalog

print( 'registering custom datasets...' )
## NOTE: choose whether or not to clear the dataset cache
register_custom_datasets(clear_cache=True); print( '\n\tWARNING: dataset cache clearing is turned ON. change this setting for performance unless applying dataloader related changes (hardcoded in ./dropclick/data/custom_dataset_registration/libs.py).\n' )
# register_custom_datasets(clear_cache=False); print( '\n\tWARNING: dataset cache clearing is turned OFF by default for performance reasons. Need to change this setting when applying dataloader related changes (hardcoded in ./dropclick/data/custom_dataset_registration/libs.py)!\n' )
print( '...done.' )
