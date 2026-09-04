# Adapted by Patrick Zimmer from https://github.com/amitrana001/DynaMITe

import numpy as np
import torch
import random
from functools import lru_cache
import cv2
from pycocotools import mask as coco_mask
from detectron2.data import transforms as T


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
       
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    if not is_train:
        augmentation = []
        if not isinstance(cfg.INPUT.IMAGE_SIZE, tuple) or cfg.INPUT.IMAGE_SIZE[0]==cfg.INPUT.IMAGE_SIZE[1]:
            print('INFO: doing test resizing )')
            augmentation.append(T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            ))
        else:
            print('WARNING: skipping test resizing because image size is not square')
        return augmentation

    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale,
            target_height=image_size if isinstance(image_size, int) else image_size[1],
            target_width=image_size if isinstance(image_size, int) else image_size[0]
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size) if isinstance(image_size, int) else (image_size[1], image_size[0]),
                        seg_pad_value = 0)
    ])

    return augmentation

def _point_candidates_dt(mask, k=1.7):
    mask = mask.astype(np.uint8)

    padded_mask = np.pad(mask, ((1, 1), (1, 1)), 'constant')
    dt = cv2.distanceTransform(padded_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]

    candidates = np.argwhere(dt > (dt.max()/k))
    indices = np.random.randint(0,candidates.shape[0])
    return candidates[indices]

@lru_cache(maxsize=None)
def generate_probs(max_num_points, gamma):
    probs = []
    last_value = 1
    for i in range(max_num_points):
        probs.append(last_value)
        last_value *= gamma

    probs = np.array(probs)
    probs /= probs.sum()

    return probs

import copy
def get_clicks_coords(masks, max_num_points=6, first_click_center=True, all_masks=None, t= 0, keypoints=None):

    masks = np.asarray(masks).astype(np.uint8)
    all_masks = np.asarray(all_masks)

    I, H, W = masks.shape
    num_clicks_per_object = [1]*I  # always 1 for us
    fg_coords_list = []
    keypoints = [ None for _ in range(len(masks)) ] if keypoints is None else keypoints
    for i, (_m,_k) in enumerate(zip(masks, keypoints)):
        coords = []
  
        click_coords = getOneclickCoords( _m, is_eval=False, keypoint=_k[:2][[1,0]].numpy() if (_k is not None and _k.sum()) else None )
        
        coords.append([click_coords[0], click_coords[1], t])
        fg_coords_list.append(coords)
        
    return num_clicks_per_object, fg_coords_list


def getOneclickCoords( gt, is_eval=False, keypoint=None ):
    if keypoint is not None and sum(keypoint)!=0:
        click_coords = keypoint
        if not checkClickValidity( click_coords, gt ):
            click_coords = generateCenterOfMassClick( gt )
            if not checkClickValidity( click_coords, gt ):
                click_coords = generateErosionClick( gt )
    else:
        click_coords = generateCenterOfMassClick( gt )
        if not checkClickValidity( click_coords, gt ):
            click_coords = generateErosionClick( gt )
    ## click randomization
    if not is_eval:
        click_coords = np.array( addCenterNoise( gt, list(click_coords)) )
    return click_coords


def generateCenterOfMassClick( gt ):
    if not isinstance( gt, np.ndarray ):
        gt = gt.numpy()
    # calculate moments of binary image
    M = cv2.moments( gt, binaryImage=True )
    # calculate x,y coordinate of center
    if M["m00"] == 0:
        M["m00"] = .000000000000001
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    center = (cY, cX)
    return center


def checkClickValidity( click, gt ):
    '''
        click has to be format (Y,Z)
    '''
    if not isinstance( gt, np.ndarray ):
        gt = gt.numpy()
    y = click[0]
    x = click[1]
    if ( y>=0 and y<gt.shape[0] and x>=0 and x<gt.shape[1] ):  # double checking if click is inside image, otherwise indexing in next line will fail
        return True if gt[y,x]==1 else False
    else:
        return False


from scipy.ndimage.morphology import binary_dilation, binary_erosion
def generateErosionClick( gt ):
    '''
        determines click position based on the last leftover pixel(s) when completely eroding the mask
        if there are multiple ones in last iteration, a random one is chosen
    '''
    if not isinstance( gt, np.ndarray ):
        gt = gt.numpy()
    while gt.sum() > 0:
        nz = gt.nonzero()
        gt = binary_erosion( gt, iterations=1 )
    i = random.randint( 0, len(nz[0])-1 )
    click = ( nz[0][i], nz[1][i] )
    return click


from scipy.stats import multivariate_normal
def addCenterNoise( gt, click ):
    '''
        Adding center noise in a realistic way using a gaussian distribution.
        TODO This is relatively slow, further optimize?
            - slowest leftover step is calling the distribution function to create the probability matrix
            - also np.where() calls
    '''

    if not isinstance( gt, np.ndarray ):
        gt = gt.numpy()
    
    # get variable noiserange
    obj_ncols = np.count_nonzero( gt, axis=0 ).astype(bool).sum()
    obj_nrows = np.count_nonzero( gt, axis=1 ).astype(bool).sum()
    noiserange = .2 * max( [obj_ncols, obj_nrows] )
    
    # set up the gaussian distribution
    sigma = noiserange / 3
    mean = np.array( [ click[1], click[0] ] )
    cov = np.array( [ [sigma, 0], [0, sigma] ] )
    distr = multivariate_normal( cov=cov, mean=mean )

    # make an array of probability values
    w = gt.shape[1]
    h = gt.shape[0]
    x = np.linspace(0, w, num=w, endpoint=False)
    y = np.linspace(0, h, num=h, endpoint=False)
    X, Y = np.meshgrid(x,y)
    pos = np.dstack( (X,Y) )
    pd = distr.pdf(pos)

    # actually limit it to the desired noiserange to avoid huge outliers
    # with tested settings, probabilities become so low they get automatically rounded to 0 after +/-20px
    prob = distr.pdf( [ click[1]-noiserange, click[0] ] )
    pd = np.where( pd>=prob, pd, 0 )

    # make sure to only pick points that are valid clicks (inside gt):
    pd_masked = np.where( gt==1, pd, 0 )
    if np.all(pd_masked==0):  # return None if no valid point can be found
      click_new = (None,None)
      return click_new
    
    # pick a point
    pd_masked = pd_masked / np.sum(pd_masked)  # scale back so probs sum to 1
    pd_masked_flattened = pd_masked.flatten()
    idcs = np.arange( pd_masked_flattened.shape[0] )
    pick = np.random.choice( idcs, p=pd_masked_flattened )
    click_new = np.unravel_index( pick, pd_masked.shape )
    
    return click_new
