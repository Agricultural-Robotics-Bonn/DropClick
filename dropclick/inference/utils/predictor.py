# Adapted by Patrick Zimmer from https://github.com/amitrana001/DynaMITe
import torch

class PredictorOneclick:

    def __init__(self, model):
        
        self.model = model
        self.images=None
        self.features = None
        self.mask_features = None
        self.multi_scale_features=None
        self.pred_masks = None

    def get_prediction(self, clicker):
        if self.features is None:
            (processed_results, outputs, self.images,
            _, self.features, self.mask_features,
            self.multi_scale_features, _,_) = self.model(clicker.inputs, max_timestamp=[0])

        else:
            out = self.model(clicker.inputs, self.images, clicker.num_insts,
                        self.features, self.mask_features,
                        self.multi_scale_features,
                        clicker.num_clicks_per_object,
                        clicker.fg_coords,
                        max_timestamp = [0]
                  )
            processed_results = out[0]
        pred_masks = processed_results[0]['instances'].pred_masks.to('cpu',dtype=torch.uint8)
        pred_scores = processed_results[0]['instances'].pred_scores
        pred_scores = pred_scores.to('cpu',dtype=torch.float) if pred_scores.numel()==0 or pred_scores[0] is not None else None
       
        return pred_masks, pred_scores
        
