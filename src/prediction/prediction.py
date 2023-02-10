import pandas as pd
import statistics
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from typing import Any


def load_weights(model, weights):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load state_dict from ckpt with 'model.' and 'loss_fn.W' key deleted
    state_dict = torch.load(weights.name, map_location=torch.device(device))['state_dict']
    state_dict = {k.replace('model.', '') : v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    return model


def predict_fn(
        inference_model: Any, 
        test_dataloader: DataLoader, 
        index_to_target: dict, 
        k: int,
    ) -> tuple:
    filenames, distances, preds, top1s, modes = [], [], [], [], []
    with torch.no_grad():
        for image, filename in tqdm(test_dataloader):
            # distance & index of dataset
            distance, index = inference_model.get_nearest_neighbors(image, k=k)
            distance = distance.cpu().numpy()

            # index => target
            pred = [index_to_target[int(k)] for k in index.cpu().numpy().squeeze()]

            # top1 pred
            top1 = pred[0]

            # mode of topK
            mode = statistics.mode(pred)
            
            filenames.extend(filename)
            distances.extend(distance)
            preds.append(pred)
            top1s.append(top1)
            modes.append(mode)
        
        df = pd.DataFrame({
            "filename": filenames,
            "distance": distances,
            f"preds_top{k}": preds,
            "pred_top1": top1s,
            f"pred_mode@top{k}": modes
        })

        df_top1 = pd.DataFrame({
            "filename": filenames,
            "pred_top1": top1s,
        })

        df_mode = pd.DataFrame({
            "filename": filenames,
            f"pred_mode@top{k}": modes,
        })

    return df, df_top1, df_mode