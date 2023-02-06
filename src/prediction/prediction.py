import pandas as pd
import statistics
from tqdm import tqdm
import torch


def predict_fn(inference_model, test_dataloader, index_to_target, k):
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