from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd

class LocDataset(Dataset):
    def __init__(self, data_df):
        super().__init__()
        self.data_df = data_df

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        image_name = self.data_df.iloc[index]["filename"]
        label = self.data_df.iloc[index]["class_num"]
        box = (
            self.data_df.iloc[index][["xmin", "ymin", "xmax", "ymax"]]
            .astype("float32")
            .values
        )
        image = cv2.imread("dataset/images/" + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255
        image = np.transpose(image, (2, 0, 1))
        return (image, label, box)
