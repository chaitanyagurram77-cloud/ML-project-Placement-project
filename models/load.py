import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv(r"C:\\Users\chait\OneDrive\Desktop\ML\placement_predict_50k Dataset1.csv")
    df.rename(
        columns={"target":"placement_status"},
        inplace=True
    )
    return df