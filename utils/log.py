import os
import matplotlib.pyplot as plt
import pandas as pd


def log_csv(path, epoch, loss, loss_geo, loss_color):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(f"{epoch},{loss},{loss_geo},{loss_color}\n")


def plot_csv(path):
    df = pd.read_csv(path, header=None)
    df.columns = ["iter", "loss"]
    plt.plot(df["iter"], df["loss"])
    plt.savefig("loss.png")
