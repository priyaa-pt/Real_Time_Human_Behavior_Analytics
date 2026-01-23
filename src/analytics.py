import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# CSV PATH

CSV_PATH = os.path.join("logs", "tracking_data.csv")


def main():
    # Check if CSV exists
    if not os.path.exists(CSV_PATH):
        print("CSV file not found.")
        return

    # 
    # LOAD CSV (MUST COME FIRST)
    # 
    df = pd.read_csv(CSV_PATH)

    print("Data loaded successfully\n")

    # BASIC DATA CHECKS

    print("First 5 rows:")
    print(df.head(), "\n")

    print("Total rows:", len(df))
    print("Unique person IDs:", df["person_id"].nunique())

    start_time = df["timestamp"].min()
    end_time = df["timestamp"].max()
    duration = end_time - start_time

    print(f"Data duration (seconds): {round(duration, 2)}")

  
    # FOOTFALL

    footfall = df["person_id"].nunique()
    print("\nEstimated Footfall:", footfall)

    # DWELL TIME (APPROXIMATE)

    dwell_times = {}

    for person_id, group in df.groupby("person_id"):
        first_seen = group["timestamp"].min()
        last_seen = group["timestamp"].max()
        dwell_times[person_id] = last_seen - first_seen

    print("\nDwell Time per Person (seconds):")
    for pid, dwell in dwell_times.items():
        print(f"Person {pid}: {round(dwell, 2)} sec")

    avg_dwell = sum(dwell_times.values()) / len(dwell_times)
    print(f"\nAverage Dwell Time: {round(avg_dwell, 2)} sec")

    # HEATMAP (SPATIAL DENSITY)
    # 

    x = df["x"].values
    y = df["y"].values

    # Create heatmap using 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=50
    )

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(
        heatmap.T,
        origin="lower",
        cmap="hot"
    )
    plt.colorbar(label="Presence Density")
    plt.title("Human Presence Heatmap")
    plt.xlabel("X position")
    plt.ylabel("Y position")

    # Save heatmap
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/heatmap.png")
    plt.close()

    print("\nHeatmap saved to outputs/heatmap.png")

if __name__ == "__main__":
    main()
