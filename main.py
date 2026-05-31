import liffile
import numpy as np
import matplotlib.pyplot as plt
import trackpy as tp


def main():
    file_path = "/home/hugo/Personal/biology/M1F1_Hpo-RNAi_VDRC_x_sens-gal4_Arr2RI-A_14d_6r5b4g.lif"

    with liffile.LifFile(file_path) as lif:
        # Shape is (Z, Channel, X, Y)
        image_data = lif.images[1].asarray()

    # Isolate the green channel
    green_volume = image_data[:, 1, :, :]

    # Normalize
    if green_volume.dtype != np.uint8 and green_volume.dtype != np.uint16:
        green_volume = (green_volume / np.max(green_volume) * 255).astype(np.uint8)

    # --- TRACKING WITH TRACKPY ---
    print("Locating features...")
    f = tp.batch(green_volume, diameter=21, minmass=500, invert=False)

    print("Linking trajectories...")
    t = tp.link(f, search_range=10, memory=1)

    t_filtered = tp.filter_stubs(t, 3)

    # --- VISUALIZATION ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(green_volume[0], cmap="gray")
    tp.plot_traj(t_filtered, ax=ax, superimpose=green_volume[0])
    ax.set_title("Feature Tracking Across Z-Stack (Green Channel)")
    plt.show()


if __name__ == "__main__":
    main()
