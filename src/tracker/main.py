import napari
from magicgui import magicgui
import trackpy as tp
import numpy as np
import pandas as pd
import liffile
from scipy.spatial import cKDTree
from napari.utils.colormaps import Colormap
from napari.utils.color import transform_color
from napari.qt.threading import thread_worker

from collections.abc import Sequence
from pathlib import Path


da_raw = None
channel_map = {}


def main():
    viewer = napari.Viewer()

    kwargs = {
        "face_color": "transparent",
        "ndim": 3,
        "size": 15,
        "opacity": 0.5,
    }
    layer_rh4 = viewer.add_points(name="Rh4 Points", border_color="orange", **kwargs)
    layer_rh5 = viewer.add_points(name="Rh5 Points", border_color="orange", **kwargs)
    layer_rh6 = viewer.add_points(name="Rh6 Points", border_color="orange", **kwargs)
    layer_roi = viewer.add_labels(
        np.zeros((10, 10), dtype=np.int32),
        name="1. Draw ROI Here",
        opacity=0.4,
    )

    def get_image_choices(widget):
        """Reads the current path from the widget and returns the valid images."""
        if widget.parent is None:
            return [("Select a LIF file first", 0)]
        path = widget.parent.path.value
        if path and Path(path).is_file():
            with liffile.LifFile(path) as lif:
                return [(f"[{i}] {img.name}", i) for i, img in enumerate(lif.images)]
        return [("Select a LIF file first", 0)]

    @magicgui(
        path={"label": "1. Choose LIF file:", "filter": "*.lif"},
        image_index={"label": "2. Select Image:", "choices": get_image_choices},
        color_profile={
            "choices": [
                "Profile 1 (Rh4: Green, Rh6: Red, Rh5: Blue)",
                "Profile 2 (Rh4: Blue, Rh6: Green, Rh5: Red)",
            ],
            "label": "3. Colors:",
        },
        call_button="Load Data into Viewer",
    )
    def filepicker(
        path: Path,
        image_index: int,
        color_profile: str = "Profile 1 (Rh4: Green, Rh6: Red, Rh5: Blue)",
    ):
        global da_raw, channel_map
        with liffile.LifFile(path) as lif:
            da_raw = lif.images[image_index].asxarray()

            for layer_name in ["Rh4", "Rh6", "Rh5"]:
                if layer_name in viewer.layers:
                    viewer.layers.remove(layer_name)

            try:
                c_metadata = da_raw.coords["C"].values
                channel_map = {
                    "Rh4": np.where(c_metadata == "ALEXA 488")[0][0],
                    "Rh6": np.where(c_metadata == "ALEXA 555")[0][0],
                    "Rh5": np.where(c_metadata == "ALEXA 647")[0][0],
                }
            except Exception:
                print("Warning: Channel metadata not found. Using default order.")
                channel_map = {"Rh4": 0, "Rh6": 1, "Rh5": 2}

            names_in_order = [""] * 3
            for target, idx in channel_map.items():
                names_in_order[idx] = target

            # 4. Map the requested colors to those targets
            if "Profile 1" in color_profile:
                color_map = {"Rh4": "green", "Rh6": "red", "Rh5": "blue"}
            else:
                color_map = {"Rh4": "blue", "Rh6": "green", "Rh5": "red"}

            # Create a list of colors that perfectly matches the physical channel order
            colors_in_order = [color_map[target] for target in names_in_order]

            # Reset other spatial layers before add_image below
            layer_rh4.data = np.empty((0, 3))
            layer_rh5.data = np.empty((0, 3))
            layer_rh6.data = np.empty((0, 3))

            size_y, size_x = da_raw.shape[-2], da_raw.shape[-1]
            layer_roi.data = np.zeros((size_y, size_x), dtype=np.int32)

            # 5. The elegant single-call plot!
            # We pass da_raw.values directly because we already made it a safe RAM block
            viewer.add_image(
                da_raw.values,
                channel_axis=len(da_raw.shape) - 3,
                name=names_in_order,
                colormap=colors_in_order,
                blending="additive",
                rendering="mip",
            )

            # 6. Formatting and resets
            viewer.dims.set_current_step(0, 0)  # Force z=0 at first

            for layer_name in ["Rh4", "Rh6", "Rh5"]:
                viewer.layers.move(viewer.layers.index(layer_name), 0)

    @filepicker.path.changed.connect
    def _update_dropdown(new_path: Path):
        filepicker.image_index.reset_choices()

    @magicgui(call_button="Flip Z-Axis (Up/Down)")
    def flip_z_axis_widget():
        global da_raw, channel_map

        if da_raw is None:
            print("Please load an image first.")
            return

        print("\n--- Flipping Z-Axis ---")

        if "Z" in da_raw.dims:
            da_raw = da_raw.reindex(Z=da_raw.Z[::-1])

        c_axis = len(da_raw.shape) - 3

        for target_name, c_index in channel_map.items():
            if target_name in viewer.layers:
                if c_axis == 1:
                    viewer.layers[target_name].data = da_raw.values[:, c_index, :, :]
                else:
                    viewer.layers[target_name].data = da_raw.values[c_index, :, :]

        # Frame indices refer to the old Z order, so any prior detections/tracks
        # are now stale and would render at the wrong slice
        layer_rh4.data = np.empty((0, 3))
        layer_rh5.data = np.empty((0, 3))
        layer_rh6.data = np.empty((0, 3))
        if "Trajectories" in viewer.layers:
            viewer.layers.remove("Trajectories")

        viewer.dims.set_current_step(0, 0)
        print("Success: Z-axis inverted. Cleared stale detections/tracks.")

    @thread_worker
    def _run_detection(channels_to_process, diameter, minmass, threshold):
        c_axis = len(da_raw.shape) - 3

        # Process each selected channel sequentially, off the UI thread
        for ch in channels_to_process:
            print(
                f"Processing {ch} channel (D={diameter}, Mass={minmass}, Thresh={threshold})..."
            )

            # Instantly get the correct physical index from our global dictionary
            c_index = channel_map[ch]

            # Safely extract for 3D or 4D - always the full stack, so users can
            # tune parameters while looking at any Z without losing points on
            # other frames. Which Z's actually get used is decided later, at
            # the Colocalize & Backtrack step.
            if c_axis == 1:
                volume = da_raw.values[:, c_index, :, :]
            else:
                volume = da_raw.values[c_index, :, :]

            # Run Trackpy
            f = tp.batch(
                volume,
                diameter=diameter,
                minmass=minmass,
                threshold=threshold,
                processes=1,
            )

            if not f.empty:
                yield ch, f[["frame", "y", "x"]].values
            else:
                yield ch, np.empty((0, 3))

    @magicgui(
        call_button="Predict All Frames",
        target_channel={"choices": ["All", "Rh4", "Rh6", "Rh5"], "label": "Channel"},
        diameter={"step": 2, "min": 3, "max": 51},
        minmass={"min": 100, "max": 10000, "step": 100},
        threshold={"min": 0, "max": 255},
    )
    def predict_all_frames_widget(
        target_channel: str = "All",
        diameter: int = 13,
        minmass: int = 700,
        threshold: int = 30,
    ):
        print(f"\n--- Starting Batch Prediction: {target_channel} (all Z) ---")

        target_layers = {
            "Rh4": layer_rh4,
            "Rh6": layer_rh6,
            "Rh5": layer_rh5,
        }

        if target_channel == "All":
            channels_to_process = ["Rh4", "Rh6", "Rh5"]
        else:
            channels_to_process = [target_channel]

        # Run detection on a background thread so the UI stays responsive
        def _update_layer(result):
            ch, new_coords = result
            target_layers[ch].data = new_coords
            if len(new_coords):
                print(
                    f" -> Success: Rendered {len(new_coords)} points to the {ch} layer."
                )
            else:
                print(f" -> No features found. Cleared the {ch} layer.")

        def _on_finished():
            predict_all_frames_widget.call_button.enabled = True
            print("--- Batch Prediction Complete! ---\n")

        predict_all_frames_widget.call_button.enabled = False
        worker = _run_detection(channels_to_process, diameter, minmass, threshold)
        worker.yielded.connect(_update_layer)
        worker.finished.connect(_on_finished)
        worker.start()

    @magicgui(
        call_button="Colocalize & Backtrack",
        track_radius={"label": "Max Backtrack Drift (px)", "max": 50},
        merge_radius={
            "label": "Colocalization Radius (px)",
            "max": 20,
            "tooltip": "Merge Rh5/Rh6 dots closer than this into one cell",
        },
        memory={
            "label": "Memory (frames)",
            "max": 5,
            "tooltip": "Max consecutive frames a cell can go undetected while backtracking",
        },
    )
    def link_and_track_widget(
        track_radius: int = 15,
        merge_radius: float = 5.0,
        memory: int = 1,
    ):
        print("\n--- Colocalizing at the last Z and backtracking to origin ---")

        # ---------------------------------------------------------
        # 1. Harvest raw per-channel detections
        # ---------------------------------------------------------
        df_list = []
        for layer, target_name in [
            (layer_rh4, "Rh4"),
            (layer_rh6, "Rh6"),
            (layer_rh5, "Rh5"),
        ]:
            if len(layer.data) > 0:
                df = pd.DataFrame(layer.data, columns=["Z", "Y", "X"])
                df["target"] = target_name
                df_list.append(df)

        if not df_list:
            print("No points found in any layer. Run prediction first.")
            return

        df_all = pd.concat(df_list, ignore_index=True)
        df_all["Z"] = df_all["Z"].round().astype(int)

        # ---------------------------------------------------------
        # 2. Colocalize Rh5/Rh6 at the last useful Z, inside the ROI
        # ---------------------------------------------------------
        # Whatever Z the user is currently viewing is the last useful frame -
        # navigate there (where Rh5/Rh6 are clearest) before clicking this.
        # Detection already ran on the full stack, so this is purely an
        # analysis-time cutoff; deeper frames are simply never looked at.
        z_max = int(viewer.dims.current_step[0])
        end_df = df_all[
            (df_all["Z"] == z_max) & (df_all["target"].isin(["Rh5", "Rh6"]))
        ]

        if np.any(layer_roi.data):
            roi_mask = layer_roi.data.astype(bool)
            yy = np.clip(end_df["Y"].round().astype(int), 0, roi_mask.shape[0] - 1)
            xx = np.clip(end_df["X"].round().astype(int), 0, roi_mask.shape[1] - 1)
            end_df = end_df[roi_mask[yy, xx]]
            print(f"Seeding from Z={z_max}, restricted to the drawn ROI.")
        else:
            print(f"Seeding from Z={z_max}. No ROI drawn - using the entire slice.")

        seeds = []
        if not end_df.empty:
            pts = end_df[["Y", "X"]].values
            labels = end_df["target"].values
            tree = cKDTree(pts)
            clusters = tree.query_ball_tree(tree, r=merge_radius)

            processed = set()
            for i, neighbors in enumerate(clusters):
                if i in processed:
                    continue
                processed.update(neighbors)

                cluster_targets = set(labels[neighbors])
                if "Rh5" in cluster_targets and "Rh6" in cluster_targets:
                    phenotype = "Rh5+Rh6"
                elif "Rh5" in cluster_targets:
                    phenotype = "Rh5"
                else:
                    phenotype = "Rh6"

                seeds.append(
                    {
                        "y": float(np.mean(pts[neighbors, 0])),
                        "x": float(np.mean(pts[neighbors, 1])),
                        "phenotype": phenotype,
                        # Raw detections making up this cluster, for the
                        # point-layer cleanup below
                        "z_points": [
                            (pts[n, 0], pts[n, 1], labels[n]) for n in neighbors
                        ],
                    }
                )

        if not seeds:
            print("No Rh5/Rh6 points found at the last Z (inside the ROI).\n")
            return

        # ---------------------------------------------------------
        # 3. Backtrack each seed frame-by-frame down to Z=0
        # ---------------------------------------------------------
        # Backtracking is unrestricted by the ROI - a cell's origin (an Rh4
        # point, or nothing observed = Rh3) can lie outside the footprint
        # drawn at the last Z, since cells drift as they differentiate.
        points_by_z = {z: g[["Y", "X"]].values for z, g in df_all.groupby("Z")}
        targets_by_z = {z: g["target"].values for z, g in df_all.groupby("Z")}

        # Collect the raw per-channel points that actually belong to a
        # surviving trajectory, so the Points layers can be cleaned up below
        channel_points = {"Rh4": [], "Rh5": [], "Rh6": []}

        results = []
        for seed in seeds:
            cur_y, cur_x = seed["y"], seed["x"]
            path = [(z_max, cur_y, cur_x)]
            missed = 0
            origin = "Rh3"  # default: trail never confirms an Rh4 point

            for y, x, target in seed["z_points"]:
                channel_points[target].append((z_max, y, x))

            for z in range(z_max - 1, -1, -1):
                pts = points_by_z.get(z)
                found = False
                if pts is not None and len(pts):
                    dists = np.hypot(pts[:, 0] - cur_y, pts[:, 1] - cur_x)
                    j = int(np.argmin(dists))
                    if dists[j] <= track_radius:
                        cur_y, cur_x = pts[j]
                        path.append((z, cur_y, cur_x))
                        missed = 0
                        found = True
                        target = targets_by_z[z][j]
                        channel_points[target].append((z, cur_y, cur_x))
                        if z == 0 and target == "Rh4":
                            origin = "Rh4"

                if not found:
                    missed += 1
                    if missed > memory:
                        break  # trail went cold before reaching Z=0 -> Rh3

            results.append(
                {
                    "phenotype": seed["phenotype"],
                    "label": f"{origin} -> {seed['phenotype']}",
                    "path": path,
                }
            )

        # Drop every detection that isn't part of a surviving trajectory
        for target_name, layer in (
            ("Rh4", layer_rh4),
            ("Rh5", layer_rh5),
            ("Rh6", layer_rh6),
        ):
            coords = channel_points[target_name]
            layer.data = np.array(coords) if coords else np.empty((0, 3))

        # ---------------------------------------------------------
        # 4. Report counts
        # ---------------------------------------------------------
        counts = pd.Series([r["label"] for r in results]).value_counts()
        categories = [
            "Rh4 -> Rh5",
            "Rh4 -> Rh6",
            "Rh4 -> Rh5+Rh6",
            "Rh3 -> Rh5",
            "Rh3 -> Rh6",
            "Rh3 -> Rh5+Rh6",
        ]

        print("\n" + "=" * 40)
        print(f"📊 FINAL COUNT (colocalized at Z={z_max}, backtracked to origin)")
        print("=" * 40)
        for cat in categories:
            print(f" -> {cat:<16}: {counts.get(cat, 0)}")
        print("=" * 40 + "\n")

        # ---------------------------------------------------------
        # 5. Push backtracked paths to the viewer as Tracks
        # ---------------------------------------------------------
        master_color_dict = {
            "Rh4 -> Rh5": "cyan",
            "Rh4 -> Rh6": "magenta",
            "Rh4 -> Rh5+Rh6": "yellow",
            "Rh3 -> Rh5": "steelblue",
            "Rh3 -> Rh6": "orchid",
            "Rh3 -> Rh5+Rh6": "khaki",
        }

        if "Trajectories" in viewer.layers:
            viewer.layers.remove("Trajectories")

        track_rows = [
            {
                "particle": idx,
                "frame": z,
                "y": y,
                "x": x,
                "Track_Phenotype": r["label"],
            }
            for idx, r in enumerate(results)
            for (z, y, x) in r["path"]
        ]
        t_display = (
            pd.DataFrame(track_rows)
            .sort_values(["particle", "frame"])
            .reset_index(drop=True)
        )

        active_phenos = t_display["Track_Phenotype"].unique().tolist()
        N = len(active_phenos)

        if N == 1:
            pheno_float_list = [0.5] * len(t_display)
            c = transform_color(master_color_dict[active_phenos[0]])[0]
            custom_cmap = Colormap(colors=[c, c], name="exact_cmap")
        else:
            pheno_to_float = {
                pheno: float(i) / (N - 1) for i, pheno in enumerate(active_phenos)
            }
            pheno_float_list = [
                pheno_to_float[p] for p in t_display["Track_Phenotype"]
            ]

            rgba_colors = [
                transform_color(master_color_dict[k])[0] for k in active_phenos
            ]
            controls = np.linspace(0, 1, N)
            custom_cmap = Colormap(
                colors=rgba_colors, controls=controls, name="exact_cmap"
            )

        track_data = t_display[["particle", "frame", "y", "x"]].values
        track_props = {
            "Track_Phenotype": t_display["Track_Phenotype"].tolist(),
            "color_val": pheno_float_list,
        }

        viewer.add_tracks(
            track_data,
            properties=track_props,
            color_by="color_val",
            colormaps_dict={"color_val": custom_cmap},
            name="Trajectories",
            tail_width=4,
            tail_length=z_max + 5,
        )

    @magicgui(
        call_button="Save Points to CSV",
        save_path={"label": "Save Location:", "mode": "w", "filter": "*.csv"},
    )
    def export_points_widget(save_path: Path = Path("raw_points_export.csv")):
        print(f"\n--- Exporting Point Data ---")

        # Make sure a valid path was provided
        if save_path is None:
            print("Export canceled: No save path selected.")
            return

        df_list = []

        # Harvest the data directly from the active layers
        for layer, target_name in [
            (layer_rh4, "Rh4"),
            (layer_rh6, "Rh6"),
            (layer_rh5, "Rh5"),
        ]:
            # Ensure the layer exists and has data before trying to extract it
            if layer is not None and len(layer.data) > 0:
                df = pd.DataFrame(layer.data, columns=["Z", "Y", "X"])
                df["Channel"] = target_name
                df_list.append(df)

        if df_list:
            # Combine all channels
            df_export = pd.concat(df_list, ignore_index=True)

            # Enforce the .csv extension just in case it was forgotten in the UI
            if save_path.suffix != ".csv":
                save_path = save_path.with_suffix(".csv")

            df_export.to_csv(save_path, index=False)

            print(f"Success! Saved {len(df_export)} total points to:")
            print(f" -> {save_path.absolute()}")
            print("\nPoint Counts by Channel:")
            print(df_export["Channel"].value_counts().to_string())
            print("----------------------------\n")
        else:
            print("Export failed: No points found in any of the layers.")

    # Swap out or add this widget to the sidebar
    viewer.window.add_dock_widget(filepicker, name="1. Load File", area="right")
    viewer.window.add_dock_widget(
        flip_z_axis_widget, name="2. Orientation", area="right"
    )
    viewer.window.add_dock_widget(
        predict_all_frames_widget, name="2. Point detection", area="right"
    )
    viewer.window.add_dock_widget(
        link_and_track_widget, name="3. Colocalize & Backtrack", area="right"
    )
    viewer.window.add_dock_widget(
        export_points_widget, name="4. Export Results", area="right"
    )

    napari.run()


if __name__ == "__main__":
    main()
