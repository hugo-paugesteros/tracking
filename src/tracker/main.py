import napari
from magicgui import magicgui
import trackpy as tp
import numpy as np
import pandas as pd
import liffile
from scipy.spatial import cKDTree
from napari.utils.colormaps import Colormap
from napari.utils.color import transform_color
from skimage.draw import polygon2mask

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
    layer_roi = viewer.add_shapes(
        name="1. Draw ROI Here",
        shape_type="polygon",
        edge_width=5,
        edge_color="yellow",
        face_color="transparent",
        ndim=3,
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

            # Reset data
            layer_rh4.data = np.empty((0, 3))
            layer_rh5.data = np.empty((0, 3))
            layer_rh6.data = np.empty((0, 3))
            layer_roi.data = []

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

        viewer.dims.set_current_step(0, 0)
        print("Success: Z-axis inverted.")

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
        global da_raw, channel_map
        print(f"\n--- Starting Batch Prediction: {target_channel} ---")

        # 1. Generate the ROI Mask
        size_y, size_x = da_raw.shape[-2], da_raw.shape[-1]

        # Default mask is all True (entire image)
        roi_mask = np.ones((size_y, size_x), dtype=bool)

        # If the user drew shapes, build a custom mask
        if len(layer_roi.data) > 0:
            print("Applying custom ROI from Shapes layer...")
            roi_mask = np.zeros((size_y, size_x), dtype=bool)
            for shape_data in layer_roi.data:
                # Napari shape coords in 3D are [Z, Y, X]. We extract just [Y, X]
                yx_coords = shape_data[:, -2:]
                shape_mask = polygon2mask((size_y, size_x), yx_coords)
                roi_mask = roi_mask | shape_mask  # Combine multiple shapes if drawn
        else:
            print("No ROI drawn. Processing the entire image...")

        target_layers = {
            "Rh4": layer_rh4,
            "Rh6": layer_rh6,
            "Rh5": layer_rh5,
        }

        if target_channel == "All":
            channels_to_process = ["Rh4", "Rh6", "Rh5"]
        else:
            channels_to_process = [target_channel]

        c_axis = len(da_raw.shape) - 3

        # Process each selected channel sequentially
        for ch in channels_to_process:
            print(
                f"Processing {ch} channel (D={diameter}, Mass={minmass}, Thresh={threshold})..."
            )

            # Instantly get the correct physical index from our global dictionary
            c_index = channel_map[ch]
            active_layer = target_layers[ch]

            # Safely extract for 3D or 4D
            if c_axis == 1:
                volume = da_raw.values[:, c_index, :, :]
            else:
                volume = da_raw.values[c_index, :, :]

            masked_volume = volume * roi_mask[np.newaxis, :, :]

            # Run Trackpy
            f = tp.batch(
                masked_volume,
                diameter=diameter,
                minmass=minmass,
                threshold=threshold,
                processes=1,
            )

            # Update the corresponding Napari layer
            if not f.empty:
                new_coords = f[["frame", "y", "x"]].values
                active_layer.data = new_coords
                print(
                    f" -> Success: Rendered {len(new_coords)} points to the {ch} layer."
                )
            else:
                active_layer.data = np.empty((0, 3))
                print(f" -> No features found. Cleared the {ch} layer.")

        print("--- Batch Prediction Complete! ---\n")

    @magicgui(
        call_button="Link & Track Channels",
        track_radius={"label": "Max Tracking Drift (px)", "max": 50},
        merge_radius={
            "label": "Colocalization Radius (px)",
            "max": 20,
            "tooltip": "Merge dots closer than this",
        },
        memory={
            "label": "Memory (frames)",
            "max": 5,
            "tooltip": "Max frames a cell can disappear and be remembered",
        },
        min_track_length={
            "label": "Min Track Length",
            "max": 11,
            "tooltip": "Delete noise! Tracks shorter than this are removed.",
        },
    )
    def link_and_track_widget(
        track_radius: int = 15,
        merge_radius: float = 5.0,
        memory: int = 1,
        min_track_length: int = 3,
    ):
        print("\n--- Starting Colocalization & Tracking ---")

        # ---------------------------------------------------------
        # 1. Harvest Data
        # ---------------------------------------------------------
        df_list = []
        for layer, color_name in [
            (layer_green, "Green"),
            (layer_red, "Red"),
            (layer_blue, "Blue"),
        ]:
            if len(layer.data) > 0:
                df = pd.DataFrame(layer.data, columns=["Z", "Y", "X"])
                df["color"] = color_name
                df_list.append(df)

        if not df_list:
            print("No points found in any layer. Run prediction first.")
            return

        df_all = pd.concat(df_list, ignore_index=True)

        # ---------------------------------------------------------
        # 2. Spatial Deduplication (Colocalization per Slice)
        # ---------------------------------------------------------
        merged_points = []
        for z, group in df_all.groupby("Z"):
            pts = group[["Y", "X"]].values
            colors = group["color"].values

            tree = cKDTree(pts)
            clusters = tree.query_ball_tree(tree, r=merge_radius)

            processed_indices = set()
            for i, neighbors in enumerate(clusters):
                if i in processed_indices:
                    continue
                for n in neighbors:
                    processed_indices.add(n)

                mean_y, mean_x = np.mean(pts[neighbors, 0]), np.mean(pts[neighbors, 1])
                cluster_colors = set(colors[neighbors])

                # Name the state for THIS specific slice
                if len(cluster_colors) == 3:
                    slice_pheno = "Triple Positive"
                elif "Green" in cluster_colors and "Red" in cluster_colors:
                    slice_pheno = "Green + Red"
                elif "Green" in cluster_colors and "Blue" in cluster_colors:
                    slice_pheno = "Green + Blue"
                elif "Red" in cluster_colors and "Blue" in cluster_colors:
                    slice_pheno = "Red + Blue"
                else:
                    slice_pheno = f"{list(cluster_colors)[0]} Only"

                merged_points.append(
                    {
                        "frame": int(z),
                        "y": mean_y,
                        "x": mean_x,
                        "Slice_Phenotype": slice_pheno,
                    }
                )

        df_merged = pd.DataFrame(merged_points)

        # ---------------------------------------------------------
        # 3. Track & Filter
        # ---------------------------------------------------------
        if df_merged.empty:
            return

        tp.quiet()
        t = tp.link(df_merged, search_range=track_radius, memory=memory)

        # Determine the GLOBAL identity of the track based on its history
        def get_track_phenotype(group):
            colors_in_track = set()
            for pheno in group["Slice_Phenotype"]:
                if "Green" in pheno or "Triple" in pheno:
                    colors_in_track.add("Green")
                if "Red" in pheno or "Triple" in pheno:
                    colors_in_track.add("Red")
                if "Blue" in pheno or "Triple" in pheno:
                    colors_in_track.add("Blue")

            has_g, has_r, has_b = (
                "Green" in colors_in_track,
                "Red" in colors_in_track,
                "Blue" in colors_in_track,
            )

            if has_g and has_r and has_b:
                return "Blue -> Green -> Red"
            if has_g and has_r:
                return "Green -> Red"
            if has_g and has_b:
                return "Blue -> Green"
            if has_r and has_b:
                return "Blue -> Red"
            if has_g:
                return "Green Only"
            if has_r:
                return "Red Only"
            if has_b:
                return "Blue Only"
            return "Unknown"

        # Map the global identity to every point in the track
        track_phenotypes = t.groupby("particle").apply(get_track_phenotype)
        t["Track_Phenotype"] = t["particle"].map(track_phenotypes)

        # Filter out the short noise tracks
        t_filtered = tp.filter_stubs(t, min_track_length)

        if t_filtered.empty:
            print(
                f"No tracks longer than {min_track_length} frames survived! Try lowering the threshold."
            )
            return

        # ---------------------------------------------------------
        # 4. Strict Color Mapping (Fixes the White/Green bug)
        # ---------------------------------------------------------
        master_color_dict = {
            "Green Only": "lime",
            "Red Only": "red",
            "Blue Only": "cyan",
            "Green -> Red": "yellow",
            "Blue -> Green": "dodgerblue",
            "Blue -> Red": "magenta",
            "Blue -> Green -> Red": "white",
            "Unknown": "gray",
        }

        active_phenos = t_filtered["Track_Phenotype"].unique().tolist()
        N = len(active_phenos)

        # Force the property values onto a strict 0.0 to 1.0 scale
        if N == 1:
            pheno_float_list = [0.5] * len(t_filtered)
            c = transform_color(master_color_dict[active_phenos[0]])[0]
            custom_cmap = Colormap(colors=[c, c], name="exact_cmap")
        else:
            pheno_to_float = {
                pheno: float(i) / (N - 1) for i, pheno in enumerate(active_phenos)
            }
            pheno_float_list = [
                pheno_to_float[p] for p in t_filtered["Track_Phenotype"]
            ]

            rgba_colors = [
                transform_color(master_color_dict[k])[0] for k in active_phenos
            ]
            controls = np.linspace(0, 1, N)
            custom_cmap = Colormap(
                colors=rgba_colors, controls=controls, name="exact_cmap"
            )

        # ---------------------------------------------------------
        # 5. Push to Viewer
        # ---------------------------------------------------------
        track_data = t_filtered[["particle", "frame", "y", "x"]].values
        track_props = {
            "Track_Phenotype": t_filtered["Track_Phenotype"].tolist(),
            "color_val": pheno_float_list,
        }

        # Delete the old layer to avoid internal Napari state bugs
        if "Trajectories" in viewer.layers:
            viewer.layers.remove("Trajectories")

        viewer.add_tracks(
            track_data,
            properties=track_props,
            color_by="color_val",
            # --- THE FIX: Use colormaps_dict, NOT colormap ---
            colormaps_dict={"color_val": custom_cmap},
            name="Trajectories",
            tail_width=4,
            tail_length=max(3, memory + 2),
        )

        # ---------------------------------------------------------
        # 6. Cleanup Original Point Layers
        # ---------------------------------------------------------
        # We use the *Slice_Phenotype* here so we only put dots back where they actually existed
        green_coords, red_coords, blue_coords = [], [], []
        for _, row in t_filtered.iterrows():
            z, y, x, slice_pheno = (
                row["frame"],
                row["y"],
                row["x"],
                row["Slice_Phenotype"],
            )
            coord = [z, y, x]

            if "Green" in slice_pheno or slice_pheno == "Triple Positive":
                green_coords.append(coord)
            if "Red" in slice_pheno or slice_pheno == "Triple Positive":
                red_coords.append(coord)
            if "Blue" in slice_pheno or slice_pheno == "Triple Positive":
                blue_coords.append(coord)

        layer_green.data = np.array(green_coords) if green_coords else np.empty((0, 3))
        layer_red.data = np.array(red_coords) if red_coords else np.empty((0, 3))
        layer_blue.data = np.array(blue_coords) if blue_coords else np.empty((0, 3))

        print(
            f"--- Success! Generated {t_filtered['particle'].nunique()} 3D trajectories. ---\n"
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
        for layer, color_name in [
            (layer_green, "Green"),
            (layer_red, "Red"),
            (layer_blue, "Blue"),
        ]:
            # Ensure the layer exists and has data before trying to extract it
            if layer is not None and len(layer.data) > 0:
                df = pd.DataFrame(layer.data, columns=["Z", "Y", "X"])
                df["Channel"] = color_name
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
        link_and_track_widget, name="3. Tracking & Colocalization", area="right"
    )
    viewer.window.add_dock_widget(
        export_points_widget, name="4. Export Results", area="right"
    )

    napari.run()


if __name__ == "__main__":
    main()
