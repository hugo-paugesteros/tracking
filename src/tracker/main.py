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
import xml.etree.ElementTree as ET

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
                if "Profile 1" in color_profile:
                    channel_map = {
                        "Rh4": np.where(c_metadata == "ALEXA 488")[0][0],  # Green
                        "Rh6": np.where(c_metadata == "ALEXA 555")[0][0],  # Red
                        "Rh5": np.where(c_metadata == "ALEXA 647")[0][0],  # Blue
                    }
                else:
                    channel_map = {
                        "Rh4": np.where(c_metadata == "ALEXA 647")[0][0],
                        "Rh6": np.where(c_metadata == "ALEXA 488")[0][0],
                        "Rh5": np.where(c_metadata == "ALEXA 555")[0][0],
                    }
            except Exception as e:
                print(
                    f"Warning: Channel metadata lookup failed ({e}). Using default order."
                )
                channel_map = {"Rh4": 0, "Rh6": 1, "Rh5": 2}

            print(channel_map)

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

            if volume.ndim == 2:
                # No Z axis (single 2D image): unlike a multi-Z stack, there's
                # no "cell drifted outside the ROI at an earlier Z" concern
                # here - it's the same one slice for every channel - so it's
                # safe (and desired) to restrict detection itself to the ROI,
                # rather than relying on later steps to filter it out.
                roi_data = np.asarray(layer_roi.data)
                if np.any(roi_data):
                    volume = volume * roi_data.astype(bool)
                # trackpy.batch still needs a leading frame axis to iterate
                # over, even if there's only one
                volume = volume[np.newaxis, :, :]

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

    def _compute_transitions(df_all, z_max, track_radius, merge_radius, memory):
        """Colocalize Rh5/Rh6 at z_max (inside the ROI drawn on that slice),
        then, for multi-Z stacks, backtrack each colocalized point toward
        Z=0 to determine whether it originates from an Rh4 point or from
        nothing observed (Rh3). Single-slice images (z_max == 0) have no
        lower Z to backtrack through, so "origin" stays None for them -
        only the raw colocalization is meaningful, not a transition.

        Returns (seeds, results, channel_points):
        - seeds: the Rh5/Rh6 clusters found at z_max
        - results: one dict per seed with "phenotype" (Rh5/Rh6/Rh5+Rh6),
          "origin" ("Rh4"/"Rh3"/None), and "path" ((z, y, x) tuples,
          seed frame first)
        - channel_points: {"Rh4"/"Rh5"/"Rh6": [(z, y, x), ...]} - every raw
          detection that is part of a surviving seed or backtrack path
        """
        end_df = df_all[
            (df_all["Z"] == z_max) & (df_all["target"].isin(["Rh5", "Rh6"]))
        ]

        if np.any(layer_roi.data):
            roi_mask = layer_roi.data.astype(bool)
            yy = np.clip(end_df["Y"].round().astype(int), 0, roi_mask.shape[0] - 1)
            xx = np.clip(end_df["X"].round().astype(int), 0, roi_mask.shape[1] - 1)
            end_df = end_df[roi_mask[yy, xx]]

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
                        "z_points": [
                            (pts[n, 0], pts[n, 1], labels[n]) for n in neighbors
                        ],
                    }
                )

        channel_points = {"Rh4": [], "Rh5": [], "Rh6": []}
        if not seeds:
            return seeds, [], channel_points

        for seed in seeds:
            for y, x, target in seed["z_points"]:
                channel_points[target].append((z_max, y, x))

        if z_max == 0:
            results = [
                {
                    "phenotype": s["phenotype"],
                    "origin": None,
                    "path": [(z_max, s["y"], s["x"])],
                }
                for s in seeds
            ]
            return seeds, results, channel_points

        # Backtracking is unrestricted by the ROI - a cell's origin (an Rh4
        # point, or nothing observed = Rh3) can lie outside the footprint
        # drawn at the last Z, since cells drift as they differentiate.
        points_by_z = {z: g[["Y", "X"]].values for z, g in df_all.groupby("Z")}
        targets_by_z = {z: g["target"].values for z, g in df_all.groupby("Z")}

        results = []
        for seed in seeds:
            cur_y, cur_x = seed["y"], seed["x"]
            path = [(z_max, cur_y, cur_x)]
            missed = 0
            origin = "Rh3"  # default: trail never confirms an Rh4 point

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
                {"phenotype": seed["phenotype"], "origin": origin, "path": path}
            )

        return seeds, results, channel_points

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

        # Whatever Z the user is currently viewing is the last useful frame -
        # navigate there (where Rh5/Rh6 are clearest) before clicking this.
        z_max = int(viewer.dims.current_step[0])
        if np.any(layer_roi.data):
            print(f"Seeding from Z={z_max}, restricted to the drawn ROI.")
        else:
            print(f"Seeding from Z={z_max}. No ROI drawn - using the entire slice.")

        seeds, results, channel_points = _compute_transitions(
            df_all, z_max, track_radius, merge_radius, memory
        )

        if not seeds:
            print("No Rh5/Rh6 points found at the last Z (inside the ROI).\n")
            return

        single_slice = z_max == 0
        if single_slice:
            print(
                "Single-slice image: no lower Z to backtrack through, so Rh4/Rh3 "
                "origin can't be determined - showing raw colocalization only."
            )
        for r in results:
            r["label"] = (
                r["phenotype"]
                if single_slice
                else f"{r['origin']} -> {r['phenotype']}"
            )

        # Keep only the points that belong to a surviving Rh5/Rh6 seed (and, for
        # multi-Z stacks, a confirmed backtrack path). A single slice can't say
        # anything about Rh4's relevance, so its points are left untouched.
        layers_to_clean = [("Rh5", layer_rh5), ("Rh6", layer_rh6)]
        if not single_slice:
            layers_to_clean.append(("Rh4", layer_rh4))
        for target_name, layer in layers_to_clean:
            coords = channel_points[target_name]
            layer.data = np.array(coords) if coords else np.empty((0, 3))

        # ---------------------------------------------------------
        # Report counts
        # ---------------------------------------------------------
        counts = pd.Series([r["label"] for r in results]).value_counts()
        if single_slice:
            categories = ["Rh5", "Rh6", "Rh5+Rh6"]
            header = f"📊 FINAL COUNT (colocalized at Z={z_max})"
        else:
            categories = [
                "Rh4 -> Rh5",
                "Rh4 -> Rh6",
                "Rh4 -> Rh5+Rh6",
                "Rh3 -> Rh5",
                "Rh3 -> Rh6",
                "Rh3 -> Rh5+Rh6",
            ]
            header = f"📊 FINAL COUNT (colocalized at Z={z_max}, backtracked to origin)"

        print("\n" + "=" * 40)
        print(header)
        print("=" * 40)
        for cat in categories:
            print(f" -> {cat:<16}: {counts.get(cat, 0)}")
        print("=" * 40 + "\n")

        # ---------------------------------------------------------
        # Push results to the viewer as Tracks
        # ---------------------------------------------------------
        master_color_dict = {
            "Rh4 -> Rh5": "cyan",
            "Rh4 -> Rh6": "magenta",
            "Rh4 -> Rh5+Rh6": "yellow",
            "Rh3 -> Rh5": "steelblue",
            "Rh3 -> Rh6": "orchid",
            "Rh3 -> Rh5+Rh6": "khaki",
            "Rh5": "cyan",
            "Rh6": "magenta",
            "Rh5+Rh6": "yellow",
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
            pheno_float_list = [pheno_to_float[p] for p in t_display["Track_Phenotype"]]

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

    @magicgui(
        call_button="Export to ImageJ Cell Counter",
        save_path={"label": "Save Location:", "mode": "w", "filter": "*.xml"},
    )
    def export_cellcounter_widget(
        save_path: Path = Path("cell_counter_export.xml"),
    ):
        print("\n--- Exporting to ImageJ Cell Counter ---")

        if da_raw is None:
            print("Please load an image first.")
            return

        if save_path is None:
            print("Export canceled: No save path selected.")
            return

        markers = {t: [] for t in range(1, 9)}

        def _add(marker_type, z, y, x):
            # ImageJ Z slices are 1-indexed; ours are 0-indexed
            markers[marker_type].append(
                (int(round(x)), int(round(y)), int(round(z)) + 1)
            )

        # Types 4/5/6: raw points, straight from the layers
        for layer, marker_type in [(layer_rh4, 4), (layer_rh5, 5), (layer_rh6, 6)]:
            for z, y, x in layer.data:
                _add(marker_type, z, y, x)

        # Types 1/2/3/7 need colocalization/backtracking, reusing the exact
        # same computation as "Colocalize & Backtrack" (and whatever
        # parameters are currently set on that panel), but without touching
        # any layers - this works standalone, without that button having
        # been clicked first.
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

        if df_list:
            df_all = pd.concat(df_list, ignore_index=True)
            df_all["Z"] = df_all["Z"].round().astype(int)
            z_max = int(viewer.dims.current_step[0])

            _, results, _ = _compute_transitions(
                df_all,
                z_max,
                link_and_track_widget.track_radius.value,
                link_and_track_widget.merge_radius.value,
                link_and_track_widget.memory.value,
            )

            for r in results:
                z_end, y_end, x_end = r["path"][0]
                if r["phenotype"] == "Rh5+Rh6":
                    _add(7, z_end, y_end, x_end)
                    if r["origin"] == "Rh4":
                        _add(2, z_end, y_end, x_end)
                    elif r["origin"] == "Rh3":
                        _add(3, z_end, y_end, x_end)
                elif r["phenotype"] == "Rh5" and r["origin"] == "Rh4":
                    _add(1, z_end, y_end, x_end)

        # Physical calibration is purely informational here - marker
        # coordinates stay in pixels, matching ImageJ's own convention
        def _spacing(dim):
            coord = da_raw.coords.get(dim)
            if coord is not None and len(coord) > 1:
                return float(abs(coord[1] - coord[0]))
            return 1.0

        image_filename = da_raw.attrs.get("path", da_raw.attrs.get("name", "unknown"))

        root = ET.Element("CellCounter_Marker_File")
        img_props = ET.SubElement(root, "Image_Properties")
        ET.SubElement(img_props, "Image_Filename").text = str(image_filename)
        ET.SubElement(img_props, "X_Calibration").text = str(_spacing("X"))
        ET.SubElement(img_props, "Y_Calibration").text = str(_spacing("Y"))
        ET.SubElement(img_props, "Z_Calibration").text = str(_spacing("Z"))
        ET.SubElement(img_props, "Calibration_Unit").text = "micron"

        marker_data = ET.SubElement(root, "Marker_Data")
        ET.SubElement(marker_data, "Current_Type").text = "1"

        for t in range(1, 9):
            mt = ET.SubElement(marker_data, "Marker_Type")
            ET.SubElement(mt, "Type").text = str(t)
            ET.SubElement(mt, "Name").text = f"Type {t}"
            for x, y, z in markers[t]:
                m = ET.SubElement(mt, "Marker")
                ET.SubElement(m, "MarkerX").text = str(x)
                ET.SubElement(m, "MarkerY").text = str(y)
                ET.SubElement(m, "MarkerZ").text = str(z)

        if save_path.suffix != ".xml":
            save_path = save_path.with_suffix(".xml")

        ET.indent(root, space="    ")
        ET.ElementTree(root).write(save_path, encoding="UTF-8", xml_declaration=True)

        type_labels = {
            1: "Rh4 -> Rh5",
            2: "Rh4 -> Rh5+Rh6",
            3: "nothing(Rh3) -> Rh5+Rh6",
            4: "Rh4 points",
            5: "Rh5 points",
            6: "Rh6 points",
            7: "Rh5+Rh6 points",
            8: "(unused)",
        }
        print("Success! Saved Cell Counter markers to:")
        print(f" -> {save_path.absolute()}")
        print("\nMarker counts by type:")
        for t in range(1, 9):
            print(f"  Type {t} ({type_labels[t]}): {len(markers[t])}")
        print("----------------------------\n")

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
    viewer.window.add_dock_widget(
        export_cellcounter_widget, name="5. Export to ImageJ", area="right"
    )

    napari.run()


if __name__ == "__main__":
    main()
