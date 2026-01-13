import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from ttkthemes import ThemedTk
import os
from PIL import Image
import threading
import subprocess
import json
import glob
import time
import shutil
from pathlib import Path
import tempfile
import torch
import numpy as np
from depth_anything_3.api import DepthAnything3
import torch.nn.functional as F
import trimesh
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues

#download pre-compiled COLMAP from https://github.com/colmap/colmap/releases

#this is my HF token for VGGT (read only)
os.environ["HF_TOKEN"] = "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"

DEFAULT_COLMAP_PATH = r"C:\Users\cimci\Desktop\colmap-x64-windows-cuda\COLMAP.bat"
DEFAULT_SAI_PATH = r"C:\Users\cimci\Desktop\LFR_New\python\SAI.exe"
DEFAULT_IMAGE_FOLDER_PATH = r"C:\Users\cimci\Desktop\LFR_New\python\test_images"


class LFR_Interface(ThemedTk):
    def __init__(self):
        super().__init__(theme="krok")
        self.title("SAI Interface")
        self.geometry("600x600")
        self.colmap_cmd = tk.StringVar(value=DEFAULT_COLMAP_PATH)
        self.sai_exe_path = tk.StringVar(value=DEFAULT_SAI_PATH)
        self.image_folder_path = tk.StringVar(value=DEFAULT_IMAGE_FOLDER_PATH)
        self.pose_method = tk.StringVar(value="vggt")
        self.status_msg = tk.StringVar(value="Ready")
        self.progress_val = tk.DoubleVar(value=0.0)
        self.sai_process = None

        self.create_widgets()

    def clear_console(self):
        os.system("cls")
        print("\033[2J\033[H", end="", flush=True)

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding=15)
        main_frame.pack(fill="both", expand=True)

        action_frame = ttk.LabelFrame(main_frame, text="Workflow", padding=10)
        action_frame.pack(fill="x", pady=10)
        action_frame.columnconfigure(0, weight=1)

        method_frame = ttk.Frame(action_frame)
        method_frame.pack(fill="x", pady=5)
        ttk.Label(method_frame, text="Pose Estimation Method:").pack(side="left", padx=5)

        da3_rb = ttk.Radiobutton(method_frame, text="DA3", variable=self.pose_method, value="da3")
        da3_rb.pack(side="left", padx=10)
        vggt_rb = ttk.Radiobutton(method_frame, text="VGGT", variable=self.pose_method, value="vggt")
        vggt_rb.pack(side="left", padx=10)
        colmap_rb = ttk.Radiobutton(method_frame, text="COLMAP", variable=self.pose_method, value="colmap")
        colmap_rb.pack(side="left", padx=10)

        folder_frame = ttk.Frame(action_frame)
        folder_frame.pack(fill="x", pady=5)
        ttk.Label(folder_frame, text="Image Folder:").pack(side="left", padx=5)
        folder_entry = ttk.Entry(folder_frame, textvariable=self.image_folder_path, state="readonly", width=50)
        folder_entry.pack(side="left", padx=5, fill="x", expand=True)
        ttk.Button(folder_frame, text="Browse...", command=self.browse_folder).pack(side="left", padx=5)

        self.run_btn = ttk.Button(action_frame, text="GENERATE POSES", command=self.start_processing)
        self.run_btn.pack(fill="x", ipady=10, pady=5)

        status_frame = ttk.LabelFrame(main_frame, text="Process Log", padding=10)
        status_frame.pack(fill="both", expand=True, pady=10)
        status_frame.rowconfigure(1, weight=1)
        status_frame.columnconfigure(0, weight=1)

        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_val, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        self.log_text = tk.Text(status_frame, height=10, state="disabled", font=("Consolas", 10), relief="solid", borderwidth=1)
        self.log_text.grid(row=1, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(status_frame, orient='vertical', command=self.log_text.yview)
        scrollbar.grid(row=1, column=1, sticky='ns')
        self.log_text['yscrollcommand'] = scrollbar.set

    def browse_folder(self):
        path = filedialog.askdirectory(title="Select Image Folder", initialdir=self.image_folder_path.get())
        if path:
            self.image_folder_path.set(os.path.normpath(path))
            self.log(f"Selected image folder: {self.image_folder_path.get()}")

    def log(self, message):
        def _log():
            self.log_text.config(state="normal")
            self.log_text.insert("end", message + "\n")
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self.after(0, _log)

    def set_progress(self, value):
        self.after(0, lambda: self.progress_val.set(value))

    def start_processing(self):
        self.clear_console()
        method = self.pose_method.get()
        colmap_path = self.colmap_cmd.get()
        sai_path = self.sai_exe_path.get()
        source_img_dir = self.image_folder_path.get()

        if not os.path.isdir(source_img_dir):
            messagebox.showerror("Error", f"Image folder not found:\n{source_img_dir}")
            return
        if not os.path.isfile(sai_path):
            messagebox.showerror("Error", f"SAI.exe not found:\n{sai_path}")
            return
        if method == "colmap" and not os.path.isfile(colmap_path):
            messagebox.showerror("Error", f"COLMAP not found:\n{colmap_path}")
            return

        self.run_btn.config(state="disabled")
        self.set_progress(0)
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, "end")
        self.log_text.config(state="disabled")

        threading.Thread(target=self.process_workflow, args=(source_img_dir,), daemon=True).start()

    def process_workflow(self, source_img_dir):
        if self.sai_process and self.sai_process.poll() is None:
            self.log("Terminating existing SAI process...")
            self.sai_process.kill()

        self.log("Starting workflow...")
        self.log(f"Paths:")
        self.log(f"  COLMAP: {self.colmap_cmd.get()}")
        self.log(f"  SAI.exe: {self.sai_exe_path.get()}")
        self.log(f"  Images: {self.image_folder_path.get()}")

        sai_dir = os.path.dirname(self.sai_exe_path.get())
        img_dir = os.path.join(sai_dir, "images")
        out_dir = os.path.join(sai_dir, "poses")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        source_img_dir = os.path.normpath(source_img_dir)
        self.log(f"Copying images from '{source_img_dir}' to '{img_dir}'...")
        self.set_progress(5)

        for item in os.listdir(img_dir):
            item_path = os.path.join(img_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

        allowed_exts = {".jpg", ".jpeg", ".png"}
        img_files = sorted({
            str(p) for p in Path(source_img_dir).iterdir()
            if p.is_file() and p.suffix.lower() in allowed_exts
        })

        if not img_files:
            raise FileNotFoundError("No valid images found in the selected directory!")

        for f in img_files:
            dest_path = os.path.join(img_dir, os.path.basename(f))
            with Image.open(f) as im:
                w, h = im.size
                self.log(f"{os.path.basename(f)} Image size: {w}x{h}")
                if w != h:
                    s = min(w, h)
                    left = (w - s) // 2
                    top = (h - s) // 2
                    right = left + s
                    bottom = top + s
                    im = im.crop((left, top, right, bottom))
                    self.log(f"{os.path.basename(f)} cropped to square format: {s}x{s}")
                im.save(dest_path)
        self.set_progress(10)

        method = self.pose_method.get()
        self.log(f"Pose Estimation Method: {method.upper()}")
        if method == "vggt":
            self.run_vggt(img_dir, out_dir)

        elif method == "da3":
            self.run_da3(img_dir, out_dir)

        elif method == "colmap":
            self.run_colmap_pipeline(img_dir, out_dir)


        self.set_progress(100)
        self.log("\nWorkflow complete!")

        self.log(f"Launching {self.sai_exe_path.get()}...")
        subprocess.Popen([self.sai_exe_path.get()], cwd=sai_dir)
        self.run_btn.config(state="normal")

    def run_da3(self, img_dir, out_dir):

        self.log("--- Starting DA3 Pipeline ---")

        allowed_exts = {".jpg", ".jpeg", ".png"}
        image_names = sorted({
            str(p) for p in Path(img_dir).iterdir()
            if p.is_file() and p.suffix.lower() in allowed_exts
        })

        if not image_names:
            raise FileNotFoundError(f"No images found in {img_dir}")
        self.log(f"Found {len(image_names)} images")
        self.set_progress(15)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log(f"Using device: {str(device).upper()}")

        self.log("Loading DA3 model (da3-large)...")
        model = DepthAnything3.from_pretrained("depth-anything/da3-large").to(device=device)
        self.set_progress(25)

        export_root = os.path.join(out_dir, "da3_output")
        os.makedirs(export_root, exist_ok=True)

        for old_npz in glob.glob(os.path.join(export_root, "**", "*.npz"), recursive=True):
            os.remove(old_npz)

        self.log("Running DA3 inference...")
        start_time = time.time()
        _ = model.inference(
            image_names,
            export_dir=export_root,
            export_format="npz",
            process_res=504
        )

        time.sleep(1)

        self.log(f"Inference complete in {time.time() - start_time:.2f} seconds")
        self.set_progress(60)

        npz_files = glob.glob(os.path.join(export_root, "**", "*.npz"), recursive=True)
        if not npz_files:
            raise FileNotFoundError(f"No NPZ exports found under {export_root}")
        latest_npz = max(npz_files, key=lambda p: os.path.getmtime(p))
        self.log(f"Using NPZ: {latest_npz}")

        data = np.load(latest_npz)
        if "extrinsics" not in data:
            raise KeyError("NPZ does not contain 'extrinsics' key")
        extrinsics = data["extrinsics"]
        if extrinsics.ndim == 2:
            extrinsics = extrinsics[None, ...]  # single frame case

        n_imgs = len(image_names)
        n_ext = extrinsics.shape[0]
        n = min(n_imgs, n_ext)
        if n_ext != n_imgs:
            self.log(f"Warning: extrinsics count ({n_ext}) != image count ({n_imgs}). Truncating to {n}.")

        output_data = {"images": []}
        for i in range(n):
            image_filename = os.path.basename(image_names[i])
            M3x4 = extrinsics[i].tolist()
            output_data["images"].append({
                "imagefile": image_filename,
                "M3x4": M3x4
            })

        json_path = os.path.join(out_dir, "poses.json")
        with open(json_path, "w") as f:
            json.dump(output_data, f, indent=4)
        self.log(f"Saved camera poses to {json_path}")
        self.set_progress(95)

    def run_vggt(self, img_dir, out_dir):

        self.log("--- Starting VGGT Pipeline ---")
        self.log("Loading images...")

        allowed_exts = {".jpg", ".jpeg", ".png"}
        image_names = sorted({
            str(p) for p in Path(img_dir).iterdir()
            if p.is_file() and p.suffix.lower() in allowed_exts
        })

        if not image_names:
            raise FileNotFoundError(f"No images found in {img_dir}")
        self.log(f"Found {len(image_names)} images")
        self.set_progress(15)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.log(f"Using device: {device.upper()}")

        self.log("Loading VGGT model (facebook/VGGT-1B)...")
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        images = load_and_preprocess_images(image_names).to(device)
        self.set_progress(30)

        self.log("Running model inference...")
        start_time = time.time()
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                predictions = model(images)
        self.log(f"Inference complete in {time.time() - start_time:.2f} seconds.")
        self.set_progress(60)

        self.log("Processing model output...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions['pose_enc'], images.shape[-2:])
        depth_map = predictions['depth']
        depth_conf = predictions['depth_conf']

        extrinsic_np = extrinsic.squeeze(0).cpu().numpy()
        depth_map_np = depth_map.squeeze(0).cpu().numpy()
        depth_conf_np = depth_conf.squeeze(0).cpu().numpy()
        intrinsic_np = intrinsic.squeeze(0).cpu().numpy()

        output_data = {"images": []}
        for i, image_path in enumerate(image_names):
            image_filename = os.path.basename(image_path)
            extrinsic_matrix = extrinsic_np[i].tolist()
            output_data["images"].append({
                "imagefile": image_filename,
                "M3x4": extrinsic_matrix
            })

        json_path = os.path.join(out_dir, "poses.json")
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        self.log(f"Saved camera poses to {json_path}")
        self.set_progress(72)

        self.log("Generating point cloud (VGGT defaults)...")
        points_3d = unproject_depth_map_to_point_map(depth_map_np, extrinsic_np, intrinsic_np)

        ##### did not use shared_camera here #####
        conf_thres_value = 5.0
        max_points_for_colmap = 100000
        shared_camera = False
        camera_type = "PINHOLE"

        num_frames, height, width, _ = points_3d.shape
        vggt_fixed_resolution_h = height
        vggt_fixed_resolution_w = width
        image_size = np.array([vggt_fixed_resolution_h, vggt_fixed_resolution_w])

        points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution_h, vggt_fixed_resolution_w),
            mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf_np >= conf_thres_value
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d_flat = points_3d[conf_mask]
        points_rgb_flat = points_rgb[conf_mask]
        #### points_xyf_flat for COLMAP export ####
        points_xyf_flat = points_xyf[conf_mask]

        sparse_dir = os.path.join(out_dir, "sparse")
        os.makedirs(sparse_dir, exist_ok=True)
        ply_path = os.path.join(sparse_dir, "vggt_points.ply")
        trimesh.PointCloud(points_3d_flat, colors=points_rgb_flat).export(ply_path)
        self.log(f"Saved point cloud to {ply_path}")
        self.set_progress(95)

    def _run_colmap_command(self, args):
        cmd_path = self.colmap_cmd.get()
        if cmd_path.lower().endswith(".bat"):
            cmd = ["cmd.exe", "/C", os.path.normpath(cmd_path)] + [os.path.normpath(arg) if os.path.sep in arg or '/' in arg else arg for arg in args]
        else:
            cmd = [os.path.normpath(cmd_path)] + [os.path.normpath(arg) if os.path.sep in arg or '/' in arg else arg for arg in args]

        self.log(f"Executing: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW
        )
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ''):
            self.log(line.strip())
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"COLMAP failed with exit code {process.returncode}.")

    def run_colmap_pipeline(self, img_dir, out_dir):
        self.log("--- Starting COLMAP Pipeline ---")
        json_out_path = os.path.join(out_dir, "poses.json")

        with tempfile.TemporaryDirectory(prefix="lfr_colmap_") as work_dir:
            database_path = os.path.join(work_dir, "database.db")
            sparse_model_path = os.path.join(work_dir, "sparse")
            os.makedirs(sparse_model_path, exist_ok=True)

            self.log("Feature Extraction...")
            self._run_colmap_command([
                "feature_extractor", "--database_path", database_path,
                "--image_path", img_dir, "--ImageReader.single_camera", "1",
            ])
            self.set_progress(35)

            self.log("Feature Matching...")
            self._run_colmap_command(["exhaustive_matcher", "--database_path", database_path])
            self.set_progress(60)

            self.log("Sparse Reconstruction...")
            self._run_colmap_command([
                "mapper", "--database_path", database_path,
                "--image_path", img_dir, "--output_path", sparse_model_path,
            ])
            self.set_progress(85)

            model_dirs = sorted([p for p in Path(sparse_model_path).iterdir() if p.is_dir()])
            if not model_dirs:
                raise RuntimeError("COLMAP produced no sparse model!")
            best_model_dir = max(model_dirs, key=lambda p: (p / "points3D.bin").stat().st_size if (p / "points3D.bin").exists() else 0)

            self.log("Converting model and saving poses.json...")
            txt_dir = os.path.join(work_dir, "sparse_txt")
            os.makedirs(txt_dir, exist_ok=True)
            self._run_colmap_command([
                "model_converter", "--input_path", str(best_model_dir),
                "--output_path", txt_dir, "--output_type", "TXT",
            ])
            self.generate_poses_json_from_colmap(txt_dir, img_dir, json_out_path)
            self.set_progress(95)

    def generate_poses_json_from_colmap(self, sparse_txt_path, img_dir, json_out_path):
        images_txt = os.path.join(sparse_txt_path, "images.txt")
        if not os.path.exists(images_txt):
            raise FileNotFoundError("COLMAP did not generate images.txt!")

        existing_files = {os.path.basename(p) for p in glob.glob(os.path.join(img_dir, "*.*"))}
        images_list = []

        with open(images_txt, "r") as f:
            lines = f.readlines()

        for line in lines:
            if line.strip().startswith("#"): continue
            parts = line.strip().split()
            if len(parts) < 10: continue

            img_name = parts[9]
            if img_name not in existing_files: continue

            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])

            Rcw = self._qvec_to_rotmat(qw, qx, qy, qz)
            Rwc_T = [[Rcw[0][0], Rcw[1][0], Rcw[2][0]], [Rcw[0][1], Rcw[1][1], Rcw[2][1]], [Rcw[0][2], Rcw[1][2], Rcw[2][2]]]

            cam_center = [- (Rwc_T[0][0] * tx + Rwc_T[0][1] * ty + Rwc_T[0][2] * tz),
                          - (Rwc_T[1][0] * tx + Rwc_T[1][1] * ty + Rwc_T[1][2] * tz),
                          - (Rwc_T[2][0] * tx + Rwc_T[2][1] * ty + Rwc_T[2][2] * tz)]

            M3x4 = [
                [1.0, 0.0, 0.0, cam_center[0]],
                [0.0, 1.0, 0.0, cam_center[1]],
                [0.0, 0.0, 1.0, -35.0],  # Fixed Z height
            ]
            images_list.append({"imagefile": img_name, "M3x4": M3x4})

        if not images_list:
            raise RuntimeError("No registered images found in COLMAP output!")

        images_list.sort(key=lambda x: x["imagefile"])
        with open(json_out_path, "w") as jf:
            json.dump({"images": images_list}, jf, indent=4)

        self.log(f"Saved poses.json for {len(images_list)} images")

    def _qvec_to_rotmat(self, qw, qx, qy, qz):
        R = [[0.0] * 3 for _ in range(3)]
        R[0][0] = 1 - 2 * (qy * qy + qz * qz)
        R[0][1] = 2 * (qx * qy - qz * qw)
        R[0][2] = 2 * (qx * qz + qy * qw)
        R[1][0] = 2 * (qx * qy + qz * qw)
        R[1][1] = 1 - 2 * (qx * qx + qz * qz)
        R[1][2] = 2 * (qy * qz - qx * qw)
        R[2][0] = 2 * (qx * qz - qy * qw)
        R[2][1] = 2 * (qy * qz + qx * qw)
        R[2][2] = 1 - 2 * (qx * qx + qy * qy)
        return R


if __name__ == "__main__":
    #to create the exe:
    #pyinstaller --noconfirm --clean aos_fire.spec
    app = LFR_Interface()
    app.mainloop()