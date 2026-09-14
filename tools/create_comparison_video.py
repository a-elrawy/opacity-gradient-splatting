import cv2
import os
import argparse
import numpy as np

def create_video(ours_dir, theirs_dir, out_path, fps=30, zoom_center=(0.5, 0.5), zoom_size=100, zoom_scale=3, theirs_name="Baseline"):
    ours_files = sorted([f for f in os.listdir(ours_dir) if f.endswith(('.png', '.jpg'))])
    theirs_files = sorted([f for f in os.listdir(theirs_dir) if f.endswith(('.png', '.jpg'))])

    if len(ours_files) != len(theirs_files):
        print(f"Warning: Number of images differ! Ours: {len(ours_files)}, Theirs: {len(theirs_files)}")
        # We will iterate up to the minimum length
    
    min_len = min(len(ours_files), len(theirs_files))
    if min_len == 0:
        print("Error: No images found.")
        return

    # Read first image to get dimensions
    first_img_path = os.path.join(ours_dir, ours_files[0])
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        print(f"Failed to read {first_img_path}")
        return
        
    H, W, C = first_img.shape
    
    out_W = W * 2
    out_H = H
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (out_W, out_H))

    # Helper function to add inset
    def add_inset(img, zx, zy, zw, zh, scale):
        # copy img to avoid modifying original yet
        res = img.copy()
        # Ensure bounds
        y1 = max(0, zy - zh//2)
        y2 = min(H, zy + zh//2)
        x1 = max(0, zx - zw//2)
        x2 = min(W, zx + zw//2)
        
        # crop
        patch = res[y1:y2, x1:x2]
        if patch.size == 0: return res
        
        # resize
        patch_resized = cv2.resize(patch, (zw*scale, zh*scale), interpolation=cv2.INTER_CUBIC)
        pH, pW = patch_resized.shape[:2]
        
        # Draw rectangle on original image showing where the patch is from
        cv2.rectangle(res, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Place the magnified patch in the bottom right corner
        place_y2 = H - 10
        place_y1 = max(10, H - pH - 10)
        place_x2 = W - 10
        place_x1 = max(10, W - pW - 10)
        
        actual_h = place_y2 - place_y1
        actual_w = place_x2 - place_x1
        patch_resized_fit = cv2.resize(patch_resized, (actual_w, actual_h))
        
        # Draw border for inset
        res[place_y1:place_y2, place_x1:place_x2] = patch_resized_fit
        cv2.rectangle(res, (place_x1, place_y1), (place_x2, place_y2), (0, 0, 255), 3)
        return res

    center_x = int(W * zoom_center[0])
    center_y = int(H * zoom_center[1])

    print(f"Generating video... {min_len} frames")
    for i in range(min_len):
        img1 = cv2.imread(os.path.join(ours_dir, ours_files[i]))
        img2 = cv2.imread(os.path.join(theirs_dir, theirs_files[i]))
        
        if img1 is None or img2 is None:
            continue
            
        # Ensure same size just in case
        if img1.shape != first_img.shape: img1 = cv2.resize(img1, (W, H))
        if img2.shape != first_img.shape: img2 = cv2.resize(img2, (W, H))
        
        # Apply inset zoom
        img1_zoomed = add_inset(img1, center_x, center_y, zoom_size, zoom_size, zoom_scale)
        img2_zoomed = add_inset(img2, center_x, center_y, zoom_size, zoom_size, zoom_scale)
        
        # Add labels
        cv2.putText(img1_zoomed, "Our Method", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(img2_zoomed, theirs_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Concatenate horizontally
        frame = np.concatenate((img1_zoomed, img2_zoomed), axis=1)
        
        # Add a vertical divider line
        cv2.line(frame, (W, 0), (W, H), (255, 255, 255), 4)

        out.write(frame)
        
    out.release()
    print(f"Video saved to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", required=True, help="Directory containing images for our method")
    parser.add_argument("--theirs", required=True, help="Directory containing images for baseline method")
    parser.add_argument("--out", required=True, help="Output MP4 path")
    parser.add_argument("--name", default="FSGS", help="Name of baseline method for text label")
    parser.add_argument("--zx", type=float, default=0.5, help="Normalized X center for zoom inset (0.0 to 1.0)")
    parser.add_argument("--zy", type=float, default=0.5, help="Normalized Y center for zoom inset (0.0 to 1.0)")
    parser.add_argument("--zsize", type=int, default=150, help="Size of the zoom patch (width and height)")
    parser.add_argument("--zscale", type=int, default=3, help="Scale factor for the zoom patch")
    parser.add_argument("--fps", type=int, default=15, help="Video FPS")
    args = parser.parse_args()
    
    create_video(args.ours, args.theirs, args.out, args.fps, (args.zx, args.zy), args.zsize, args.zscale, args.name)
