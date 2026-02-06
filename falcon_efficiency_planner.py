import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os
import heapq
from torch import nn

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "segmentation_head_optimized.pth"
TEST_IMAGE_DIR = "data/test/Color_Images"
OUTPUT_DIR = "path_planning_results"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class Definitions
SAFE_CLASSES = [3, 8]  # Dry Grass (3), Landscape (8)
OBSTACLE_CLASSES = [0, 1, 2, 4, 5, 6, 7] # Rocks, Bushes, Trees, etc.

# ============================================================================
# 1. ROBUST A* ALGORITHM
# ============================================================================
class AStarPlanner:
    def __init__(self, downscale=0.1):
        self.scale = downscale # Work on a smaller grid for speed

    def heuristic(self, a, b):
        # Euclidean distance
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def get_path(self, mask):
        h, w = mask.shape
        # Downscale mask for grid
        small_h, small_w = int(h * self.scale), int(w * self.scale)
        grid = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

        # START: Bottom Center
        start = (small_h - 1, small_w // 2)

        # GOAL: Find the "furthest safe point" (highest up in the image)
        # Create a binary map: 1=Safe, 0=Obstacle
        binary_grid = np.isin(grid, SAFE_CLASSES).astype(int)
        
        # Safety Buffer: Erode safe zones so we don't hug rocks too closely
        kernel = np.ones((2,2), np.uint8)
        binary_grid = cv2.erode(binary_grid.astype(np.uint8), kernel)

        # Find safe pixels
        safe_pixels = np.argwhere(binary_grid == 1)
        if len(safe_pixels) == 0: return None # trapped
        
        # Pick the point with smallest Y (highest up) that is reachable
        # We prefer points near the center column to avoid driving off-screen
        candidates = safe_pixels[safe_pixels[:, 0] < small_h * 0.6] # Look in top 60%
        if len(candidates) == 0:
            candidates = safe_pixels # Fallback to anywhere safe

        # Sort by "High Up" (y) then "Center Aligned" (x)
        def score_goal(pt):
            y, x = pt
            dist_from_center = abs(x - (small_w // 2))
            return y + (dist_from_center * 0.5) # Penalize side exits slightly
            
        goal = tuple(sorted(candidates, key=score_goal)[0])

        # Force start/goal to be valid
        binary_grid[start] = 1
        binary_grid[goal] = 1

        # Run A*
        neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: self.heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[start], start))

        while oheap:
            current = heapq.heappop(oheap)[1]

            if current == goal:
                data = []
                while current in came_from:
                    # Upscale back to original resolution
                    orig_pt = (int(current[1] / self.scale), int(current[0] / self.scale))
                    data.append(orig_pt)
                    current = came_from[current]
                return data[::-1] # Path found!

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                if 0 <= neighbor[0] < small_h and 0 <= neighbor[1] < small_w:
                    if binary_grid[neighbor[0]][neighbor[1]] == 1: # Walkable
                        tentative_g_score = gscore[current] + self.heuristic(current, neighbor)
                        if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                            continue
                        if tentative_g_score < gscore.get(neighbor, float('inf')):
                            came_from[neighbor] = current
                            gscore[neighbor] = tentative_g_score
                            fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                            heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return None

# ============================================================================
# 2. MODEL LOADER (Optimized)
# ============================================================================
class ImprovedSegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH, dropout=0.1):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.GELU(), nn.Dropout2d(dropout)
        )
        self.block1 = nn.Sequential(    
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256), nn.GELU(), nn.Dropout2d(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256), nn.GELU(), nn.Dropout2d(dropout)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
        )
        self.classifier = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = x + self.block1(x)
        x = x + self.block2(x)
        x = self.refine(x)
        return self.classifier(x)

def load_model(model_path):
    print(f"Loading Model...")
    backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14")
    backbone.to(DEVICE).eval()
    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)
    head = ImprovedSegmentationHead(in_channels=384, out_channels=10, tokenW=w//14, tokenH=h//14)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint: head.load_state_dict(checkpoint['model_state_dict'])
    else: head.load_state_dict(checkpoint)
    head.to(DEVICE).eval()
    return backbone, head

# ============================================================================
# 3. VISUALIZATION
# ============================================================================
planner = AStarPlanner(downscale=0.15)

def process_image(img_path, output_path, backbone, head, transform):
    original = cv2.imread(img_path)
    if original is None: return
    pil_img = Image.open(img_path).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    # 1. Inference
    with torch.no_grad():
        features = backbone.forward_features(input_tensor)["x_norm_patchtokens"]
        logits = head(features)
        probs = torch.nn.functional.interpolate(logits, size=(original.shape[0], original.shape[1]), mode="bilinear")
        pred_mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy()

    # 2. A* Pathfinding
    path = planner.get_path(pred_mask)

    # 3. Draw Results
    vis = original.copy()
    
    # Overlay Obstacles in Red
    obstacle_mask = np.isin(pred_mask, OBSTACLE_CLASSES).astype(np.uint8) * 255
    red_layer = np.zeros_like(vis)
    red_layer[obstacle_mask > 0] = [0, 0, 255]
    cv2.addWeighted(red_layer, 0.4, vis, 1.0, 0, vis)
    
    status = "Searching..."
    if path:
        status = "A* OPTIMAL PATH"
        # Draw the Path as a Thick Line
        pts = np.array(path, np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=False, color=(255, 255, 0), thickness=6) # Cyan line
        # Start Dot
        cv2.circle(vis, path[0], 10, (0, 255, 0), -1) # Green Start
        # End Dot
        cv2.circle(vis, path[-1], 10, (0, 255, 255), -1) # Yellow Goal
    else:
        status = "NO PATH FOUND"

    # UI Text
    cv2.putText(vis, f"MODE: {status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, vis)
    print(f"Generated Plan: {output_path}")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    backbone, head = load_model(MODEL_PATH)
    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)
    transform = transforms.Compose([
        transforms.Resize((h, w)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process only 25 diverse images to show your portfolio
    images = sorted(os.listdir(TEST_IMAGE_DIR))[:25]
    
    print("Generating Navigation Portfolio...")
    for img in images:
        if img.endswith(".png"):
            process_image(os.path.join(TEST_IMAGE_DIR, img), 
                          os.path.join(OUTPUT_DIR, f"nav_plan_{img}"), 
                          backbone, head, transform)
    
    print(f"\nDONE! Check '{OUTPUT_DIR}' for your A* Road Maps.")

if __name__ == "__main__":
    main()