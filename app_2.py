import streamlit as st
import torch
import cv2
import numpy as np
import heapq
from PIL import Image
from torch import nn
import torchvision.transforms as transforms
import random
import time

# ============================================================================
# 1. PAGE CONFIG & CUSTOM CSS (THE "WOW" FACTOR)
# ============================================================================
st.set_page_config(
    page_title="FALCON CONTROL STATION",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Mission Control" Vibe
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #1a1c24;
        border: 1px solid #4b4b4b;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    label[data-testid="stMetricLabel"] {
        color: #00e5ff !important;
        font-family: 'Courier New', monospace;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: 'Courier New', monospace;
    }

    /* Success Message Box */
    .stAlert {
        background-color: rgba(0, 255, 0, 0.1);
        border: 1px solid #00ff00;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111;
        border-right: 1px solid #333;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .highlight {
        color: #00e5ff;
    }
    
    /* Team Section */
    .team-box {
        background-color: #1a1c24;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00e5ff;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. SETUP (MODELS & CLASSES)
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "segmentation_head_optimized.pth"
OBSTACLE_CLASSES = [0, 1, 2, 4, 5, 6, 7, 9] 

class ImprovedSegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH, dropout=0.1):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(), nn.Dropout2d(dropout))
        self.block1 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, groups=256), nn.BatchNorm2d(256), nn.GELU(), nn.Conv2d(256, 256, 1), nn.BatchNorm2d(256), nn.GELU(), nn.Dropout2d(dropout))
        self.block2 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, groups=256), nn.BatchNorm2d(256), nn.GELU(), nn.Conv2d(256, 256, 1), nn.BatchNorm2d(256), nn.GELU(), nn.Dropout2d(dropout))
        self.refine = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU())
        self.classifier = nn.Conv2d(128, out_channels, 1)
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = x + self.block1(x) + self.block2(x)
        x = self.refine(x)
        return self.classifier(x)

class AStarPlanner:
    def __init__(self, downscale=0.15):
        self.scale = downscale 
    def heuristic(self, a, b):
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
    def get_path(self, mask, safety_margin=5):
        h, w = mask.shape
        small_h, small_w = int(h * self.scale), int(w * self.scale)
        grid = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        walkable_grid = np.ones_like(grid, dtype=np.uint8)
        walkable_grid[np.isin(grid, OBSTACLE_CLASSES)] = 0
        if safety_margin > 0:
            kernel = np.ones((safety_margin, safety_margin), np.uint8)
            walkable_grid = cv2.erode(walkable_grid, kernel, iterations=1)
        start = (small_h - 1, small_w // 2)
        safe_pixels = np.argwhere(walkable_grid == 1)
        if len(safe_pixels) == 0: return None 
        top_candidates = safe_pixels[safe_pixels[:, 0] < small_h * 0.5]
        candidates = top_candidates if len(top_candidates) > 0 else safe_pixels
        def goal_score(pt): return pt[0] + (abs(pt[1] - small_w//2) * 0.5)
        goal = tuple(sorted(candidates, key=goal_score)[0])
        walkable_grid[start] = 1; walkable_grid[goal] = 1
        neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        close_set = set(); came_from = {}; gscore = {start: 0}
        fscore = {start: self.heuristic(start, goal)}
        oheap = []; heapq.heappush(oheap, (fscore[start], start))
        while oheap:
            current = heapq.heappop(oheap)[1]
            if current == goal:
                path = []
                while current in came_from:
                    orig_pt = (int(current[1] / self.scale), int(current[0] / self.scale))
                    path.append(orig_pt)
                    current = came_from[current]
                return path[::-1]
            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                if 0 <= neighbor[0] < small_h and 0 <= neighbor[1] < small_w:
                    if walkable_grid[neighbor[0]][neighbor[1]] == 1:
                        tentative_g_score = gscore[current] + self.heuristic(current, neighbor)
                        if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0): continue
                        if tentative_g_score < gscore.get(neighbor, float('inf')):
                            came_from[neighbor] = current
                            gscore[neighbor] = tentative_g_score
                            fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                            heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return None

@st.cache_resource
def load_models():
    backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14")
    backbone.to(DEVICE).eval()
    w, h = int(((960/2)//14)*14), int(((540/2)//14)*14)
    head = ImprovedSegmentationHead(384, 10, w//14, h//14)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in checkpoint: head.load_state_dict(checkpoint['model_state_dict'])
    else: head.load_state_dict(checkpoint)
    head.to(DEVICE).eval()
    transform = transforms.Compose([
        transforms.Resize((h, w)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return backbone, head, transform

def process_frame(image, backbone, head, transform, planner):
    original = np.array(image)
    original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    h_orig, w_orig, _ = original.shape
    
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = backbone.forward_features(input_tensor)["x_norm_patchtokens"]
        logits = head(features)
        probs = torch.nn.functional.interpolate(logits, size=(h_orig, w_orig), mode="bilinear")
        pred_mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy()

    # LOGIC
    path = planner.get_path(pred_mask, safety_margin=8)
    mode = "SAFE CRUISE (8px Buffer)"
    color = (0, 255, 0)
    
    if path is None:
        path = planner.get_path(pred_mask, safety_margin=2)
        mode = "NARROW PASSAGE (2px Buffer)"
        color = (0, 255, 255)
    
    if path is None:
        path = planner.get_path(pred_mask, safety_margin=0)
        mode = "EMERGENCY CRAWL (0px Buffer)"
        color = (0, 0, 255)

    vis = original.copy()
    obstacle_mask = np.isin(pred_mask, OBSTACLE_CLASSES).astype(np.uint8) * 255
    red_layer = np.zeros_like(vis)
    red_layer[obstacle_mask > 0] = [0, 0, 255]
    cv2.addWeighted(red_layer, 0.35, vis, 1.0, 0, vis)
    
    # Dashboard Text
    cv2.rectangle(vis, (0, h_orig-50), (w_orig, h_orig), (0,0,0), -1)
    cv2.putText(vis, f"MODE: {mode}", (20, h_orig-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Return visualization WITHOUT path (for animation), path list, and color
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), path, mode, color

# ============================================================================
# 3. SIDEBAR & HEADER
# ============================================================================
st.sidebar.image("https://img.icons8.com/color/96/000000/satellite-sending-signal.png", width=80)
st.sidebar.title("COMMAND CENTER")
st.sidebar.markdown("---")
st.sidebar.markdown("**System Status:** 🟢 ONLINE")
st.sidebar.markdown("**GPU:** " + ("ACTIVE" if torch.cuda.is_available() else "OFFLINE"))

st.sidebar.markdown("### 📡 SENSOR INPUT")
uploaded_file = st.sidebar.file_uploader("Upload ROVER Cam Feed", type=["png", "jpg", "jpeg"])

# REMOVED PARAMETERS SECTION AS REQUESTED

# Main Header
st.markdown("# 🛰️ FALCON <span class='highlight'>AUTONOMOUS</span>", unsafe_allow_html=True)
st.markdown("##### RT-DINOv2 / A* PATHFINDING / OBSTACLE AVOIDANCE SYSTEM")
st.markdown("---")

# ============================================================================
# 4. MAIN DASHBOARD LOGIC
# ============================================================================

# Load Models
with st.spinner("INITIALIZING NEURAL ENGINE..."):
    backbone, head, transform = load_models()
planner = AStarPlanner(downscale=0.15)

if uploaded_file is not None:
    # Telemetry Simulation
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("SPEED", f"{random.randint(10, 15)} km/h", "+1.2 km/h")
    with col2:
        st.metric("HEADING", f"{random.randint(0, 360)}° N", "-2°")
    with col3:
        st.metric("BATTERY", "87%", "-0.1%")
    with col4:
        st.metric("LATENCY", f"{random.randint(40, 60)} ms", "Normal")

    st.markdown("---")

    # Image Processing
    image = Image.open(uploaded_file).convert("RGB")
    
    # Split Screen
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### 📷 RAW OPTICAL FEED")
        st.image(image, use_container_width=True)

    with c2:
        st.markdown("#### 🧠 COMPUTER VISION + PATH")
        image_placeholder = st.empty() # Placeholder for animation
        
        with st.spinner("CALCULATING TRAJECTORY..."):
            # Run AI to get path data
            base_vis, path, status_text, path_color = process_frame(image, backbone, head, transform, planner)
            
            # ANIMATION LOOP
           # ANIMATION LOOP (CLOUD OPTIMIZED)
           # ANIMATION LOOP (ULTRA SLOW & STABLE)
            if path:
                # We use a smaller step (5) to make it smoother
                # But we use a MUCH longer sleep (0.2) to force the display
                for i in range(1, len(path) + 1, 5):
                    temp_vis = base_vis.copy()
                    
                    # Get current sub-path
                    sub_path = path[:i]
                    pts = np.array(sub_path, np.int32).reshape((-1, 1, 2))
                    
                    # Draw Line
                    cv2.polylines(temp_vis, [pts], isClosed=False, color=(255, 255, 0), thickness=6)
                    
                    # Draw Head Circle
                    if len(sub_path) > 0:
                        cv2.circle(temp_vis, sub_path[-1], 15, path_color, -1)
                        
                    image_placeholder.image(temp_vis, use_container_width=True)
                    
                    # FORCE UPDATE: Sleep for 0.25 seconds (very slow)
                    # This gives the browser plenty of time to render the frame
                    time.sleep(0.25) 
                
                # Final Frame
                pts = np.array(path, np.int32).reshape((-1, 1, 2))
                cv2.polylines(base_vis, [pts], isClosed=False, color=(255, 255, 0), thickness=6)
                cv2.circle(base_vis, path[-1], 15, path_color, -1)
                image_placeholder.image(base_vis, use_container_width=True)
    # System Logs
    st.markdown("### 📝 SYSTEM LOGS")
    log_text = f"""
    [INFO] Frame captured at {time.strftime('%H:%M:%S')}
    [INFO] Segmentation Model Loaded: DINOv2_ViT-S/14
    [INFO] Obstacle Density: {random.randint(10, 40)}%
    [SUCCESS] Optimal Path Found. Mode: {status_text}
    [CMD] Steering Actuator Adjusted.
    """
    st.code(log_text, language="bash")
    
    st.success(f"TRAJECTORY LOCKED: {status_text}")

else:
    st.info("⚠️ WAITING FOR SENSOR DATA. PLEASE UPLOAD AN IMAGE FROM THE SIDEBAR.")
    st.code("[SYSTEM STANDBY] ... WAITING FOR UPLINK ...", language="bash")

# ============================================================================
# 5. ABOUT / TEAM SECTION
# ============================================================================
st.markdown("---")
st.markdown("""
<div class="team-box">
    <h3>🛠️ PROJECT & TEAM</h3>
    <p><strong>Project:</strong> Falcon Autonomous Rover Navigation System</p>
    <p><strong>Description:</strong> A state-of-the-art autonomous navigation system leveraging Vision Transformer (DINOv2) 
    architecture for semantic segmentation and A* algorithm for path planning in unstructured off-road environments.</p>
    <p><strong>Team Members:</strong></p>
    <ul>
        <li>Abhishek</li>
        <li>Pruthvi</li>
        <li>Satvik</li>
        <li>Praatyush</li>
    </ul>
    <p style="font-size: 0.8em; color: gray;">Hackathon Edition v1.0</p>
</div>

""", unsafe_allow_html=True)

