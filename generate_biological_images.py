"""
Generate Realistic Biological Images for BioFlow Demo
======================================================

Creates synthetic but realistic:
- Western blot images with band patterns
- Gel electrophoresis images  
- Fluorescence microscopy images
- Cell microscopy images

Each image includes rich experimental metadata for search and filtering.
"""

import os
import json
import random
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np

# Output directory
DATA_DIR = Path(__file__).parent / "data" / "images" / "biological"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
random.seed(42)
np.random.seed(42)

# ============================================================================
# EXPERIMENTAL METADATA GENERATORS
# ============================================================================

CELL_LINES = [
    "HeLa", "A549", "MCF7", "HEK293", "PC3", "U2OS", "HCT116", "MDA-MB-231",
    "A431", "H1975", "HCC827", "SW480", "Jurkat", "K562", "THP-1", "RAW264.7"
]

TREATMENTS = [
    {"drug": "Gefitinib", "target": "EGFR", "concentration": "100nM"},
    {"drug": "Imatinib", "target": "BCR-ABL", "concentration": "1μM"},
    {"drug": "Sorafenib", "target": "VEGFR/RAF", "concentration": "5μM"},
    {"drug": "Erlotinib", "target": "EGFR", "concentration": "500nM"},
    {"drug": "Lapatinib", "target": "HER2/EGFR", "concentration": "1μM"},
    {"drug": "Dasatinib", "target": "SRC/ABL", "concentration": "100nM"},
    {"drug": "Nilotinib", "target": "BCR-ABL", "concentration": "500nM"},
    {"drug": "Crizotinib", "target": "ALK/MET", "concentration": "250nM"},
    {"drug": "Vemurafenib", "target": "BRAF V600E", "concentration": "1μM"},
    {"drug": "Trametinib", "target": "MEK1/2", "concentration": "100nM"},
    {"drug": "Rapamycin", "target": "mTOR", "concentration": "100nM"},
    {"drug": "Staurosporine", "target": "Pan-kinase", "concentration": "1μM"},
    {"drug": "DMSO", "target": "Vehicle control", "concentration": "0.1%"},
]

TARGET_PROTEINS = [
    {"name": "pEGFR (Y1068)", "mw": 170, "function": "RTK signaling"},
    {"name": "total EGFR", "mw": 170, "function": "RTK"},
    {"name": "pAKT (S473)", "mw": 60, "function": "PI3K signaling"},
    {"name": "total AKT", "mw": 60, "function": "Kinase"},
    {"name": "pERK1/2 (T202/Y204)", "mw": 42, "function": "MAPK signaling"},
    {"name": "total ERK1/2", "mw": 42, "function": "MAPK"},
    {"name": "cleaved PARP", "mw": 89, "function": "Apoptosis marker"},
    {"name": "cleaved Caspase-3", "mw": 17, "function": "Apoptosis"},
    {"name": "p53", "mw": 53, "function": "Tumor suppressor"},
    {"name": "p21", "mw": 21, "function": "Cell cycle inhibitor"},
    {"name": "Cyclin D1", "mw": 36, "function": "Cell cycle"},
    {"name": "β-actin", "mw": 42, "function": "Loading control"},
    {"name": "GAPDH", "mw": 37, "function": "Loading control"},
    {"name": "BCL-2", "mw": 26, "function": "Anti-apoptotic"},
    {"name": "BAX", "mw": 21, "function": "Pro-apoptotic"},
]

OUTCOMES = ["positive", "negative", "inconclusive", "dose_dependent"]

STAINING_METHODS = [
    {"name": "Annexin V-FITC / PI", "purpose": "Apoptosis detection", "colors": ["green", "red"]},
    {"name": "DAPI", "purpose": "Nuclear stain", "colors": ["blue"]},
    {"name": "Hoechst 33342", "purpose": "DNA stain", "colors": ["blue"]},
    {"name": "MitoTracker Red", "purpose": "Mitochondria", "colors": ["red"]},
    {"name": "Phalloidin-FITC", "purpose": "Actin cytoskeleton", "colors": ["green"]},
    {"name": "CellMask Orange", "purpose": "Plasma membrane", "colors": ["orange"]},
    {"name": "LysoTracker Green", "purpose": "Lysosomes", "colors": ["green"]},
    {"name": "ER-Tracker Blue", "purpose": "Endoplasmic reticulum", "colors": ["blue"]},
]


def generate_experiment_id() -> str:
    """Generate a realistic experiment ID."""
    prefix = random.choice(["EXP", "WB", "IF", "FC", "GEL"])
    date = datetime.now() - timedelta(days=random.randint(1, 365))
    num = random.randint(1, 999)
    return f"{prefix}-{date.strftime('%Y%m%d')}-{num:03d}"


def generate_date_metadata() -> Dict[str, str]:
    """Generate experiment date metadata."""
    exp_date = datetime.now() - timedelta(days=random.randint(1, 365))
    return {
        "experiment_date": exp_date.strftime("%Y-%m-%d"),
        "acquisition_date": (exp_date + timedelta(hours=random.randint(1, 48))).strftime("%Y-%m-%d %H:%M"),
    }


# ============================================================================
# IMAGE GENERATORS
# ============================================================================

def create_western_blot(
    width: int = 400,
    height: int = 300,
    num_lanes: int = 6,
    bands_config: List[Dict] = None
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Generate a realistic Western blot image.
    
    Returns:
        (image, metadata) tuple
    """
    # Create dark background (like X-ray film)
    img = Image.new('L', (width, height), color=20)
    draw = ImageDraw.Draw(img)
    
    # Lane parameters
    lane_width = (width - 40) // num_lanes
    lane_spacing = lane_width + 5
    start_x = 20
    
    # Generate band configuration if not provided
    if bands_config is None:
        # Random protein targets
        proteins = random.sample(TARGET_PROTEINS, min(3, len(TARGET_PROTEINS)))
        treatment = random.choice(TREATMENTS)
        
        bands_config = []
        for i in range(num_lanes):
            lane_bands = []
            for protein in proteins:
                # Intensity varies by lane (dose-response or treatment effect)
                base_intensity = random.randint(150, 255)
                if i == 0:  # Control
                    intensity = base_intensity
                else:
                    # Treatment effect (could increase or decrease)
                    effect = random.choice([-1, 1]) * random.uniform(0.2, 0.8) * i / num_lanes
                    intensity = int(base_intensity * (1 + effect))
                    intensity = max(50, min(255, intensity))
                
                lane_bands.append({
                    "mw": protein["mw"],
                    "intensity": intensity,
                    "name": protein["name"]
                })
            bands_config.append(lane_bands)
    
    # Draw lanes and bands
    for lane_idx in range(num_lanes):
        lane_x = start_x + lane_idx * lane_spacing
        
        # Draw lane background (slightly lighter)
        draw.rectangle([lane_x, 30, lane_x + lane_width - 5, height - 30], fill=25)
        
        # Draw bands
        for band in bands_config[lane_idx]:
            # Y position based on molecular weight (log scale)
            y_pos = int(40 + (height - 80) * (1 - math.log10(band["mw"]) / 2.5))
            y_pos = max(40, min(height - 40, y_pos))
            
            # Band thickness varies with intensity
            band_height = random.randint(8, 15)
            
            # Draw the band with Gaussian-like profile
            for dy in range(-band_height//2, band_height//2 + 1):
                # Gaussian falloff
                falloff = math.exp(-(dy**2) / (2 * (band_height/4)**2))
                pixel_intensity = int(band["intensity"] * falloff)
                
                # Add some horizontal variation
                for dx in range(lane_width - 5):
                    noise = random.randint(-10, 10)
                    final_intensity = max(0, min(255, pixel_intensity + noise))
                    try:
                        img.putpixel((lane_x + dx, y_pos + dy), final_intensity)
                    except:
                        pass
    
    # Add molecular weight markers on the left
    marker_mws = [250, 150, 100, 75, 50, 37, 25, 20, 15]
    for mw in marker_mws:
        y = int(40 + (height - 80) * (1 - math.log10(mw) / 2.5))
        if 35 < y < height - 35:
            draw.line([(5, y), (15, y)], fill=200, width=2)
    
    # Apply slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    
    # Add film grain noise
    img_array = np.array(img)
    noise = np.random.normal(0, 5, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to RGB for consistency
    img = img.convert('RGB')
    
    # Generate metadata
    cell_line = random.choice(CELL_LINES)
    treatment = random.choice(TREATMENTS)
    target = random.choice(TARGET_PROTEINS)
    outcome = random.choice(OUTCOMES)
    
    metadata = {
        "image_type": "western_blot",
        "experiment_id": generate_experiment_id(),
        "experiment_type": "Western Blot",
        "cell_line": cell_line,
        "target_protein": target["name"],
        "target_mw": f"{target['mw']} kDa",
        "target_function": target["function"],
        "treatment": treatment["drug"],
        "treatment_target": treatment["target"],
        "concentration": treatment["concentration"],
        "conditions": {
            "drug": treatment["drug"],
            "concentration": treatment["concentration"],
            "time": f"{random.choice([6, 12, 24, 48, 72])}h",
            "replicates": random.randint(2, 4)
        },
        "outcome": outcome,
        "protocol": f"Standard Western protocol with {random.choice([8, 10, 12])}% SDS-PAGE, transfer to {random.choice(['PVDF', 'nitrocellulose'])}, primary antibody 1:{random.choice([500, 1000, 2000])}, HRP-secondary 1:5000",
        "notes": _generate_western_notes(outcome, target, treatment),
        "quality_score": round(random.uniform(0.7, 0.98), 2),
        "num_lanes": num_lanes,
        **generate_date_metadata()
    }
    
    return img, metadata


def _generate_western_notes(outcome: str, target: Dict, treatment: Dict) -> str:
    """Generate realistic lab notes for Western blot."""
    notes = []
    
    if outcome == "positive":
        notes.append(f"Clear {target['name']} band detected at expected MW ({target['mw']} kDa).")
        notes.append(f"{treatment['drug']} treatment shows dose-dependent effect.")
    elif outcome == "negative":
        notes.append(f"No significant change in {target['name']} levels observed.")
        notes.append("Consider repeating with higher drug concentration.")
    elif outcome == "dose_dependent":
        notes.append(f"Dose-dependent inhibition of {target['name']}.")
        notes.append(f"IC50 approximately {random.choice(['10', '50', '100', '500'])}nM.")
    else:
        notes.append("Results inconclusive. High background noise.")
        notes.append("Recommend optimizing blocking conditions.")
    
    notes.append(f"Loading control ({random.choice(['β-actin', 'GAPDH'])}) unchanged.")
    
    return " ".join(notes)


def create_gel_electrophoresis(
    width: int = 400,
    height: int = 300,
    num_lanes: int = 8
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Generate a realistic agarose/SDS-PAGE gel image.
    """
    # Create background (blue-ish for Coomassie or UV for DNA)
    gel_type = random.choice(["coomassie", "sypro", "ethidium_bromide"])
    
    if gel_type == "coomassie":
        bg_color = (240, 240, 255)  # Light blue tint
        band_color = (30, 30, 150)  # Dark blue
    elif gel_type == "sypro":
        bg_color = (20, 20, 20)  # Dark
        band_color = (255, 150, 50)  # Orange-ish
    else:  # ethidium bromide (DNA gel)
        bg_color = (30, 10, 10)  # Dark with slight red
        band_color = (255, 120, 50)  # Orange glow
    
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    lane_width = (width - 40) // num_lanes
    start_x = 20
    
    # Generate random band patterns
    for lane_idx in range(num_lanes):
        lane_x = start_x + lane_idx * lane_width
        
        # Number of bands per lane
        num_bands = random.randint(3, 8)
        
        # First lane is often marker
        if lane_idx == 0:
            # Standard ladder pattern
            ladder_sizes = [250, 150, 100, 75, 50, 37, 25, 20, 15, 10]
            for size in ladder_sizes:
                y = int(40 + (height - 80) * (size / 250))
                band_height = random.randint(3, 6)
                intensity = 200 + random.randint(-20, 20)
                
                for dy in range(-band_height//2, band_height//2 + 1):
                    for dx in range(lane_width - 10):
                        try:
                            if gel_type == "coomassie":
                                img.putpixel((lane_x + dx + 5, y + dy), 
                                           (band_color[0], band_color[1], band_color[2]))
                            else:
                                factor = abs(dy) / (band_height/2 + 0.1)
                                r = int(band_color[0] * (1 - factor * 0.5))
                                g = int(band_color[1] * (1 - factor * 0.5))
                                b = int(band_color[2] * (1 - factor * 0.5))
                                img.putpixel((lane_x + dx + 5, y + dy), (r, g, b))
                        except:
                            pass
        else:
            # Random bands
            band_positions = sorted([random.randint(50, height - 50) for _ in range(num_bands)])
            for y in band_positions:
                band_height = random.randint(4, 10)
                intensity = random.randint(100, 255)
                
                for dy in range(-band_height//2, band_height//2 + 1):
                    for dx in range(lane_width - 10):
                        noise = random.randint(-15, 15)
                        try:
                            if gel_type == "coomassie":
                                c = max(0, min(255, band_color[2] - intensity // 3 + noise))
                                img.putpixel((lane_x + dx + 5, y + dy), (30, 30, c))
                            else:
                                r = max(0, min(255, band_color[0] * intensity // 255 + noise))
                                g = max(0, min(255, band_color[1] * intensity // 255 + noise))
                                b = max(0, min(255, band_color[2] * intensity // 255 + noise))
                                img.putpixel((lane_x + dx + 5, y + dy), (r, g, b))
                        except:
                            pass
    
    # Apply blur
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Metadata
    protein = random.choice(TARGET_PROTEINS)
    experiment_type = random.choice(["Protein purification", "Expression analysis", "Pull-down assay", "Co-IP"])
    
    metadata = {
        "image_type": "gel",
        "experiment_id": generate_experiment_id(),
        "experiment_type": f"SDS-PAGE ({gel_type.replace('_', ' ').title()})",
        "gel_percentage": f"{random.choice([8, 10, 12, 15])}%",
        "stain": gel_type.replace("_", " ").title(),
        "protein": protein["name"],
        "expression_system": random.choice(["E. coli BL21", "Sf9 insect cells", "HEK293", "CHO"]),
        "purpose": experiment_type,
        "conditions": {
            "gel_type": "SDS-PAGE",
            "running_buffer": "Tris-Glycine",
            "voltage": f"{random.choice([100, 120, 150])}V"
        },
        "outcome": random.choice(OUTCOMES),
        "notes": f"{experiment_type} of {protein['name']}. {random.choice(['High purity achieved.', 'Multiple bands suggest partial degradation.', 'Expected MW confirmed.', 'Yield approximately 2mg/L.'])}",
        "quality_score": round(random.uniform(0.65, 0.95), 2),
        "num_lanes": num_lanes,
        **generate_date_metadata()
    }
    
    return img, metadata


def create_fluorescence_microscopy(
    width: int = 400,
    height: int = 400,
    num_cells: int = None
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Generate a realistic fluorescence microscopy image.
    """
    # Dark background
    img = Image.new('RGB', (width, height), color=(10, 10, 15))
    
    # Select staining method
    staining = random.choice(STAINING_METHODS)
    
    # Number of cells
    if num_cells is None:
        num_cells = random.randint(8, 25)
    
    # Color mapping
    color_map = {
        "green": (50, 255, 100),
        "red": (255, 80, 80),
        "blue": (80, 150, 255),
        "orange": (255, 180, 50),
    }
    
    # Draw cells
    for _ in range(num_cells):
        cx = random.randint(30, width - 30)
        cy = random.randint(30, height - 30)
        
        # Cell size
        cell_radius = random.randint(15, 35)
        
        # Draw nucleus (usually blue/DAPI)
        nucleus_radius = cell_radius // 2
        if "blue" in staining["colors"] or random.random() > 0.3:
            _draw_gradient_circle(img, cx, cy, nucleus_radius, color_map["blue"], intensity=0.8)
        
        # Draw cytoplasmic staining
        for color_name in staining["colors"]:
            if color_name == "blue":
                continue  # Already drew nucleus
            color = color_map.get(color_name, (200, 200, 200))
            
            if staining["purpose"] in ["Apoptosis detection", "Plasma membrane"]:
                # Ring pattern for membrane/apoptosis
                _draw_ring(img, cx, cy, cell_radius, cell_radius - 5, color, intensity=0.7)
            elif staining["purpose"] in ["Mitochondria", "Lysosomes"]:
                # Punctate pattern
                for _ in range(random.randint(10, 30)):
                    px = cx + random.randint(-cell_radius, cell_radius)
                    py = cy + random.randint(-cell_radius, cell_radius)
                    if (px - cx)**2 + (py - cy)**2 < cell_radius**2:
                        _draw_gradient_circle(img, px, py, random.randint(2, 5), color, intensity=0.6)
            else:
                # Diffuse cytoplasmic
                _draw_gradient_circle(img, cx, cy, cell_radius, color, intensity=0.4)
    
    # Add noise
    img_array = np.array(img)
    noise = np.random.normal(0, 3, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    # Slight blur
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Metadata
    cell_line = random.choice(CELL_LINES)
    treatment = random.choice(TREATMENTS)
    outcome = random.choice(OUTCOMES)
    
    metadata = {
        "image_type": "microscopy",
        "experiment_id": generate_experiment_id(),
        "experiment_type": "Fluorescence Microscopy",
        "staining": staining["name"],
        "staining_purpose": staining["purpose"],
        "channels": staining["colors"],
        "cell_line": cell_line,
        "treatment": treatment["drug"],
        "treatment_target": treatment["target"],
        "concentration": treatment["concentration"],
        "conditions": {
            "drug": treatment["drug"],
            "concentration": treatment["concentration"],
            "time": f"{random.choice([6, 12, 24, 48, 72])}h",
        },
        "magnification": f"{random.choice([10, 20, 40, 63])}x",
        "objective": random.choice(["Plan-Apochromat", "EC Plan-Neofluar", "LD Plan-Neofluar"]),
        "microscope": random.choice(["Zeiss LSM 880", "Nikon A1R", "Leica SP8", "Olympus FV3000"]),
        "outcome": outcome,
        "notes": _generate_microscopy_notes(outcome, staining, treatment, num_cells),
        "quality_score": round(random.uniform(0.7, 0.98), 2),
        "cell_count": num_cells,
        **generate_date_metadata()
    }
    
    return img, metadata


def _draw_gradient_circle(img: Image.Image, cx: int, cy: int, radius: int, 
                          color: Tuple[int, int, int], intensity: float = 1.0):
    """Draw a gradient-filled circle (Gaussian falloff)."""
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            dist = math.sqrt(dx**2 + dy**2)
            if dist <= radius:
                # Gaussian falloff
                falloff = math.exp(-(dist**2) / (2 * (radius/2)**2)) * intensity
                x, y = cx + dx, cy + dy
                if 0 <= x < img.width and 0 <= y < img.height:
                    current = img.getpixel((x, y))
                    new_color = (
                        min(255, int(current[0] + color[0] * falloff)),
                        min(255, int(current[1] + color[1] * falloff)),
                        min(255, int(current[2] + color[2] * falloff))
                    )
                    img.putpixel((x, y), new_color)


def _draw_ring(img: Image.Image, cx: int, cy: int, outer_r: int, inner_r: int,
               color: Tuple[int, int, int], intensity: float = 1.0):
    """Draw a ring (membrane-like pattern)."""
    for dx in range(-outer_r, outer_r + 1):
        for dy in range(-outer_r, outer_r + 1):
            dist = math.sqrt(dx**2 + dy**2)
            if inner_r <= dist <= outer_r:
                x, y = cx + dx, cy + dy
                if 0 <= x < img.width and 0 <= y < img.height:
                    current = img.getpixel((x, y))
                    new_color = (
                        min(255, int(current[0] + color[0] * intensity)),
                        min(255, int(current[1] + color[1] * intensity)),
                        min(255, int(current[2] + color[2] * intensity))
                    )
                    img.putpixel((x, y), new_color)


def _generate_microscopy_notes(outcome: str, staining: Dict, treatment: Dict, cell_count: int) -> str:
    """Generate realistic lab notes for microscopy."""
    notes = []
    
    if staining["purpose"] == "Apoptosis detection":
        if outcome == "positive":
            percent = random.randint(40, 80)
            notes.append(f"~{percent}% apoptotic cells at 48h.")
            notes.append("Green = early apoptosis (Annexin V+/PI-), Red = late apoptosis.")
        elif outcome == "negative":
            notes.append("Minimal apoptosis (<10%). No significant drug effect observed.")
        else:
            notes.append("Variable results. Some cells show early apoptotic markers.")
    elif staining["purpose"] == "Mitochondria":
        notes.append(f"Mitochondrial network visualized in {cell_count} cells.")
        if outcome == "positive":
            notes.append("Drug treatment causes mitochondrial fragmentation.")
    else:
        notes.append(f"{staining['purpose']} imaging in {cell_count} cells.")
    
    notes.append(f"Treatment: {treatment['drug']} {treatment['concentration']}.")
    
    return " ".join(notes)


def create_brightfield_microscopy(
    width: int = 400,
    height: int = 400
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Generate a brightfield/phase contrast microscopy image.
    """
    # Light background
    bg_color = (230, 225, 220)
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw cells (dark outlines, lighter interior)
    num_cells = random.randint(15, 40)
    
    for _ in range(num_cells):
        cx = random.randint(20, width - 20)
        cy = random.randint(20, height - 20)
        
        # Cell shape (somewhat irregular)
        points = []
        base_radius = random.randint(12, 30)
        num_points = 12
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            r = base_radius + random.randint(-5, 5)
            x = cx + int(r * math.cos(angle))
            y = cy + int(r * math.sin(angle))
            points.append((x, y))
        
        # Draw cell with dark outline
        cell_color = (200 + random.randint(-20, 10), 195 + random.randint(-20, 10), 190 + random.randint(-20, 10))
        outline_color = (80 + random.randint(-20, 20), 75 + random.randint(-20, 20), 70 + random.randint(-20, 20))
        
        draw.polygon(points, fill=cell_color, outline=outline_color)
        
        # Draw nucleus (darker spot)
        nucleus_radius = base_radius // 3
        nucleus_color = (100 + random.randint(-20, 20), 95 + random.randint(-20, 20), 90 + random.randint(-20, 20))
        draw.ellipse([cx - nucleus_radius, cy - nucleus_radius, 
                      cx + nucleus_radius, cy + nucleus_radius], fill=nucleus_color)
    
    # Add texture/noise
    img_array = np.array(img)
    noise = np.random.normal(0, 3, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    # Slight blur
    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
    
    cell_line = random.choice(CELL_LINES)
    treatment = random.choice(TREATMENTS)
    
    metadata = {
        "image_type": "microscopy",
        "experiment_id": generate_experiment_id(),
        "experiment_type": "Brightfield Microscopy",
        "imaging_mode": random.choice(["Phase Contrast", "DIC", "Brightfield"]),
        "cell_line": cell_line,
        "treatment": treatment["drug"],
        "concentration": treatment["concentration"],
        "conditions": {
            "drug": treatment["drug"],
            "concentration": treatment["concentration"],
            "time": f"{random.choice([6, 12, 24, 48, 72])}h",
        },
        "magnification": f"{random.choice([10, 20, 40])}x",
        "outcome": random.choice(OUTCOMES),
        "notes": f"Cell morphology assessment. {num_cells} cells visible. " + random.choice([
            "Cells appear healthy with normal morphology.",
            "Some cells show rounded morphology suggesting stress.",
            "Increased cell density observed.",
            "Drug-treated cells show elongated morphology."
        ]),
        "quality_score": round(random.uniform(0.7, 0.95), 2),
        "cell_count": num_cells,
        **generate_date_metadata()
    }
    
    return img, metadata


# ============================================================================
# MAIN GENERATION FUNCTIONS
# ============================================================================

def generate_dataset(
    num_western_blots: int = 25,
    num_gels: int = 15,
    num_fluorescence: int = 25,
    num_brightfield: int = 15
) -> List[Dict[str, Any]]:
    """
    Generate a complete dataset of biological images.
    
    Returns:
        List of metadata dicts with 'file_path' added
    """
    manifest = []
    total = num_western_blots + num_gels + num_fluorescence + num_brightfield
    count = 0
    
    print(f"\n{'='*60}")
    print("BioFlow Biological Image Generator")
    print(f"{'='*60}")
    print(f"Target: {total} images")
    print(f"  - Western Blots: {num_western_blots}")
    print(f"  - Gel Electrophoresis: {num_gels}")
    print(f"  - Fluorescence Microscopy: {num_fluorescence}")
    print(f"  - Brightfield Microscopy: {num_brightfield}")
    print(f"Output: {DATA_DIR}")
    print()
    
    # Generate Western Blots
    print("🧬 Generating Western Blots...")
    for i in range(num_western_blots):
        img, metadata = create_western_blot()
        filename = f"western_blot_{i+1:03d}.png"
        filepath = DATA_DIR / filename
        img.save(filepath)
        
        metadata["file_path"] = str(filepath)
        metadata["filename"] = filename
        manifest.append(metadata)
        count += 1
        print(f"  [{count}/{total}] {filename}")
    
    # Generate Gels
    print("\n🧫 Generating Gel Electrophoresis...")
    for i in range(num_gels):
        img, metadata = create_gel_electrophoresis()
        filename = f"gel_electrophoresis_{i+1:03d}.png"
        filepath = DATA_DIR / filename
        img.save(filepath)
        
        metadata["file_path"] = str(filepath)
        metadata["filename"] = filename
        manifest.append(metadata)
        count += 1
        print(f"  [{count}/{total}] {filename}")
    
    # Generate Fluorescence Microscopy
    print("\n🔬 Generating Fluorescence Microscopy...")
    for i in range(num_fluorescence):
        img, metadata = create_fluorescence_microscopy()
        filename = f"fluorescence_{i+1:03d}.png"
        filepath = DATA_DIR / filename
        img.save(filepath)
        
        metadata["file_path"] = str(filepath)
        metadata["filename"] = filename
        manifest.append(metadata)
        count += 1
        print(f"  [{count}/{total}] {filename}")
    
    # Generate Brightfield Microscopy
    print("\n📷 Generating Brightfield Microscopy...")
    for i in range(num_brightfield):
        img, metadata = create_brightfield_microscopy()
        filename = f"brightfield_{i+1:03d}.png"
        filepath = DATA_DIR / filename
        img.save(filepath)
        
        metadata["file_path"] = str(filepath)
        metadata["filename"] = filename
        manifest.append(metadata)
        count += 1
        print(f"  [{count}/{total}] {filename}")
    
    # Save manifest
    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Generated {len(manifest)} images")
    print(f"📄 Manifest saved to: {manifest_path}")
    print(f"{'='*60}")
    
    return manifest


if __name__ == "__main__":
    # Generate default dataset
    manifest = generate_dataset()
    
    print("\nSample metadata:")
    print(json.dumps(manifest[0], indent=2))
