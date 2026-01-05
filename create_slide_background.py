#!/usr/bin/env python3
"""
Create a blurred, light-toned background for the pitch slide.
Fitness/AI themed with soft gradients and subtle shapes.
"""

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import os

# Slide dimensions (16:9 ratio, standard presentation)
WIDTH = 1920
HEIGHT = 1080

def create_gradient_background():
    """Create a light-toned gradient background with fitness/AI theme."""
    
    # Create base image with very light background
    base_color = '#fafbfc'  # Almost white, very light gray
    img = Image.new('RGB', (WIDTH, HEIGHT), color=base_color)
    
    # Convert to RGBA for alpha blending
    img = img.convert('RGBA')
    
    # Light color palette (fitness/AI themed - very soft pastels)
    # Soft blues, light oranges, subtle purples - all very light
    colors = [
        '#e8f2ff',  # Very light blue (fitness/tech)
        '#fff5e8',  # Very light cream/orange (warmth)
        '#f0f4ff',  # Very light blue-purple (AI/tech)
        '#fff8f0',  # Very light warm white
        '#e8f8ff',  # Very light sky blue
    ]
    
    # Create radial gradient effect from multiple points
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    max_radius = int(np.sqrt(WIDTH**2 + HEIGHT**2) * 0.6)
    
    # Gradient centers (top-left, top-right, bottom-left, bottom-right, center)
    gradient_centers = [
        (WIDTH * 0.25, HEIGHT * 0.25),  # Top-left
        (WIDTH * 0.75, HEIGHT * 0.25),  # Top-right
        (WIDTH * 0.25, HEIGHT * 0.75),  # Bottom-left
        (WIDTH * 0.75, HEIGHT * 0.75),  # Bottom-right
        (center_x, center_y),            # Center
    ]
    
    # Draw radial gradients from each center
    for center_idx, (cx, cy) in enumerate(gradient_centers):
        color = colors[center_idx % len(colors)]
        rgb = hex_to_rgb(color)
        
        # Create temporary image for this gradient
        temp_img = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        
        # Draw concentric circles with decreasing opacity
        num_circles = 15
        for i in range(num_circles):
            radius = max_radius * (i + 1) / num_circles
            # Opacity decreases from center to edge
            alpha = int(25 * (1 - i / num_circles) * 0.5)  # Very subtle
            
            if alpha > 0:
                color_rgba = (*rgb, alpha)
                temp_draw.ellipse(
                    [cx - radius, cy - radius, cx + radius, cy + radius],
                    fill=color_rgba
                )
        
        # Blend with base image
        img = Image.alpha_composite(img, temp_img)
    
    # Add very subtle geometric shapes (fitness/exercise themed)
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Very subtle circles (representing motion/tracking points)
    circle_positions = [
        (WIDTH * 0.15, HEIGHT * 0.2),
        (WIDTH * 0.85, HEIGHT * 0.25),
        (WIDTH * 0.2, HEIGHT * 0.8),
        (WIDTH * 0.8, HEIGHT * 0.75),
        (WIDTH * 0.5, HEIGHT * 0.15),
        (WIDTH * 0.5, HEIGHT * 0.85),
    ]
    
    for x, y in circle_positions:
        radius = 120 + np.random.randint(-30, 30)
        color_rgba = (*hex_to_rgb('#d0e0ff'), 12)  # Very light blue, very transparent
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                    fill=color_rgba, outline=None)
    
    # Very subtle connecting lines (representing body landmarks/skeleton structure)
    line_pairs = [
        ((WIDTH * 0.1, HEIGHT * 0.3), (WIDTH * 0.3, HEIGHT * 0.25)),
        ((WIDTH * 0.7, HEIGHT * 0.2), (WIDTH * 0.9, HEIGHT * 0.3)),
        ((WIDTH * 0.15, HEIGHT * 0.7), (WIDTH * 0.3, HEIGHT * 0.75)),
        ((WIDTH * 0.7, HEIGHT * 0.65), (WIDTH * 0.85, HEIGHT * 0.7)),
    ]
    
    for (start_x, start_y), (end_x, end_y) in line_pairs:
        color_rgba = (*hex_to_rgb('#ffe8d0'), 10)  # Very light orange, very transparent
        draw.line([start_x, start_y, end_x, end_y], fill=color_rgba, width=2)
    
    return img.convert('RGB')

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def apply_blur_and_final_touches(img):
    """Apply blur effect and final adjustments."""
    
    # Apply strong Gaussian blur for soft, dreamy effect
    # High radius for smooth, non-distracting background
    blurred = img.filter(ImageFilter.GaussianBlur(radius=100))
    
    # Blend original with blurred (80% blurred, 20% original for subtle texture)
    result = Image.blend(blurred, img, alpha=0.8)
    
    # Slight brightness increase for lighter, more open feel
    arr = np.array(result).astype(np.float32)
    arr = arr * 1.08  # Slight brightening (ensures very light tone)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    result = Image.fromarray(arr)
    
    # Final slight blur pass for extra smoothness
    result = result.filter(ImageFilter.GaussianBlur(radius=20))
    
    return result

def main():
    """Main function to create the background."""
    print("Creating pitch slide background...")
    
    # Create gradient background
    print("  → Creating gradient base...")
    background = create_gradient_background()
    
    # Apply blur and final touches
    print("  → Applying blur effect...")
    final_background = apply_blur_and_final_touches(background)
    
    # Save the image
    output_path = "pitch_slide_background.png"
    final_background.save(output_path, "PNG", quality=95)
    print(f"  ✅ Background saved to: {output_path}")
    print(f"  → Dimensions: {WIDTH}x{HEIGHT} (16:9 ratio)")
    print(f"  → File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print("\n💡 Tip: Use this as the background image in PowerPoint/Google Slides")
    print("   Set transparency/opacity to 100% (fully opaque) for best effect")

if __name__ == "__main__":
    main()

