#!/usr/bin/env python3
"""
IMU CSV Data Plotter
Plots each attribute separately with timestamp on X-axis
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def plot_imu_csv(csv_path: str):
    """Load and plot each IMU attribute separately with timestamp on X-axis."""
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Convert timestamp to relative time (seconds from start) for cleaner X-axis
    df['time'] = df['timestamp'] - df['timestamp'].min()
    
    # Get unique nodes
    nodes = df['node_name'].unique()
    node_colors = {
        'left_wrist': '#3b82f6',   # Blue
        'right_wrist': '#a855f7',  # Purple
        'chest': '#f97316'         # Orange
    }
    
    # All attributes to plot (excluding timestamp, node_id, node_name, rep_number)
    attributes = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'qw', 'qx', 'qy', 'qz', 'roll', 'pitch', 'yaw']
    
    print(f"📊 Loaded {len(df)} samples from {csv_path}")
    print(f"📍 Nodes: {nodes}")
    print(f"⏱️  Duration: {df['time'].max():.2f} seconds")
    print(f"📈 Attributes to plot: {len(attributes)}")
    
    # Create figure with subplots - 7 rows x 2 columns for 14 attributes (last one empty)
    # Or use 4x4 = 16, leaving 2 empty
    n_rows = 4
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))
    fig.suptitle(f'IMU Data - All Attributes Over Time\n{Path(csv_path).name}', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes array for easier indexing
    axes_flat = axes.flatten()
    
    # Units for each attribute
    units = {
        'ax': 'g', 'ay': 'g', 'az': 'g',
        'gx': '°/s', 'gy': '°/s', 'gz': '°/s',
        'qw': '', 'qx': '', 'qy': '', 'qz': '',
        'roll': '°', 'pitch': '°', 'yaw': '°'
    }
    
    # Plot each attribute
    for idx, attr in enumerate(attributes):
        ax = axes_flat[idx]
        
        # Plot each node
        for node in nodes:
            node_df = df[df['node_name'] == node]
            color = node_colors.get(node, '#666')
            ax.plot(node_df['time'], node_df[attr], 
                   color=color, alpha=0.7, linewidth=0.8, label=node)
        
        # Formatting
        unit = units.get(attr, '')
        ylabel = f'{attr} ({unit})' if unit else attr
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(attr, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
        
        # Set x-label for bottom row
        if idx >= n_cols * (n_rows - 1):
            ax.set_xlabel('Time (seconds)', fontsize=10)
    
    # Hide unused subplots (if any)
    for idx in range(len(attributes), len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = csv_path.replace('.csv', '_all_attributes.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot to: {output_path}")
    
    return output_path


if __name__ == '__main__':
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Default path - user should provide the CSV file
        print("❌ Please provide CSV file path as argument")
        print("Usage: python plot_imu_csv.py <csv_file_path>")
        sys.exit(1)
    
    plot_imu_csv(csv_file)

