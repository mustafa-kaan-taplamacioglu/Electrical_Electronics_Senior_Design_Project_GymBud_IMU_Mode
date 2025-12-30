#!/usr/bin/env python3
"""
IMU Fusion Data Plotter
Plots all attributes from gymbud fusion CSV files over timeline
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def plot_imu_data(csv_path: str):
    """Load and plot all IMU data attributes from the fusion CSV."""
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Convert timestamp to relative time (seconds from start)
    df['time'] = df['timestamp'] - df['timestamp'].min()
    
    # Get unique nodes
    nodes = df['node_name'].unique()
    node_colors = {
        'left_wrist': '#3b82f6',   # Blue
        'right_wrist': '#a855f7',  # Purple
        'chest': '#f97316'         # Orange
    }
    
    print(f"📊 Loaded {len(df)} samples from {csv_path}")
    print(f"📍 Nodes: {nodes}")
    print(f"⏱️  Duration: {df['time'].max():.2f} seconds")
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    fig.suptitle(f'IMU Fusion Data Timeline\n{Path(csv_path).name}', fontsize=14, fontweight='bold')
    
    # Plot groups
    plot_groups = [
        ('Accelerometer', ['ax', 'ay', 'az'], 'g'),
        ('Gyroscope', ['gx', 'gy', 'gz'], '°/s'),
        ('Quaternion', ['qw', 'qx', 'qy', 'qz'], ''),
        ('Euler Angles', ['roll', 'pitch', 'yaw'], '°')
    ]
    
    for row_idx, (group_name, columns, unit) in enumerate(plot_groups):
        # Special handling for quaternion (4 values vs 3)
        if group_name == 'Quaternion':
            # Plot qw, qx, qy in first subplot
            ax = axes[row_idx, 0]
            for node in nodes:
                node_df = df[df['node_name'] == node]
                color = node_colors.get(node, '#666')
                ax.plot(node_df['time'], node_df['qw'], color=color, alpha=0.7, linewidth=0.5, label=f'{node} (qw)')
            ax.set_ylabel('qw')
            ax.set_title(f'{group_name} - qw')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='upper right')
            
            ax = axes[row_idx, 1]
            for node in nodes:
                node_df = df[df['node_name'] == node]
                color = node_colors.get(node, '#666')
                ax.plot(node_df['time'], node_df['qx'], color=color, alpha=0.7, linewidth=0.5, label=f'{node}')
                ax.plot(node_df['time'], node_df['qy'], color=color, alpha=0.4, linewidth=0.5, linestyle='--')
            ax.set_ylabel('qx, qy')
            ax.set_title(f'{group_name} - qx (solid), qy (dashed)')
            ax.grid(True, alpha=0.3)
            
            ax = axes[row_idx, 2]
            for node in nodes:
                node_df = df[df['node_name'] == node]
                color = node_colors.get(node, '#666')
                ax.plot(node_df['time'], node_df['qz'], color=color, alpha=0.7, linewidth=0.5, label=f'{node}')
            ax.set_ylabel('qz')
            ax.set_title(f'{group_name} - qz')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='upper right')
        else:
            for col_idx, col in enumerate(columns):
                ax = axes[row_idx, col_idx]
                
                for node in nodes:
                    node_df = df[df['node_name'] == node]
                    color = node_colors.get(node, '#666')
                    ax.plot(node_df['time'], node_df[col], color=color, alpha=0.7, linewidth=0.5, label=node)
                
                ax.set_ylabel(f'{col} ({unit})' if unit else col)
                ax.set_title(f'{group_name} - {col}')
                ax.grid(True, alpha=0.3)
                
                if col_idx == 2:  # Add legend to rightmost column
                    ax.legend(fontsize=8, loc='upper right')
    
    # Set x-label for bottom row
    for ax in axes[-1]:
        ax.set_xlabel('Time (seconds)')
    
    plt.tight_layout()
    
    # Save figure
    output_path = csv_path.replace('.csv', '_plots.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot to: {output_path}")
    
    # Also create per-node detailed plots
    fig2, axes2 = plt.subplots(3, 3, figsize=(16, 12))
    fig2.suptitle('Per-Node Detailed View\n(Roll, Pitch, Yaw)', fontsize=14, fontweight='bold')
    
    for node_idx, node in enumerate(nodes):
        node_df = df[df['node_name'] == node]
        color = node_colors.get(node, '#666')
        
        for col_idx, col in enumerate(['roll', 'pitch', 'yaw']):
            ax = axes2[node_idx, col_idx]
            ax.plot(node_df['time'], node_df[col], color=color, alpha=0.8, linewidth=0.8)
            ax.fill_between(node_df['time'], node_df[col], alpha=0.2, color=color)
            ax.set_title(f'{node} - {col}')
            ax.set_ylabel(f'{col} (°)')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = node_df[col].mean()
            std_val = node_df[col].std()
            ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(0.02, 0.98, f'μ={mean_val:.1f}°, σ={std_val:.1f}°', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    for ax in axes2[-1]:
        ax.set_xlabel('Time (seconds)')
    
    plt.tight_layout()
    
    output_path2 = csv_path.replace('.csv', '_euler_detailed.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✅ Saved detailed Euler plot to: {output_path2}")
    
    # Create acceleration magnitude plot
    fig3, axes3 = plt.subplots(2, 1, figsize=(14, 8))
    fig3.suptitle('Motion Analysis', fontsize=14, fontweight='bold')
    
    # Acceleration magnitude
    ax = axes3[0]
    for node in nodes:
        node_df = df[df['node_name'] == node].copy()
        node_df['accel_mag'] = np.sqrt(node_df['ax']**2 + node_df['ay']**2 + node_df['az']**2)
        color = node_colors.get(node, '#666')
        ax.plot(node_df['time'], node_df['accel_mag'], color=color, alpha=0.7, linewidth=0.5, label=node)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='1g reference')
    ax.set_ylabel('Acceleration Magnitude (g)')
    ax.set_title('Total Acceleration (should be ~1g when stationary)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Gyroscope magnitude (rotational velocity)
    ax = axes3[1]
    for node in nodes:
        node_df = df[df['node_name'] == node].copy()
        node_df['gyro_mag'] = np.sqrt(node_df['gx']**2 + node_df['gy']**2 + node_df['gz']**2)
        color = node_colors.get(node, '#666')
        ax.plot(node_df['time'], node_df['gyro_mag'], color=color, alpha=0.7, linewidth=0.5, label=node)
    ax.set_ylabel('Angular Velocity Magnitude (°/s)')
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Total Rotational Motion')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path3 = csv_path.replace('.csv', '_motion.png')
    plt.savefig(output_path3, dpi=150, bbox_inches='tight')
    print(f"✅ Saved motion analysis plot to: {output_path3}")
    
    # Print statistics summary
    print("\n📈 Statistics Summary:")
    print("=" * 60)
    for node in nodes:
        node_df = df[df['node_name'] == node]
        print(f"\n🏷️  {node.upper()} ({len(node_df)} samples)")
        print("-" * 40)
        for col in ['roll', 'pitch', 'yaw']:
            print(f"  {col:6s}: mean={node_df[col].mean():7.2f}°, std={node_df[col].std():6.2f}°, range=[{node_df[col].min():.1f}°, {node_df[col].max():.1f}°]")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Default to the most recent file
        csv_file = '/Users/kaantaplamacioglu/Desktop/github_repo_elec_491/Elec-491/CameraV2/logs/gymbud_fusion_20251222_145418.csv'
    
    plot_imu_data(csv_file)

