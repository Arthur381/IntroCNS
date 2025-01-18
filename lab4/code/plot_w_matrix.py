import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_W_heatmap(W, output_dir, vmin=0, vmax=4, figsize=(10, 8), cmap='coolwarm', dpi=300):
    """
    Plot heatmap of weight matrix W with customizable parameters
    
    Args:
        W (np.array): Weight matrix
        output_dir (str): Directory to save output
        vmin (float): Minimum value for color scaling
        vmax (float): Maximum value for color scaling
        figsize (tuple): Figure size in inches (width, height)
        cmap (str): Colormap name
        dpi (int): DPI for saved figure
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=figsize)
    plt.imshow(W, cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Weight Value')
    plt.title('Weight Matrix (W) Heatmap')
    plt.xlabel('Neuron Index')
    plt.ylabel('Neuron Index')
    
    save_name = os.path.join(output_dir, f'W_heatmap_vmin{vmin}_vmax{vmax}_cmap{cmap}.png')
    plt.savefig(save_name, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap as: {save_name}")

def plot_W_row(W, row_idx, output_dir, figsize=(10, 6), dpi=300):
    """
    Plot a single row from the weight matrix as a line plot
    
    Args:
        W (np.array): Weight matrix
        row_idx (int): Index of row to plot
        output_dir (str): Directory to save output
        figsize (tuple): Figure size in inches
        dpi (int): DPI for saved figure
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=figsize)
    neuron_indices = np.arange(W.shape[1])
    plt.plot(neuron_indices, W[row_idx], 'b-', linewidth=2)
    plt.title(f'Weight Matrix Row {row_idx}')
    plt.xlabel('Target Neuron Index')
    plt.ylabel('Weight Value')
    plt.grid(True)
    
    save_name = os.path.join(output_dir, f'W_row_{row_idx}.png')
    plt.savefig(save_name, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved row plot as: {save_name}")
    print(f"Row statistics:")
    print(f"Min value: {W[row_idx].min():.4f}")
    print(f"Max value: {W[row_idx].max():.4f}")
    print(f"Mean value: {W[row_idx].mean():.4f}")

def print_matrix_stats(W):
    """Print basic statistics about the weight matrix"""
    print(f"Matrix shape: {W.shape}")
    print(f"Matrix statistics:")
    print(f"Min value: {W.min():.4f}")
    print(f"Max value: {W.max():.4f}")
    print(f"Mean value: {W.mean():.4f}")
    print(f"Median value: {np.median(W):.4f}")

def main():
    parser = argparse.ArgumentParser(description='W Matrix Visualization Tool')
    
    # Operation mode
    parser.add_argument('mode', choices=['heatmap', 'row'],
                      help='Operation mode: create heatmap or plot single row')
    
    # Common parameters
    parser.add_argument('W_path', type=str, help='Path to W matrix .npy file')
    parser.add_argument('--output_dir', type=str, default='W_output',
                      help='Directory to save outputs')
    
    # Parameters for heatmap
    parser.add_argument('--vmin', type=float, default=0,
                      help='Minimum value for color scaling')
    parser.add_argument('--vmax', type=float, default=4,
                      help='Maximum value for color scaling')
    parser.add_argument('--figsize', type=int, nargs=2, default=[10, 8],
                      help='Figure size (width height)')
    parser.add_argument('--cmap', type=str, default='coolwarm',
                      help='Colormap name')
    parser.add_argument('--dpi', type=int, default=300,
                      help='DPI for saved figure')
    
    # Parameters for row plot
    parser.add_argument('--row_idx', type=int,
                      help='Index of row to plot (required for row mode)')
    
    args = parser.parse_args()
    
    # Load W matrix
    W = np.load(args.W_path)
    print_matrix_stats(W)
    
    if args.mode == 'heatmap':
        plot_W_heatmap(
            W,
            args.output_dir,
            vmin=args.vmin,
            vmax=args.vmax,
            figsize=tuple(args.figsize),
            cmap=args.cmap,
            dpi=args.dpi
        )
    
    elif args.mode == 'row':
        if args.row_idx is None:
            raise ValueError("row_idx is required for row mode")
        if args.row_idx >= W.shape[0]:
            raise ValueError(f"row_idx {args.row_idx} is out of range for matrix with shape {W.shape}")
        plot_W_row(
            W,
            args.row_idx,
            args.output_dir,
            figsize=tuple(args.figsize),
            dpi=args.dpi
        )

if __name__ == "__main__":
    main()

#查看单行
#python plot_w_matrix.py row W_matrix_l128_n64_sigma2.npy --row_idx 100 --output_dir w_visualizations
