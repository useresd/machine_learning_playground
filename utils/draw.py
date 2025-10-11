import torch
import matplotlib.pyplot as plt

def visualize_tensors(*tensors, figsize=(12, 4), cmap='gray', show_values=True, title_prefix="Tensor"):
    """
    Visualize multiple tensors side by side as images with optional value annotations.
    
    Args:
        *tensors: Variable number of 2D tensors to visualize
        figsize: Figure size (width, height)
        cmap: Colormap for imshow
        show_values: Whether to show numerical values on each cell
        title_prefix: Prefix for subplot titles
    """
    num_tensors = len(tensors)
    fig, ax = plt.subplots(1, num_tensors, figsize=figsize)
    
    # Handle single tensor case
    if num_tensors == 1:
        ax = [ax]
    
    for i, tensor in enumerate(tensors):
        # Convert to numpy if it's a torch tensor
        if hasattr(tensor, 'detach'):
            display_tensor = tensor.detach().cpu()
        else:
            display_tensor = tensor
            
        ax[i].imshow(display_tensor, cmap=cmap)
        ax[i].set_title(f'{title_prefix} {i+1}: {tensor.shape[0]}x{tensor.shape[1]}')
        
        if show_values:
            for j, k in torch.cartesian_prod(torch.arange(tensor.shape[0]), torch.arange(tensor.shape[1])).tolist():
                val = tensor[j, k]
                color = 'white' if val.item() < 0.5 else 'black'
                ax[i].text(k, j, f'{val.item():.2f}', ha='center', va='center', color=color, fontsize=8)
        
        ax[i].set_xticks(range(tensor.shape[1]))
        ax[i].set_yticks(range(tensor.shape[0]))
    
    plt.tight_layout()
    plt.show()

def compare_tensor_operation(original, transformed, operation_name="Operation"):
    """
    Compare original tensor with its transformed version.
    
    Args:
        original: Original tensor
        transformed: Transformed tensor
        operation_name: Name of the operation for titles
    """
    visualize_tensors(original, transformed, 
                     title_prefix=f"{operation_name}",
                     figsize=(10, 4))
    
    print(f"Original shape: {original.shape}")
    print(f"Transformed shape: {transformed.shape}")