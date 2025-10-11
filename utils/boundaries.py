import numpy as np
import torch
import matplotlib.pylab as plt

def plot_decision_boundary(model, X, y, device='cpu', cmap='coolwarm'):
    model.eval()

    # Create a grid over the input space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Stack grid coordinates
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)

    # Get model predictions
    with torch.no_grad():
        preds = model(grid_tensor)
        if preds.shape[1] > 1:
            preds = torch.argmax(preds, dim=1)
        else:
            preds = (torch.sigmoid(preds) > 0.5).long()
    Z = preds.cpu().numpy().reshape(xx.shape)

    plt.clf()  # Clear the previous plot
    # Plot the contour and training points
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary with Scatter")

    plt.show()