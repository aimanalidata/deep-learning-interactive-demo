import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier

# Generate data
X, y = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=0)

# Define plotting function
def plot_decision_boundary(hidden_layer_size):
    clf = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,),
                        activation='relu', max_iter=3000, random_state=1)
    clf.fit(X, y)

    # Create grid for background color
    x_vals = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 100)
    y_vals = np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 100)
    X_plane, Y_plane = np.meshgrid(x_vals, y_vals)
    grid_points = np.column_stack((X_plane.ravel(), Y_plane.ravel()))
    Z = clf.predict(grid_points).reshape(X_plane.shape)

    ax.clear()
    ax.contourf(X_plane, Y_plane, Z, cmap=plt.cm.RdYlGn, alpha=0.6)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='red', edgecolors='k', label='Class 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='green', edgecolors='k', label='Class 1')
    ax.set_title(f"Decision Boundary (Hidden Layer Size = {hidden_layer_size})")
    ax.legend()

# Create initial plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
plot_decision_boundary(1)

ax_slider = plt.axes([0.25, 0.05, 0.5, 0.04])
slider = Slider(ax_slider, 'Hidden Layer Size', 1, 10, valinit=1, valstep=1)

def update(val):
    hidden_layer_size = int(slider.val)
    plot_decision_boundary(hidden_layer_size)
    plt.draw()

slider.on_changed(update)
plt.show()

