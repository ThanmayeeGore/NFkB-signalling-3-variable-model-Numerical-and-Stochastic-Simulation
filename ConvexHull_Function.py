import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


# define a function to plot the coordinates along with the convex hull

def State_Boundary(x, y):    # x and y are the coordinates

    
    points = np.vstack((x, y)).T    # combine x and y to vertical stack
    
    
    hull = ConvexHull(points)       # make convex hull
    
    # plot the coordinates
    plt.plot(points[:, 0], points[:, 1], '-1g', ms = 20, alpha = 0.3)
    
    # plot the convex hull
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-', lw = 2)
    
    # fill the polygon with lightyellow color
    plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], 'lightyellow', alpha = 0.1)
    
