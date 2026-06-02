import numpy as np
from drawsvg import Drawing, Circle, Line, Text
import drawsvg as draw
import euclid   
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from hyperbolic import euclid, util
from hyperbolic.poincare import *
from hypnetperturb.embedding.distance import hyp_dist_poincare
import warnings 

"""
PLOTS
- 3D hyperboloid
- 2D Poincare disk
"""


def plot_hyperbolic_embedding_1(X, r, zmax, colors=None, sizes=None): #@Lili,modif@Ali

    if colors is None: # NOT WORKING
        colors = ['red']*len(X[:, 0])
    if sizes is None:# NOT WORKING
        sizes = [2]*len(X[:, 0])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(0, zmax)

    plt.rcParams['axes.linewidth'] = 2 
        
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color = colors, s = sizes, alpha=1)
    
    #ax.set_title('Hyperbolic embedding', fontsize = 15)
    
    fig.show()

    return 


def plot_hyperbolic_embedding(X, r, zmax, ax, colors=None, sizes=None, t=0): #@Lili,modif@Ali

    if colors is None: # NOT WORKING
        colors = ['red']*len(X[:, 0])
    if sizes is None:# NOT WORKING
        sizes = [2]*len(X[:, 0])

    #fig = plt.figure(figsize=(10, 10))
    #ax = fig.add_subplot(projection='3d')

    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(0, zmax)
    ax.set_title('time %.1f'%t, fontsize=20)
    ax.tick_params(axis='both', labelsize=20)

    plt.rcParams['axes.linewidth'] = 2 
        
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color = colors, s = sizes, alpha=1)
    
    #ax.set_title('Hyperbolic embedding', fontsize = 15)
    
"""
to make a single plot:
fig, axs = plt.subplots()
generate_plot(axs)
"""

# add_surface
def plot_hyperbolic_embedding_surf(X, c, colors=False, sizes=False): #@Lili,modif@Ali

    if colors == False:# NOT WORKING
        colors = ['red']*len(X[:, 0])
    if sizes == False:# NOT WORKING
        sizes = [1]*len(X[:, 0])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    x, y = np.meshgrid(np.linspace(np.min(X[:, 0]), np.max(X[:, 0]),20), np.linspace(np.min(X[:, 1]), np.max(X[:, 1]),20))
    #z = (x/a)**2+(y/b)**2+c
    #z = -(((x-b)/a)**2+((y-d)/c)**2)+1.0
    z= np.sqrt(x**2+y**2+c)
    
    ax.plot_surface(x, y, z, color='gray',
                alpha=0.2)

    plt.rcParams['axes.linewidth'] = 2 
        
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color = colors, s = sizes, alpha=1)
    
    ax.set_title('Hyperbolic embedding', fontsize = 15)
    
    plt.show()


# using plotly
def plot_hyperbolic_embedding_px(X, r, colors=None, sizes=None):

    # missing init of colors and sizes
    if colors is None:
        colors = ['red']*len(X[:, 0])
    if sizes is None:
        sizes = [2]*len(X[:, 0])

    fig = go.Figure(data=(go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,              # set color to an array/list of desired values
            colorscale='Spectral_r',   # choose a colorscale
            opacity=1
        ))))
    
    #df = {'x':X[:, 0], 'y':X[:, 1], 'z':X[:, 2], 'colors':colors, 'sizes':sizes}

    #fig = px.scatter_3d(df, x='x', y='y', z='z', color='colors', size='sizes')

    #color_continuous_scale=plotly.colors.diverging.Spectral[::-1]
    #paraboloid = X[:, 0]**2 + X[:, 1]**2-100
    #paraboloid = np.array([[x**2 + y**2-100 for x in np.arange(-10,11)] for y in np.arange(-10,11)])
    #fig.add_trace(go.Surface(z = paraboloid, surfacecolor = ["gray"]*len(X[:, 2]), opacity = .7, showscale = False))

    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
        #xaxis=dict(range=[-r-1, r+1]),
        #yaxis=dict(range=[-r-1, r+1]),
        )

    fig.show()

    print("Max z = ",np.max(X[:, 2]))

    return fig

# adding surface
def plot_hyperbolic_embedding_pxsurf(X, c, colors = None, sizes = None):

    l = 100
    x, y = np.meshgrid(np.linspace(np.min(X[:, 0]), np.max(X[:, 0]),l), np.linspace(np.min(X[:, 1]), np.max(X[:, 1]),l))
    z = np.sqrt(x**2+y**2+c)
    colors_z = np.zeros(shape=z.shape)

    colorscale = [[0, 'gray'], 
            [1, '#929591']]
    
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, opacity=0.3, surfacecolor=colors_z, colorscale=colorscale)])
    
    #paraboloid = np.array([[x**2 + y**2-100 for x in np.arange(-10,11)] for y in np.arange(-10,11)])
    #trace_b = go.Surface(z = paraboloid, surfacecolor = "rgb(245,245,245)", opacity = .7, showscale = False, name = "Trace B")

    fig.add_trace(go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,              # set color to an array/list of desired values
            colorscale='Spectral_r',   # choose a colorscale
            opacity=1
        )))

    fig.update_layout(
    autosize=False,
    width=800,
    height=800,
    )

    fig.show()


# Projection to Poincare Disk

def hyperboloid_to_Pdisk(X):
    """
    #projection of points lying on the hyperboloid on the Poincare disk model.

    X: Nx(n+1) ndarray, coordinates [x1,...,xn,t] of a set of N points 
    return Y: coords of a point on the disk [y1,...,yn]
    """
    N = len(X[:,0])
    n = len(X[0,:])-1
    Y = np.zeros((N,n))

    for i in range(N):
        Y[i,:]=X[i,:n]/(1+X[i,-1])

    return Y



epsilon = 2**-30
def near_zero(val):
    return -epsilon <= val <= epsilon
    
   
def hyperboloid_to_unit_disk(X):
    """
    Project hyperboloid embedding to a unit disk by
    simple radial normalization (not Poincaré geometry).
    """

    Y = X[:, :2].copy()

    rho = np.sqrt(Y[:,0]**2 + Y[:,1]**2)
    rho_max = rho.max()

    if rho_max > 0:
        Y /= rho_max

    return Y


def plot_Pdisk_embedding_n(Y, edgelist=None, edgelist_colors=None, node_colors=None, sizes=None):
    """
    Y: coordinates of points in array form [X,Y]
    edgelist: list of edges to plot [[i,j],...] in array/list form
    colors: array of colors to assign to nodes in order

    _n: node positions are normalised to be distributed on a disk of radius 1 
    """
    d = Drawing(2, 2, origin='center')

    pts = []
    R_Y = np.sqrt(Y[:,0]**2+Y[:,1]**2)
    R = np.max(R_Y)
    print("R = ",R)
    Y = Y/R
    d.draw(euclid.Circle(0, 0, 1), fill="aliceblue")

    if node_colors is None: 
        node_colors = len(Y[:,0])*['red']
    if sizes is None: 
        sizes = len(Y[:,0])*[1]

    for i in range(len(Y[:,0])):
        #pt = Point(Y[i,0],Y[i,1])
        pt = Point(Y[i,0],Y[i,1])
        d.draw(pt, hradius=sizes[i], fill=node_colors[i])
        #d.draw(draw.Text(str(i), 0.01, *pt, center=0.7))
        pts.append(pt)

    if edgelist is not None:
        if edgelist_colors is None:
            edgelist_colors=['gray']*len(edgelist)
        for j in range(len(edgelist[:,0])):
            i1,i2 = edgelist[j,0],edgelist[j,1]
            if near_zero(pts[i1][0]-pts[i2][0]) and near_zero(pts[i2][0]-pts[i2][0]):
                continue
            hl = Line.from_points(*pts[i1],*pts[i2], segment=True)
            d.draw(hl, hwidth=0.02, fill=edgelist_colors[j], opacity=0.8) #stroke_dasharray='5,2'
            
    d.set_render_size(w=500)

    return d

# def plot_Pdisk_embedding(Y, edgelist=None, edgelist_colors=None, node_colors=None, sizes=None):
#     """
#     Y: coordinates of points in array form [X,Y]
#     edgelist: list of edges to plot [[i,j],...] in array/list form
#     colors: array of colors to assign to nodes in order
     
#     """
#     pts = []
#     R_Y = np.sqrt(Y[:,0]**2+Y[:,1]**2)
#     R = np.max(R_Y)
#     f = 1000
#     print("R = ",R)
#     #Y = Y/f

#     d = Drawing(2*R, 2*R, origin='center')
#     d.draw(euclid.Circle(0, 0, R), fill="aliceblue")

#     if node_colors is None: 
#         node_colors = len(Y[:,0])*['red']
#     if sizes is None: 
#         sizes = len(Y[:,0])*[R/len(Y[:,0])]
#     print(sizes)

#     for i in range(len(Y[:,0])):
#         #pt = Point(Y[i,0],Y[i,1])
#         pt = Point(Y[i,0],Y[i,1])
#         d.draw(pt, hradius=sizes[i], fill=node_colors[i])
#         #d.draw(draw.Text(str(i), 0.01, *pt, center=0.7))
#         pts.append(pt)

#     if edgelist is not None:
#         if edgelist_colors is None:
#             edgelist_colors=['gray']*len(edgelist)
#         for j in range(len(edgelist[:,0])):
#             i1,i2 = edgelist[j,0],edgelist[j,1]
#             if near_zero(pts[i1][0]-pts[i2][0]) and near_zero(pts[i2][0]-pts[i2][0]):
#                 continue
#             hl = Line.from_points(*pts[i1],*pts[i2], segment=True)
#             d.draw(hl, hwidth=0.02, fill=edgelist_colors[j], opacity=0.8) #stroke_dasharray='5,2'
            
#     d.set_render_size(w=500)

#     return d
    
    
def plot_Pdisk_embedding(
    Y,
    edgelist=None,
    edgelist_colors=None,
    node_colors=None,
    sizes=None,
    node_labels=None,
    r_star=0.3,
    label_fontsize=8,
    label_dx=0.01,
    label_dy=0.01,
    label_color="black",
):
    """
    Y: coordinates of points in array form [X, Y]
    edgelist: list/array of edges to plot [[i, j], ...]
    node_colors: list of node colors
    sizes: list of node sizes
    node_labels: list of labels, same order as Y
    r_star: annotate only nodes with normalized radial coordinate r < r_star
    """
    pts = []
    R_Y = np.sqrt(Y[:, 0]**2 + Y[:, 1]**2)
    R = np.max(R_Y)

    print("R =", R)

    d = Drawing(2 * R, 2 * R, origin='center')
    d.draw(euclid.Circle(0, 0, R), fill="aliceblue")

    n = len(Y[:, 0])

    if node_colors is None:
        node_colors = ['red'] * n
    if sizes is None:
        sizes = [R / n] * n

    print(sizes)

    for i in range(n):
        x, y = Y[i, 0], Y[i, 1]
        pt = Point(x, y)
        d.draw(pt, hradius=sizes[i], fill=node_colors[i])
        pts.append(pt)

        if node_labels is not None:
            r_i = np.sqrt(x**2 + y**2) / R if R > 0 else 0.0
            if r_i < r_star and str(node_labels[i]) != "":
                d.append(
                    draw.Text(
                        str(node_labels[i]),
                        label_fontsize,
                        x + label_dx,
                        y + label_dy,
                        fill=label_color,
                    )
                )

    if edgelist is not None:
        if edgelist_colors is None:
            edgelist_colors = ['gray'] * len(edgelist)

        for j in range(len(edgelist[:, 0])):
            i1, i2 = edgelist[j, 0], edgelist[j, 1]

            if near_zero(pts[i1][0] - pts[i2][0]) and near_zero(pts[i1][1] - pts[i2][1]):
                continue

            hl = Line.from_points(*pts[i1], *pts[i2], segment=True)
            d.draw(hl, hwidth=0.02, fill=edgelist_colors[j], opacity=0.8)

    d.set_render_size(w=500)
    return d

def plot_Pdisk_embedding_with_focus_node(
    Y,
    focus_node_idx,
    focus_radius,
    labels=None,
    edgelist=None,
    edgelist_colors=None,
    node_colors=None,
    sizes=None,
    focus_node_color="black",
    focus_ball_color="none",
    focus_ball_edge_color="black",
    label_fontsize=0.03,
    edge_width=0.02,
    node_opacity=1.0,
    edge_opacity=0.8,
    render_size=500,
):
    """
    Plot a Poincare disk embedding and label nodes falling inside a
    hyperbolic ball centered on a focus node.

    Parameters
    ----------
    Y : ndarray, shape (N, 2)
        Node coordinates in the Poincare disk.

    focus_node_idx : int
        Index of the focus node, e.g. the regressor node.

    focus_radius : float
        Hyperbolic radius around the focus node.
        Nodes with d_H(node, focus_node) <= focus_radius are labeled.

    labels : list of str or None
        Node labels. If None, node indices are used.

    edgelist : ndarray, shape (E, 2), optional
        Edges as integer node-index pairs.

    edgelist_colors : list, optional
        Colors for edges.

    node_colors : list, optional
        Colors for nodes.

    sizes : list, optional
        Node sizes.

    focus_node_color : str
        Color assigned to the focus node.

    label_fontsize : float
        Text size for labeled nodes.

    Returns
    -------
    d : hyperbolic.Drawing
        Drawing object.
    """

    Y = np.asarray(Y, dtype=float)

    if Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError("Y must have shape (N, 2).")

    n_nodes = Y.shape[0]

    if focus_node_idx < 0 or focus_node_idx >= n_nodes:
        raise ValueError("focus_node_idx is outside the valid node range.")

    if labels is None:
        labels = [str(i) for i in range(n_nodes)]

    if node_colors is None:
        node_colors = ["red"] * n_nodes
    else:
        node_colors = list(node_colors)

    node_colors[focus_node_idx] = focus_node_color

    R_Y = np.sqrt(Y[:, 0] ** 2 + Y[:, 1] ** 2)
    R = np.max(R_Y)

    if R <= 0:
        R = 1.0

    if sizes is None:
        sizes = [R / n_nodes] * n_nodes

    print("R = ", R)
    print("sizes = ", sizes)

    d = Drawing(2 * R, 2 * R, origin="center")
    d.draw(euclid.Circle(0, 0, R), fill="aliceblue")

    pts = []

    for i in range(n_nodes):
        pt = Point(Y[i, 0], Y[i, 1])
        d.draw(
            pt,
            hradius=sizes[i],
            fill=node_colors[i],
            opacity=node_opacity,
        )
        pts.append(pt)

    if edgelist is not None:
        edgelist = np.asarray(edgelist, dtype=int)

        if edgelist_colors is None:
            edgelist_colors = ["gray"] * len(edgelist)

        for j in range(len(edgelist)):
            i1, i2 = edgelist[j, 0], edgelist[j, 1]

            if i1 == i2:
                continue

            if np.allclose(Y[i1], Y[i2]):
                continue

            hl = Line.from_points(*pts[i1], *pts[i2], segment=True)
            d.draw(
                hl,
                hwidth=edge_width,
                fill=edgelist_colors[j],
                opacity=edge_opacity,
            )

    focus_coord = Y[focus_node_idx]

    for i in range(n_nodes):
        dist_to_focus = hyp_dist_poincare(Y[i], focus_coord)

        if dist_to_focus <= focus_radius:
            d.draw(
                euclid.Text(
                    labels[i],
                    Y[i, 0],
                    Y[i, 1],
                    font_size=label_fontsize,
                ),
                fill="black",
            )

    d.set_render_size(w=render_size)

    return d