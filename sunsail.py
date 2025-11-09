import numpy as np
import pyvista as pv

data = np.load('Finite_Element_Mesh.npz')

coordinates = data['coordinates']
membrane = data['membrane']
edge = data['edge']
support = data['support']
truss = data['truss']

points = coordinates[:, 1:4]
triangles = membrane[:, 1:4]
faces = np.hstack([np.full((triangles.shape[0], 1), 3), triangles]).flatten()

membrane_mesh = pv.PolyData(points, faces=faces)

edge_coo = edge[:, 1:3]
line_cells_edge = np.hstack([np.full((edge_coo.shape[0], 1), 2), edge_coo]).flatten()

support_coo = support[:, 1:3]
line_cells_support = np.hstack([np.full((support_coo.shape[0], 1), 2), support_coo]).flatten()

truss_coo = truss[:, 1:3]
line_cells_truss = np.hstack([np.full((truss_coo.shape[0], 1), 2), truss_coo]).flatten()

edge_mesh = pv.PolyData(points, lines=line_cells_edge)
support_mesh = pv.PolyData(points, lines=line_cells_support)
truss_mesh = pv.PolyData(points, lines=line_cells_truss)

pl = pv.Plotter()
pl.add_mesh(membrane_mesh)
pl.add_mesh(edge_mesh, line_width=2, color='darkred', label='Edges')
pl.add_mesh(support_mesh, line_width=2, color='green', label='Supports')
pl.add_mesh(truss_mesh, line_width=2, color='orange', label='Trusses')
pl.add_legend()
pl.show()