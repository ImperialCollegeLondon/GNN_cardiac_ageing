import pyvista as pv
import os
home = os.path.expanduser('~')

pv.rcParams['transparent_background'] = True

polydata = pv.read(home+'/cardiac/UKBB_40616/4D_Segmented_2.0_completed/3392039/vtks/F_LVendo_ED.vtk')
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(polydata, line_width=1, show_edges=False,
point_size=10, color='#ce5a3f'
)
polydata = pv.read(home+'/cardiac/minacio/LVendo_ED_template.vtk')
polydata.points = polydata.points*1.05
plotter.add_mesh(polydata, line_width=1, show_edges=False,
point_size=10, color='#3FB3CE', opacity=0.3,
)
plotter.show(screenshot=home+f'/cardiac/Ageing/gnn_paper_figures/diagram_ed_with_atlas.png', cpos=[1, -1, 0.9])

for decimate in [True, False]:
    for with_edges in [True, False]:
        if not decimate and with_edges:
            continue
        polydata = pv.read(home+'/cardiac/UKBB_40616/4D_Segmented_2.0_completed/3392039/vtks/F_LVendo_ED.vtk')
        if decimate:
            polydata = polydata.decimate(0.99)
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(polydata, line_width=3, show_edges=with_edges,
        point_size=10, color='#ce5a3f', edge_color='#A8BC63'
        )
        plotter.show(screenshot=home+f'/cardiac/Ageing/gnn_paper_figures/diagram_ed{"_decimated" if decimate else ""}{"_with_edges" if with_edges else ""}.png', cpos=[1, -1, 0.9])

        polydata = pv.read(home+'/cardiac/UKBB_40616/4D_Segmented_2.0_completed/3392039/vtks/F_LVendo_ES.vtk')
        if decimate:
            polydata = polydata.decimate(0.99)
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(polydata, line_width=3, show_edges=with_edges,
        point_size=10, color='#ce5a3f', edge_color='#A8BC63'
        )
        plotter.show(screenshot=home+f'/cardiac/Ageing/gnn_paper_figures/diagram_es{"_decimated" if decimate else ""}{"_with_edges" if with_edges else ""}.png', cpos=[1, -1, 0.9])

        polydata = pv.read(home+'/cardiac/UKBB_40616/4D_Segmented_2.0_completed/3392039/motion/LV_endo_fr20.vtk')
        if decimate:
            polydata = polydata.decimate(0.99)
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(polydata, line_width=3, show_edges=with_edges,
        point_size=10, color='#ce5a3f', edge_color='#A8BC63'
        )
        plotter.show(screenshot=home+f'/cardiac/Ageing/gnn_paper_figures/diagram_20{"_decimated" if decimate else ""}{"_with_edges" if with_edges else ""}.png', cpos=[1, -1, 0.9])
