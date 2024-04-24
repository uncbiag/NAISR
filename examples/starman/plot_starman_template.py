import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np
filepath_contour = "/home/jyn/NAISR/examples/starman/template.vtk"
filepath_control  = "/home/jyn/NAISR/examples/starman/control_template.npy"
a = pv.read(filepath_contour)
anchor = np.load(filepath_control)
list_contour = np.array(a.points).tolist() + np.array(a.points)[[0]].tolist()
arr_contour = np.array(list_contour)
plt.plot(arr_contour[:, 0], arr_contour[:, 1], color='black')
plt.scatter(anchor[:, 0], anchor[:, 1], color='magenta')
plt.grid(False)
plt.savefig('/home/jyn/NAISR/data_generation/starmantemplate.png', transparent=True)
#plt.show()
plt.close()




filepath_case_template_contour = "/playpen-raid/jyn/NAISR/NAISR/examples/starman/train/template/0018_template.vtk"
filepath_case_template_control= "/playpen-raid/jyn/NAISR/NAISR/examples/starman/train/template/0018_template_control_pts.npy"
a1 = pv.read(filepath_case_template_contour)
list_contour1 = np.array(a1.points).tolist() + np.array(a1.points)[[0]].tolist()
arr_contour1 = np.array(list_contour1)
anchor = np.load(filepath_case_template_control)
plt.scatter(anchor[:, 0], anchor[:, 1], color='magenta')
plt.plot(arr_contour1[:, 0], arr_contour1[:, 1], color='black')
plt.grid('on')
plt.savefig('/home/jyn/NAISR/data_generation/starman_case_template.png', transparent=True)
plt.close()



filepath_case_specific_contour = "/playpen-raid/jyn/NAISR/NAISR/examples/starman/train/2dshape/0018_3_on.vtk"
a2 = pv.read(filepath_case_specific_contour )
list_contour2 = np.array(a2.points).tolist() + np.array(a2.points)[[0]].tolist()
arr_contour2 = np.array(list_contour2)
plt.plot(arr_contour2[:, 0], arr_contour2[:, 1], color='red')
plt.plot(arr_contour1[:, 0], arr_contour1[:, 1], color='grey', linestyle="-.")
plt.scatter(anchor[:, 0], anchor[:, 1], color='magenta')
plt.grid('on')
plt.savefig('/home/jyn/NAISR/data_generation/starman_case_specific.png', transparent=True)
