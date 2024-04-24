import pandas as pd
import os
'''
train
'''

df_data = pd.read_csv("/home/jyn/NAISR/examples/starman/2dshape_train.csv")
root = "/playpen-raid/jyn/NAISR/NAISR/examples/starman/train/template/"

list_control = []
list_contour = []


for i in list(df_data['PID']):
    current_PID = '{0:04}'.format(i)
    current_control = os.path.join(root, current_PID + '_template_control_pts.npy')
    current_contour = os.path.join(root, current_PID + '_template.vtk')
    list_control.append(current_control)
    list_contour.append(current_contour)


df_data['temp_control'] = list_control
df_data['temp_contour'] = list_contour

savepath_train = "/home/jyn/NAISR/examples/starman/2dshape_train_with_temp.csv"
df_data.to_csv(savepath_train)

'''
test
'''


df_data = pd.read_csv("/home/jyn/NAISR/examples/starman/2dshape_test.csv")
root = "/playpen-raid/jyn/NAISR/NAISR/examples/starman/test/template/"

list_control = []
list_contour = []


for i in list(df_data['PID']):
    current_PID = '{0:04}'.format(i)
    current_control = os.path.join(root, current_PID + '_template_control_pts.npy')
    current_contour = os.path.join(root, current_PID + '_template.vtk')
    list_control.append(current_control)
    list_contour.append(current_contour)


df_data['temp_control'] = list_control
df_data['temp_contour'] = list_contour

savepath_test = "/home/jyn/NAISR/examples/starman/2dshape_test_with_temp.csv"
df_data.to_csv(savepath_test)

