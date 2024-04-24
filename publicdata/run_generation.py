
from utils_2d import *
from deformation import *
import pandas as pd
from tqdm import trange
if __name__ == "__main__":
    np.random.seed(980)
    '''
    train
    '''
    # define the dataset
    NUM_OF_PATIENTS = 1000

    #
    path_starman_vtk = "/home/jyn/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/2d/starmen_for_simulation/data_ground_truth/ForSimulation__Template__GroundTruth.vtk"
    path_control_points = "/home/jyn/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/2d/starmen_for_simulation/data_ground_truth/ForSimulation__ControlPoints__GoundTruth.txt"
    # "/Users/jyn/jyn/research/projects/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/2d/starmen/data/ForInitialization__ControlPoints__FromLongitudinalAtlas.txt"

    pv_template = pv.read(path_starman_vtk)
    arr_control_pts = np.loadtxt(path_control_points)

    arr_resampled_pts = resample_polygon(pv_template.points, n_points=1000)
    arr_resampled_lines = np.concatenate((np.ones(len(arr_resampled_pts)-1)[:, None]*2,
                                          np.arange(len(arr_resampled_pts)-1)[:, None],
                                          np.arange(len(arr_resampled_pts)-1)[:, None]+1), axis=-1)
    arr_resampled_lines = list(arr_resampled_lines)
    arr_resampled_lines.append(np.array([2, 999, 0]))
    arr_resampled_lines = np.array(arr_resampled_lines).ravel()


    pv_template.points = np.concatenate((arr_resampled_pts, np.zeros_like(arr_resampled_pts[:, [0]])), axis=-1)
    pv_template.lines = arr_resampled_lines.astype('int')
    pv_template.save("/home/jyn/NAISR/examples/starman/template.vtk")
    np.save("/home/jyn/NAISR/examples/starman/control_template.npy", arr_control_pts)


    rootpath = '/playpen-raid/jyn/NAISR/NAISR/examples/starman/train/' #"/home/jyn/NAISR/publicdata/starman"
    if not os.path.exists(rootpath):
        os.mkdir(rootpath)

    list_data = []
    for ith_subject in trange(NUM_OF_PATIENTS):
        # generate longitudinal templates
        num_of_observations = np.random.randint(1, 10)

        #
        PID ='{0:04}'.format(ith_subject)
        arr_deformed_contour, arr_deformed_controls = deform_polygon_randomly(pv_template.points[:, [0, 1]], arr_control_pts[:, [0, 1]])
        save_personalized_template(pv_template, arr_deformed_contour, rootpath, PID)
        save_personalized_template_contour_pts( arr_deformed_contour, rootpath, PID)
        save_personalized_template_control_pts(arr_deformed_controls, rootpath, PID)

        pv_deformed_contour = pv_template.copy()
        arr_deformed_contour = np.concatenate((arr_deformed_contour, np.zeros_like(arr_deformed_contour[:, [0]])), axis=-1)
        #arr_deformed_controls = np.concatenate((arr_deformed_controls, np.zeros_like(arr_deformed_controls[:, [0]])), axis=-1)

        pv_deformed_contour.points = arr_deformed_contour
        list_data_of_current_subj = generate_dataset_for_one_subject(PID,
                                         pv_deformed_contour,
                                         arr_deformed_controls,
                                         num_of_shapes=num_of_observations,
                                         num_of_covariates=2,
                                         rootoath=rootpath)
        list_data+=list_data_of_current_subj

    # save the data set info to a csv
    savepath_dataset = "/home/jyn/NAISR/examples/starman/2dshape_train.csv"
    pd.DataFrame.from_records(list_data).to_csv(savepath_dataset)


    '''
    test
    '''

    # define the dataset
    NUM_OF_PATIENTS = 1000

    #
    path_starman_vtk = "/home/jyn/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/2d/starmen_for_simulation/data_ground_truth/ForSimulation__Template__GroundTruth.vtk"
    path_control_points = "/home/jyn/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/2d/starmen_for_simulation/data_ground_truth/ForSimulation__ControlPoints__GoundTruth.txt"
    # "/Users/jyn/jyn/research/projects/NAISR/publicdata/deformetrica/examples/longitudinal_atlas/landmark/2d/starmen/data/ForInitialization__ControlPoints__FromLongitudinalAtlas.txt"

    pv_template = pv.read(path_starman_vtk)
    arr_control_pts = np.loadtxt(path_control_points)

    arr_resampled_pts = resample_polygon(pv_template.points, n_points=1000)
    arr_resampled_lines = np.concatenate((np.ones(len(arr_resampled_pts)-1)[:, None]*2,
                                          np.arange(len(arr_resampled_pts)-1)[:, None],
                                          np.arange(len(arr_resampled_pts)-1)[:, None]+1), axis=-1)
    arr_resampled_lines = list(arr_resampled_lines)
    arr_resampled_lines.append(np.array([2, 999, 0]))
    arr_resampled_lines = np.array(arr_resampled_lines).ravel()


    pv_template.points = np.concatenate((arr_resampled_pts, np.zeros_like(arr_resampled_pts[:, [0]])), axis=-1)
    pv_template.lines = arr_resampled_lines.astype('int')
    pv_template.save("/home/jyn/NAISR/examples/starman/template.vtk")
    np.save("/home/jyn/NAISR/examples/starman/control_template.npy", arr_control_pts)

    rootpath = '/playpen-raid/jyn/NAISR/NAISR/examples/starman/test/' #"/home/jyn/NAISR/publicdata/starman"
    if not os.path.exists(rootpath):
        os.mkdir(rootpath)

    list_data = []
    for ith_subject in trange(NUM_OF_PATIENTS):
        # generate longitudinal templates
        num_of_observations = np.random.randint(1, 10)

        #
        PID ='{0:04}'.format(ith_subject)
        arr_deformed_contour, arr_deformed_controls = deform_polygon_randomly(pv_template.points[:, [0, 1]], arr_control_pts[:, [0, 1]])
        save_personalized_template(pv_template, arr_deformed_contour, rootpath, PID)
        save_personalized_template_contour_pts(arr_deformed_contour, rootpath, PID)
        save_personalized_template_control_pts(arr_deformed_controls, rootpath, PID)


        pv_deformed_contour = pv_template.copy()
        arr_deformed_contour = np.concatenate((arr_deformed_contour, np.zeros_like(arr_deformed_contour[:, [0]])), axis=-1)
        #arr_deformed_controls = np.concatenate((arr_deformed_controls, np.zeros_like(arr_deformed_controls[:, [0]])), axis=-1)

        pv_deformed_contour.points = arr_deformed_contour
        list_data_of_current_subj = generate_dataset_for_one_subject(PID,
                                         pv_deformed_contour,
                                         arr_deformed_controls,
                                         num_of_shapes=num_of_observations,
                                         num_of_covariates=4,
                                         rootoath=rootpath)
        list_data+=list_data_of_current_subj

    # save the data set info to a csv
    savepath_dataset = "/home/jyn/NAISR/examples/starman/2dshape_test.csv"
    pd.DataFrame.from_records(list_data).to_csv(savepath_dataset)