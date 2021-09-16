from Filters import MMCF, MINACE, OT_MACH, OTSDF, MOSSE, ASEF, UOTSDF, MACE, MACH, UMACE, MVSDF, NNCF
import numpy as np
from Correlation.Correlation_utils import WorkingWithJointTransformCorrelator
from collections import defaultdict


def joint_transform_working():
    path_to_save = 'D:/MIFI/SCIENTIFIC WORK/JOINT TRANSFORM/DATA FOR CORRELATOR/Correlation data/yale'
    # path_to_save = 'D:/MIFI/SCIENTIFIC WORK/JOINT TRANSFORM/DATA FOR CORRELATOR/Hologram recovery'
    # path_to_save = 'D:/MIFI/SCIENTIFIC WORK/JOINT TRANSFORM/DATA FOR CORRELATOR/Joint spectrum'
    path_to_save = 'D:/MIFI/SCIENTIFIC WORK/JOINT TRANSFORM/DATA FROM CORRELATOR/Correlation peaks/yale_mmcf_pos'
    filter_path = 'Filter/yale_face_ASEF.npy'
    path_to_data = 'D:/MIFI/SCIENTIFIC WORK/JOINT TRANSFORM/DATA FROM CORRELATOR/Joint spectrum'
    c = WorkingWithJointTransformCorrelator(path_to_save=path_to_save, path_to_filter=filter_path,
                                            path_to_data=None, modulator_size=(1080, 1920))
    # c.prepare_data(inverse_image=False, filter_plane='freq')
    c.peaks_from_joint_spectrum(image_shape=(100, 100), build_3d_plot=True, labels=np.zeros(3), size_param=0.3)
    # c.joint_spectrum_processing(sample_level=8)
    # c.cf_hologram_recovery(sample_level=8, filter_plane='spatial')


def road_signs(corr_type, gt_label=1, plot_correlations_in_2d=False, plot_correlations_in_3d=False, show=False,
               make_holo=False):
    rs_train_images_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/train'
    rs_test_images_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/test'
    rs_train_labels_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_train.csv'
    rs_test_labels_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_test.csv'

    metric_names = ['accuracy', 'f1', 'precision', 'recall']
    num_of_images = 40
    f_metrics = {}
    inverse = False
    target_size = (500, 500)

    otsdf = OTSDF(images=rs_train_images_path, labels=rs_train_labels_path, gt_label=gt_label,
                  inverse=inverse, num_of_images=100, lambda_=0.5)
    mmcf = MMCF(images=rs_train_images_path, labels=rs_train_labels_path, lambda_=0.8, c=1000, inverse=inverse,
                num_of_images=400, gt_label=gt_label)
    mosse = MOSSE(images=rs_train_images_path, labels=rs_train_labels_path, gt_label=gt_label,
                  inverse=inverse, num_of_images=4000, fwhm=4)
    filters = [otsdf, mmcf, mosse]

    for cf in filters:
        cf.synthesis(target_size=target_size)
        cf.save_matrix(path='Filter/yale_face_' + cf.name)
        if cf.name not in f_metrics:
            f_metrics[cf.name] = np.zeros(len(metric_names))
        if show:
            cf.show()
        if make_holo:
            cf.make_amplitude_hologram(
                modulator_size=(1920, 1080), sample_level=8,
                path_to_save='D:/MIFI/SCIENTIFIC WORK/JOINT TRANSFORM/DATA FOR CORRELATOR/Hologram recovery')
    Filters = [MACE, MVSDF, OTSDF, UMACE, MACH, UOTSDF, ASEF, MOSSE, MMCF]
    Filters = [OTSDF, MOSSE, MMCF]

    for F in Filters:
        cf = F(images=rs_test_images_path, labels=rs_test_labels_path,
               num_of_images=16, gt_label=gt_label,
               batch_size=200, metric_names=metric_names,
               peak_classifier='pcnn', filter_plane='freq', inverse=inverse)
        cf.test_classification_quality(path_to_filter='Filter/yale_face_' + cf.name + '.npy', corr_type=corr_type,
                                       plot_correlations_in_3d=plot_correlations_in_3d, get=True,
                                       plot_conf_matrix=False, plot_correlations_in_2d=plot_correlations_in_2d,
                                       path_to_model='019_epoch.ONNX', target_size=target_size, use_wf=False)
        f_metrics[cf.name] = f_metrics[cf.name] + np.array(cf.m_values)


    # for cf, values in f_metrics.items():
    #     print('{}:'.format(cf))
    #     for i, metric in enumerate(metric_names):
    #         print('{}: {}%'.format(metric, round(100 * values[i] / range_gt_label, 1)))


def yale_face(corr_type, range_gt_label=1, plot_correlations_in_2d=False, plot_correlations_in_3d=False, show=False,
              make_holo=False):
    face = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/CroppedYale'
    metric_names = ['accuracy', 'f1', 'precision', 'recall']
    num_of_images = 70
    f_metrics = {}
    inverse = False
    target_size = (500, 500)

    for gt_label in range(range_gt_label):
        mace = MACE(images=face, labels=None, gt_label=gt_label, inverse=inverse, num_of_images=num_of_images)
        mvsdf = MVSDF(images=face, labels=None, gt_label=gt_label, inverse=inverse, num_of_images=num_of_images)
        otsdf = OTSDF(images=face, labels=None, gt_label=gt_label, inverse=inverse, num_of_images=num_of_images,
                      lambda_=0.5)
        umace = UMACE(images=face, labels=None, gt_label=gt_label, inverse=inverse, num_of_images=num_of_images)
        mach = MACH(images=face, labels=None, gt_label=gt_label, inverse=inverse, num_of_images=num_of_images)
        uotsdf = UOTSDF(images=face, labels=None, gt_label=gt_label, inverse=inverse, num_of_images=num_of_images,
                        lambda_=0.5)
        asef = ASEF(images=face, labels=None, gt_label=gt_label, inverse=inverse, num_of_images=num_of_images, fwhm=4)
        mosse = MOSSE(images=face, labels=None, gt_label=gt_label, inverse=inverse, num_of_images=num_of_images, fwhm=4)
        mmcf = MMCF(images=face, labels=None, gt_label=gt_label, inverse=inverse, num_of_images=2*num_of_images,
                    lambda_=0.8, c=1000)

        filters = [mace, mvsdf, otsdf, umace, mach, uotsdf, asef, mosse, mmcf]
        # filters = [mace, mvsdf, otsdf, umace, mach, uotsdf, mmcf]

        for cf in filters:
            cf.synthesis(target_size=target_size)
            cf.save_matrix(path='Filter/yale_face_' + cf.name)
            if cf.name not in f_metrics:
                f_metrics[cf.name] = np.zeros(len(metric_names))
            if show:
                cf.show()
            if make_holo:
                cf.make_amplitude_hologram(
                    modulator_size=(1920, 1080), sample_level=8, multi=True,
                    path_to_save='D:/MIFI/SCIENTIFIC WORK/JOINT TRANSFORM/DATA FOR CORRELATOR/Hologram recovery')
        Filters = [MACE, MVSDF, OTSDF, UMACE, MACH, UOTSDF, ASEF, MOSSE, MMCF]
        Filters = [OTSDF, MOSSE, MMCF]

        for F in Filters:
            cf = F(images=face, labels=None, num_of_images=16, gt_label=gt_label,
                   batch_size=200, metric_names=metric_names,
                   peak_classifier='pcnn', filter_plane='freq', inverse=inverse)
            cf.test_classification_quality(path_to_filter='Filter/yale_face_' + cf.name + '.npy', corr_type=corr_type,
                                           plot_correlations_in_3d=plot_correlations_in_3d, get=True,
                                           plot_conf_matrix=False, plot_correlations_in_2d=plot_correlations_in_2d,
                                           path_to_model='019_epoch.ONNX', target_size=target_size, use_wf=False)
            f_metrics[cf.name] = f_metrics[cf.name] + np.array(cf.m_values)

    for cf, values in f_metrics.items():
        print('{}:'.format(cf))
        for i, metric in enumerate(metric_names):
            print('{}: {}%'.format(metric, round(100 * values[i] / range_gt_label, 1)))


def mmcf_usage():
    rps_train_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Rock-Paper-Scissors/train'
    rps_test_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Rock-Paper-Scissors/train'

    planes_train_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Planes/train'
    planes_test_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Planes/test'

    rs_train_images_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/train'
    rs_test_images_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/test'
    rs_train_labels_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_train.csv'
    rs_test_labels_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_test.csv'

    tanks = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Tanks/Train'
    tanks_test = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Tanks/Test'
    face = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/CroppedYale'

    mmcf = MMCF(
        images=rs_train_images_path,
        labels=rs_train_labels_path,
        lambda_=0.8,
        c=1000,
        inverse=False,
        num_of_images=2000,
        gt_label=42,
        metric_names=None
               )

    mmcf.synthesis(target_size=(48, 48))
    mmcf.show()
    mmcf.save_matrix(path='Filter/road_signs_mmcf_1')
    mmcf = MMCF(images=rs_test_images_path,
                labels=rs_test_labels_path,
                num_of_images=3000,
                gt_label=42,
                batch_size=200,
                metric_names=['accuracy', 'precision', 'recall'],
                peak_classifier='peak_position',
                filter_plane='freq')
    mmcf.test_classification_quality(path_to_filter='Filter/road_signs_mmcf_1.npy',
                                     plot_correlations_in_3d=True,
                                     plot_conf_matrix=False,
                                     plot_correlations_in_2d=True,
                                     path_to_model=None,
                                     target_size=(48, 48))


def minace_usage():
    rps_train_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Rock-Paper-Scissors/train'
    rps_test_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Rock-Paper-Scissors/train'

    planes_train_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Planes/train'
    planes_test_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Planes/test'

    rs_train_images_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/train'
    rs_test_images_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/test'
    rs_train_labels_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_train.csv'
    rs_test_labels_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_test.csv'

    tanks = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Tanks/Train'
    tanks_test = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Tanks/Test'

    minace = MINACE(images=tanks_test,
                    labels=None,
                    gt_label=0,
                    metric_names=['accuracy', 'precision', 'recall'],
                    inverse=False,
                    num_of_images=1000)

    minace.synthesis(target_size=(256, 256))
    minace.show()
    minace.save_matrix(path='Filter/road_signs_minace_1')
    minace = MINACE(images=tanks_test,
                    labels=None,
                    num_of_images=1000,
                    gt_label=0,
                    batch_size=200,
                    metric_names=['accuracy', 'precision', 'recall'],
                    peak_classifier='peak_position',
                    filter_plane='freq')
    minace.test_classification_quality(path_to_filter='Filter/road_signs_minace_1.npy',
                                       plot_correlations_in_3d=True,
                                       plot_conf_matrix=False,
                                       plot_correlations_in_2d=True,
                                       path_to_model=None,
                                       target_size=(256, 256))


def otsdf_usage():
    rps_train_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Rock-Paper-Scissors/train'
    rps_test_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Rock-Paper-Scissors/train'

    planes_train_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Planes/train'
    planes_test_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Planes/test'
    tanks = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Tanks/Train'
    tanks_test = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Tanks/Test'

    rs_train_images_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/train'
    rs_test_images_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/test'
    rs_train_labels_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_train.csv'
    rs_test_labels_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_test.csv'

    otsdf = OTSDF(images=rs_train_images_path,
                  labels=rs_train_labels_path,
                  gt_label=42,
                  metric_names=None,
                  inverse=True,
                  num_of_images=100)

    otsdf.synthesis(target_size=(48, 48), lambda_=0.5)
    otsdf.show()
    otsdf.save_matrix(path='Filter/otsdf')
    otsdf = OTSDF(images=rs_test_images_path,
                  labels=rs_test_labels_path,
                  num_of_images=4000,
                  gt_label=42,
                  batch_size=200,
                  metric_names=['accuracy', 'precision', 'recall', 'f1'],
                  peak_classifier='peak_position',
                  filter_plane='freq')
    otsdf.test_classification_quality(path_to_filter='Filter/otsdf.npy',
                                      plot_correlations_in_3d=True,
                                      plot_conf_matrix=False,
                                      plot_correlations_in_2d=True,
                                      path_to_model=None,
                                      target_size=(48, 48))


def uotsdf_usage():
    rps_train_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Rock-Paper-Scissors/train'
    rps_test_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Rock-Paper-Scissors/train'

    planes_train_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Planes/train'
    planes_test_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Planes/test'
    tanks = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Tanks/Train'
    tanks_test = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Tanks/Test'

    rs_train_images_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/train'
    rs_test_images_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/test'
    rs_train_labels_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_train.csv'
    rs_test_labels_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_test.csv'

    uotsdf = UOTSDF(images=rs_train_images_path,
                    labels=rs_train_labels_path,
                    gt_label=42,
                    metric_names=None,
                    inverse=True,
                    num_of_images=500)

    uotsdf.synthesis(target_size=(48, 48), lambda_=0.2)
    uotsdf.show()
    uotsdf.save_matrix(path='Filter/uotsdf')
    uotsdf = UOTSDF(images=rs_test_images_path,
                    labels=rs_test_labels_path,
                    num_of_images=4000,
                    gt_label=42,
                    batch_size=200,
                    metric_names=['accuracy', 'precision', 'recall', 'f1'],
                    peak_classifier='peak_position',
                    filter_plane='freq')
    uotsdf.test_classification_quality(path_to_filter='Filter/uotsdf.npy',
                                       plot_correlations_in_3d=True,
                                       plot_conf_matrix=False,
                                       plot_correlations_in_2d=True,
                                       path_to_model=None,
                                       target_size=(48, 48))


def mosse_usage():
    rps_train_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Rock-Paper-Scissors/train'
    rps_test_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Rock-Paper-Scissors/train'

    planes_train_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Planes/train'
    planes_test_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Planes/test'

    tanks_train = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Tanks/Train'
    tanks_test = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Tanks/Test'

    fruits = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/fruits-360/Train'

    rs_train_images_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/train'
    rs_test_images_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/test'
    rs_train_labels_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_train.csv'
    rs_test_labels_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_test.csv'

    mosse = MOSSE(images=rs_train_images_path,
                  labels=rs_train_labels_path,
                  gt_label=42,
                  metric_names=None,
                  inverse=True,
                  num_of_images=2000)

    mosse.synthesis(target_size=(48, 48), fwhm=4)
    mosse.show()
    mosse.save_matrix(path='Filter/mosse')
    mosse = MOSSE(images=rs_test_images_path,
                  labels=rs_test_labels_path,
                  inverse=True,
                  num_of_images=4000,
                  gt_label=42,
                  batch_size=2000,
                  metric_names=['accuracy', 'precision', 'recall', 'f1'],
                  peak_classifier='peak_position',
                  filter_plane='freq')
    mosse.test_classification_quality(path_to_filter='Filter/mosse.npy',
                                      plot_correlations_in_3d=True,
                                      plot_conf_matrix=False,
                                      plot_correlations_in_2d=True,
                                      path_to_model=None,
                                      target_size=(48, 48),
                                      use_wf=False)


def asef_usage():
    rps_train_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Rock-Paper-Scissors/train'
    rps_test_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Rock-Paper-Scissors/train'

    planes_train_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Planes/train'
    planes_test_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Planes/test'

    tanks_train = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Tanks/Train'
    tanks_test = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Tanks/Test'

    fruits = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/fruits-360/Train'

    rs_train_images_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/train'
    rs_test_images_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/test'
    rs_train_labels_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_train.csv'
    rs_test_labels_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_test.csv'

    asef = ASEF(images=rs_train_images_path,
                labels=rs_train_labels_path,
                gt_label=42,
                metric_names=None,
                inverse=True,
                num_of_images=2000)

    asef.synthesis(target_size=(48, 48), fwhm=3)
    asef.show()
    asef.save_matrix(path='Filter/asef')
    asef = ASEF(images=rs_test_images_path,
                labels=rs_test_labels_path,
                inverse=True,
                num_of_images=4000,
                gt_label=42,
                batch_size=200,
                metric_names=['accuracy', 'precision', 'recall', 'f1'],
                peak_classifier='peak_position',
                filter_plane='freq')
    asef.test_classification_quality(path_to_filter='Filter/asef.npy',
                                     plot_correlations_in_3d=True,
                                     plot_conf_matrix=False,
                                     plot_correlations_in_2d=True,
                                     path_to_model=None,
                                     target_size=(48, 48))


def ot_mach_usage():
    rps_train_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Rock-Paper-Scissors/train'
    rps_test_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Rock-Paper-Scissors/train'

    planes_train_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Planes/train'
    planes_test_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Planes/test'

    rs_train_images_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/train'
    rs_test_images_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/test'
    rs_train_labels_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_train.csv'
    rs_test_labels_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_test.csv'

    tanks = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Tanks'

    ot_mach = OT_MACH(images=tanks,
                      labels=None,
                      gt_label=0,
                      metric_names=None,
                      inverse=False,
                      num_of_images=6000)

    ot_mach.synthesis(target_size=(256, 256))
    ot_mach.show()
    ot_mach.save_matrix(path='Filter/road_signs_ot_mach_1')
    ot_mach = OT_MACH(images=tanks,
                      labels=None,
                      num_of_images=6000,
                      gt_label=0,
                      batch_size=100,
                      metric_names=['accuracy', 'precision', 'recall', 'f1'],
                      peak_classifier='peak_position',
                      filter_plane='freq')
    ot_mach.test_classification_quality(path_to_filter='Filter/road_signs_ot_mach_1.npy',
                                        plot_correlations_in_3d=True,
                                        plot_conf_matrix=False,
                                        plot_correlations_in_2d=True,
                                        path_to_model=None,
                                        target_size=(256, 256))


joint_transform_working()
corr_types = ['fourier_correlation', 'van_der_lugt', 'joint_transform']
# road_signs(gt_label=42, plot_correlations_in_2d=True, plot_correlations_in_3d=False, corr_type=corr_types[0])
# yale_face(range_gt_label=1, plot_correlations_in_2d=True, plot_correlations_in_3d=True, corr_type=corr_types[0],
#           show=False, make_holo=False)

# otsdf_usage()
# minace_usage()
# mmcf_usage()
# ot_mach_usage()
# asef_usage()
# uotsdf_usage()
