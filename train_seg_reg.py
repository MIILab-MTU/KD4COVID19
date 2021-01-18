import os
import argparse
import tensorflow as tf

from core.dataset.lung_seg_reg_dataloader import get_generator
from core.dataset.common import get_patients
from core.executor.seg_reg_model_trainer import SegRegModelTrainer3D
from core.utils import helpers

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument('--exp', type=str, default="exp/seg_reg_cv_0")
    parser.add_argument('--num_epochs', type=int, default=1001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--validate_epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)

    # data
    parser.add_argument('--patch_x', type=int, default=384)
    parser.add_argument('--patch_y', type=int, default=384)
    parser.add_argument('--patch_z', type=int, default=16)
    parser.add_argument('--data_path', type=str, default='/media/zhaochen/data/covid_seg/data/data_sh_all/numpy')
    parser.add_argument('--csv_path', type=str, default='/media/zhaochen/data/covid_seg/data/data_sh_all/classification.csv')
    parser.add_argument('--dicom_path', type=str, default='/media/zhaochen/data/covid_seg/data/data_sh_all/images')

    # cpu
    parser.add_argument('--n_workers', type=int, default=6)

    # gpu
    parser.add_argument('--gpu', type=str, default="0")

    # CV
    parser.add_argument('--cv', type=int, default=0)  # cross validation, CV=5
    parser.add_argument('--cv_max', type=int, default=5)

    # Model
    #parser.add_argument('--restore_path', type=str, default="./experiments/exp_s0_cv4/seg_model.cpkt")
    parser.add_argument('--n_filter', type=int, default=32)
    parser.add_argument('--dropout_p', type=float, default=0.8)
    parser.add_argument('--l2', type=float, default=0.2)

    # inference data
    parser.add_argument('--train', type=helpers.str2bool, default=False)
    #parser.add_argument('--data_dir', type=str, default="/home/zhaochen/Desktop/covid/data_sh_rnn")
    #parser.add_argument('--data_dir', type=str, default="/media/zhaochen/data/covid/data_sh_normal")
    #parser.add_argument('--data_dir', type=str, default="/media/zhaochen/data/covid/data_sh_rnn")
    parser.add_argument('--data_dir', type=str, default="/media/zhaochen/data/covid_seg/data_sh_normal")

    # feature analysis
    parser.add_argument('--feature_pattern', type=str, default="pred")
    parser.add_argument('--feature_extract_clf_csv', type=str, default="/media/zhaochen/data/covid_seg/data/data_sh_all/classification_for_prediction.csv")
    parser.add_argument('--feature_coor_threshold', type=float, default="0.83")


    args = parser.parse_args()

    #print(args.brightness)
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # train
    if args.train:
        # load data
        data_gen_tr, _, train_patients, val_patients = get_generator(args)
        print("[x] train 3D segmentation seg_model")
        trainer_3d = SegRegModelTrainer3D(args, sess, input_channel=2, output_channel=1)
        trainer_3d.train(data_gen_tr, train_patients, val_patients)
    else:
        trainer_3d = SegRegModelTrainer3D(args, sess, input_channel=2, output_channel=1)
        #trainer_3d.load_inference()

        # parser.add_argument('--data_dir', type=str, default="/media/zhaochen/data/covid/data_sh_segmentation")
        # parser.add_argument('--data_dir', type=str, default="/media/zhaochen/data/covid/data_sh_normal_pneumonia")
        #trainer_3d.load_inference2()

        # parser.add_argument('--data_dir', type=str, default="/media/zhaochen/data/covid/data_sh_normal")
        #trainer_3d.load_inference3()
        trainer_3d.load_inference_for_feature_analysis()
