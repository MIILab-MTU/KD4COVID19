import tensorflow as tf
import tensorflow.contrib.slim as slim

from skimage import measure, morphology, segmentation
from core.dataset.common import get_pixel_spacing
from core.seg_model.models import VNet, count_params, STN
from core.seg_model.transform2_3d import batch_affine_warp3d as spatial_transformer_network_3d
from core.make_figures.make_edges import *
from core.utils.evaluator import evaluate_single_image, evaluate_single_image_distance
from core.utils.visualize import DataVisualizer
from core.utils.lung_segmentation_threshold import segment_morphology2
from core.utils.helpers import generate_largest_region_threshold

from generate_data_and_shape_prior import load_dicoms, load_dicoms_itk
from core.utils.evaluator import threshold_by_otsu

from core.feature_analysis import feature_extractor, feature_analyzer

class SegRegModelTrainer3D(object):

    def __init__(self, args, sess, input_channel, output_channel):
        self.args = args
        self.input_channel = input_channel
        self.output_channel = output_channel

        # build placeholders
        self.x = tf.placeholder('float', shape=[None, self.args.patch_x, self.args.patch_y, self.args.patch_z, self.input_channel], name="x")
        self.y_template = tf.placeholder('float', shape=[None, self.args.patch_x, self.args.patch_y, self.args.patch_z, self.output_channel], name="template")
        self.y_gt = tf.placeholder('float', shape=[None, self.args.patch_x, self.args.patch_y, self.args.patch_z, self.output_channel], name="gt")
        self.lr = tf.placeholder('float', name="lr")
        self.drop = tf.placeholder('float', name="drop")
        self.loss_weight = tf.placeholder('float', name="loss_weight")

        # build seg_model (image segmentation seg_model - V-Net)
        with tf.variable_scope("vnet") as scope:
            self.vnet = VNet(self.args.n_filter, self.x, self.output_channel, self.drop, self.args.l2)
            self.y_pred = self.vnet.create_model()

        # variables = slim.get_variables_to_restore()
        # saver = tf.train.Saver([v for v in variables])
        # print("[x] restore seg_model ...")
        # saver.restore(sess, self.args.restore_path)

        # build seg_model (spatial transformation network)
        self.y_pred_bin = tf.cast(self.y_pred > 0.5, "float")

        # stn input: the combination of the template and the binary prediction
        stn_inputs = tf.concat(values=[self.y_template, self.y_pred_bin], axis=4)

        with tf.variable_scope("stn") as scope:
            self.stn = STN(stn_inputs, self.args.n_filter, self.args.l2)
            self.theta = self.stn.create_model()
            self.theta = tf.clip_by_value(self.theta, clip_value_min=0.8, clip_value_max=1.2)
            theta_mask = [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]]
            self.theta = self.theta * theta_mask

            self.y_warped = spatial_transformer_network_3d(self.y_pred_bin, self.theta)

        self.cost_img, self.cost_transform, self.cost_all = self.get_cost()
        self.sess = sess
        self.saver = tf.train.Saver()

        count_params()

        # running parameters
        self.seg_loss = []
        self.reg_loss = []

    def load_inference_for_feature_analysis(self):
        """
        load model, predict and calculate feature
        :return:
        """
        assert self.args.feature_pattern in ["pred", "y"]
        print("[x] restore seg_model ...")
        self.saver.restore(self.sess, os.path.join(self.args.exp, 'seg_model.cpkt'))
        feature_analyze_path = os.path.join(self.args.exp, "feature_analyze")

        if not os.path.isdir(feature_analyze_path):
            os.makedirs(feature_analyze_path)

        patient_names = []
        for data_x_file in glob(os.path.join(self.args.data_path, "*_x.npy")):
            patient_name = data_x_file[data_x_file.rfind("/")+1: data_x_file.rfind("_x.npy")]
            patient_names.append(patient_name)

        if self.args.feature_pattern == "pred":
            for patient_name in patient_names:
                print("-------------------------{}-----------------------".format(patient_name))
                data_x = np.load(os.path.join(self.args.data_path, "{}_x.npy".format(patient_name)))
                data_y = np.load(os.path.join(self.args.data_path, "{}_y.npy".format(patient_name)))
                data_template = np.load(os.path.join(self.args.data_path, "{}_template.npy".format(patient_name)))

                data_x = np.expand_dims(data_x, axis=3)
                data_x = np.expand_dims(data_x, axis=0)
                data_template = np.expand_dims(data_template, axis=3)
                data_template = np.expand_dims(data_template, axis=0)

                data_for_input = np.concatenate([data_x, data_template], axis=4)

                pred_y, _ = self.inference_whole_volume(data_for_input, data_template)
                seg_y = threshold_by_otsu(pred_y, flatten=False)

                np.save(file=os.path.join(feature_analyze_path, "{}_pred.npy".format(patient_name)), arr=np.squeeze(seg_y))
                np.save(file=os.path.join(feature_analyze_path, "{}_y.npy".format(patient_name)), arr=np.squeeze(data_y))
                np.save(file=os.path.join(feature_analyze_path, "{}_x.npy".format(patient_name)), arr=np.squeeze(data_x))

                dv = DataVisualizer([np.squeeze(data_x),
                                     np.asarray(np.squeeze(data_template), dtype=np.float32),
                                     np.asarray(np.squeeze(seg_y), dtype=np.float32),
                                     np.asarray(np.squeeze(data_y), dtype=np.float32)],
                                     save_path=os.path.join(feature_analyze_path, "{}.png".format(patient_name)))
                dv.visualize_np(np.squeeze(data_x).shape[2], patch_size=self.args.patch_x)
        else:
            for patient_name in patient_names:
                print("-------------------------{}-----------------------".format(patient_name))
                data_x = np.load(os.path.join(self.args.data_path, "{}_x.npy".format(patient_name)))
                data_y = np.load(os.path.join(self.args.data_path, "{}_y.npy".format(patient_name)))
                data_template = np.load(os.path.join(self.args.data_path, "{}_template.npy".format(patient_name)))

                np.save(file=os.path.join(feature_analyze_path, "{}_y.npy".format(patient_name)), arr=np.squeeze(data_y))
                np.save(file=os.path.join(feature_analyze_path, "{}_x.npy".format(patient_name)), arr=np.squeeze(data_x))

                dv = DataVisualizer([np.squeeze(data_x),
                                     np.asarray(np.squeeze(data_template), dtype=np.float32),
                                     np.asarray(np.squeeze(data_y), dtype=np.float32)],
                                     save_path=os.path.join(feature_analyze_path, "{}.png".format(patient_name)))
                dv.visualize_np(np.squeeze(data_x).shape[2], patch_size=self.args.patch_x)

        print("[x] extract feature")
        fe = feature_extractor.FeatureExtractor(data_root=feature_analyze_path,
                                                csv_save_path=os.path.join(feature_analyze_path, "extracted_features_{}.csv".format(self.args.feature_pattern)),
                                                class_csv_file=self.args.feature_extract_clf_csv,
                                                pattern=self.args.feature_pattern)
        fe.extract_features()


    def load_inference3(self):
        """
        load and predict lung mask for normal patients
        # parser.add_argument('--data_dir', type=str, default="/media/zhaochen/data/covid/data_sh_normal")
        """
        print("[x] restore seg_model ...")
        self.saver.restore(self.sess, os.path.join(self.args.exp, 'model.cpkt'))

        patient_dirs = os.listdir(os.path.join(self.args.data_dir, "images"))

        if not os.path.isdir(os.path.join(self.args.data_dir, "numpy")):
            os.makedirs(os.path.join(self.args.data_dir, "numpy"))

        for patient_dir in patient_dirs:
            print("---------------------{}-------------------".format(patient_dir))
            patient_dcm_path = os.path.join(self.args.data_dir, "images", patient_dir)
            print(patient_dcm_path)
            data_x = load_dicoms_itk(os.path.join(patient_dcm_path, "*.dcm"))
            data_prior = segment_morphology2(data_x, volume_ratio=0.005, binary_threshold=-320)

            crop_size = 64
            data_x = data_x[crop_size: data_x.shape[0] - crop_size,
                     crop_size: data_x.shape[1] - crop_size, :]
            data_prior = data_prior[crop_size: data_prior.shape[0] - crop_size,
                         crop_size: data_prior.shape[1] - crop_size, :]

            data_x = np.expand_dims(data_x, axis=3)
            data_x = np.expand_dims(data_x, axis=0)
            data_prior = np.expand_dims(data_prior, axis=3)
            data_prior = np.expand_dims(data_prior, axis=0)

            data_for_input = np.concatenate([data_x, data_prior], axis=4)

            pred_y, _ = self.inference_whole_volume(data_for_input, data_prior)
            seg_y = threshold_by_otsu(pred_y, flatten=False)

            # read gt
            gt_file_path = os.path.join(self.args.data_dir, "binary", patient_dir, "label")
            gt_image_paths = glob(os.path.join(gt_file_path, "*.png"))
            gt_image_paths = sorted(gt_image_paths)[::-1]
            gt = np.zeros((512, 512, len(gt_image_paths)))

            for i in range(len(gt_image_paths)):
                gt[:, :, i] = cv2.imread(gt_image_paths[i], cv2.IMREAD_GRAYSCALE)

            gt = gt[crop_size: gt.shape[0] - crop_size, crop_size: gt.shape[1] - crop_size, :]
            gt = np.clip(gt, 0, 1)

            np.save(file=os.path.join(self.args.data_dir, "numpy", "{}_x.npy".format(patient_dir)), arr=np.squeeze(data_x))
            np.save(file=os.path.join(self.args.data_dir, "numpy", "{}_template.npy".format(patient_dir)), arr=np.squeeze(data_prior))
            np.save(file=os.path.join(self.args.data_dir, "numpy", "{}_y.npy".format(patient_dir)), arr=np.squeeze(gt))
            np.save(file=os.path.join(self.args.data_dir, "numpy", "{}_pred.npy".format(patient_dir)), arr=np.squeeze(seg_y))

            dv = DataVisualizer([np.squeeze(data_x),
                                 np.asarray(np.squeeze(data_prior), dtype=np.float32),
                                 np.asarray(np.squeeze(seg_y), dtype=np.float32),
                                 np.asarray(np.squeeze(gt), dtype=np.float32)],
                                save_path=os.path.join(self.args.data_dir, "numpy", "{}.png".format(patient_dir)))
            dv.visualize_np(np.squeeze(data_x).shape[2], patch_size=self.args.patch_x)

    def load_inference2(self):
        """
        load and predict normal pneumonia patients and covid patients
        # parser.add_argument('--data_dir', type=str, default="/media/zhaochen/data/covid/data_sh_segmentation")
        # parser.add_argument('--data_dir', type=str, default="/media/zhaochen/data/covid/data_sh_normal_pneumonia")
        """
        print("[x] restore seg_model ...")
        self.saver.restore(self.sess, os.path.join(self.args.exp, 'model.cpkt'))

        patient_dirs = os.listdir(os.path.join(self.args.data_dir, "images"))
        for patient_dir in patient_dirs:
            print("---------------------{}-------------------".format(patient_dir))
            patient_dcm_path = os.path.join(self.args.data_dir, "images", patient_dir)
            print(patient_dcm_path)
            data_x = load_dicoms(os.path.join(patient_dcm_path, "*.dcm"))
            data_prior = segment_morphology2(data_x, volume_ratio=0.005, binary_threshold=-320)

            crop_size = 64
            data_x = data_x[crop_size: data_x.shape[0] - crop_size,
                            crop_size: data_x.shape[1] - crop_size, :]
            data_prior = data_prior[crop_size: data_prior.shape[0] - crop_size,
                                    crop_size: data_prior.shape[1] - crop_size, :]

            data_x = np.expand_dims(data_x, axis=3)
            data_x = np.expand_dims(data_x, axis=0)
            data_prior = np.expand_dims(data_prior, axis=3)
            data_prior = np.expand_dims(data_prior, axis=0)

            data_for_input = np.concatenate([data_x, data_prior], axis=4)

            pred_y, _ = self.inference_whole_volume(data_for_input, data_prior)
            seg_y = threshold_by_otsu(pred_y, flatten=False)

            # read gt
            gt_file_path = os.path.join(self.args.data_dir, "binary", patient_dir, "label")
            gt_image_paths = glob(os.path.join(gt_file_path, "*.png"))
            gt_image_paths = sorted(gt_image_paths)[::-1]
            gt = np.zeros((512, 512, len(gt_image_paths)))

            for i in range(len(gt_image_paths)):
                gt[:, :, i] = cv2.imread(gt_image_paths[i], cv2.IMREAD_GRAYSCALE)

            gt = gt[crop_size: gt.shape[0] - crop_size, crop_size: gt.shape[1] - crop_size, :]
            gt = np.clip(gt, 0, 1)

            np.save(file=os.path.join(self.args.data_dir, "numpy", "{}_x.npy".format(patient_dir)), arr=np.squeeze(data_x))
            np.save(file=os.path.join(self.args.data_dir, "numpy", "{}_template.npy".format(patient_dir)), arr=np.squeeze(data_prior))
            np.save(file=os.path.join(self.args.data_dir, "numpy", "{}_y.npy".format(patient_dir)), arr=np.squeeze(gt))
            np.save(file=os.path.join(self.args.data_dir, "numpy", "{}_pred.npy".format(patient_dir)), arr=np.squeeze(seg_y))

            dv = DataVisualizer([np.squeeze(data_x),
                                 np.asarray(np.squeeze(data_prior), dtype=np.float32),
                                 np.asarray(np.squeeze(seg_y), dtype=np.float32),
                                 np.asarray(np.squeeze(gt), dtype=np.float32)],
                                save_path=os.path.join(self.args.data_dir, "numpy", "{}.png".format(patient_dir)))
            dv.visualize_np(np.squeeze(data_x).shape[2], patch_size=self.args.patch_x)

    def load_inference(self):
        """
        load time sequence data and predict the lung masks
        Returns:

        """
        print("[x] restore seg_model ...")
        self.saver.restore(self.sess, os.path.join(self.args.exp, 'model.cpkt'))

        patient_dirs = os.listdir(os.path.join(self.args.data_dir, "images"))
        if not os.path.isdir(os.path.join(self.args.data_dir, "numpy")):
            os.makedirs(os.path.join(self.args.data_dir, "numpy"))

        for patient_dir in patient_dirs:
            print("---------------------{}-------------------".format(patient_dir))
            patient_dates_dir = os.listdir(os.path.join(self.args.data_dir, "images", patient_dir))
            patient_dates_dir = sorted(patient_dates_dir)
            for patient_date_dir in patient_dates_dir:
                patient_dcm_path = os.path.join(self.args.data_dir, "images", patient_dir, patient_date_dir)
                print(patient_dcm_path)
                data_x = load_dicoms(os.path.join(patient_dcm_path, "*.dcm"))
                data_prior = segment_morphology2(data_x, volume_ratio=0.005, binary_threshold=-320)

                crop_size = 64
                data_x = data_x[crop_size: data_x.shape[0] - crop_size,
                                crop_size: data_x.shape[1] - crop_size, :]
                data_prior = data_prior[crop_size: data_prior.shape[0] - crop_size,
                                        crop_size: data_prior.shape[1] - crop_size, :]

                data_x = np.expand_dims(data_x, axis=3)
                data_x = np.expand_dims(data_x, axis=0)
                data_prior = np.expand_dims(data_prior, axis=3)
                data_prior = np.expand_dims(data_prior, axis=0)

                data_for_input = np.concatenate([data_x, data_prior], axis=4)

                pred_y, _ = self.inference_whole_volume(data_for_input, data_prior)
                seg_y = threshold_by_otsu(pred_y, flatten=False)

                np.save(file=os.path.join(self.args.data_dir, "numpy", "{}_{}_prob.npy".format(patient_dir, patient_date_dir)),
                        arr=np.squeeze(pred_y))
                np.save(file=os.path.join(self.args.data_dir, "numpy", "{}_{}_y.npy".format(patient_dir, patient_date_dir)),
                        arr=np.squeeze(seg_y))
                np.save(file=os.path.join(self.args.data_dir, "numpy", "{}_{}_x.npy".format(patient_dir, patient_date_dir)),
                        arr=np.squeeze(data_x))
                np.save(file=os.path.join(self.args.data_dir, "numpy", "{}_{}_template.npy".format(patient_dir, patient_date_dir)),
                        arr=np.squeeze(data_prior))

    def get_cost(self, dice=False):
        if dice:
            Z, H, W, C = self.y_gt.get_shape().as_list()[1:]
            smooth = 1e-5
            pred_flat = tf.reshape(self.y_pred, [-1, H * W * C * Z])
            true_flat = tf.reshape(self.y_gt, [-1, H * W * C * Z])
            intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
            denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
            loss1 = -tf.reduce_mean(intersection / denominator)

            warped_true_flat = tf.reshape(self.y_warped, [-1, H * W * C * Z])
            template_true_flat = tf.reshape(self.y_template, [-1, H * W * C * Z])
            intersection = 2 * tf.reduce_sum(warped_true_flat * template_true_flat, axis=1) + smooth
            denominator = tf.reduce_sum(warped_true_flat, axis=1) + tf.reduce_sum(template_true_flat, axis=1) + smooth
            loss2 = -tf.reduce_mean(intersection / denominator)
            return loss1, self.loss_weight * loss2, loss1 + self.loss_weight * loss2
        else:
            loss1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_pred, labels=self.y_gt)
            loss2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_warped, labels=self.y_gt)
            loss1 = tf.reduce_mean(loss1)
            loss2 = tf.reduce_mean(loss2)

            return loss1, self.loss_weight * loss2, loss1 + self.loss_weight * loss2

    def save_loss_function(self):
        np.save(file=os.path.join(self.args.exp, "loss_seg.npy"), arr=np.array(self.seg_loss))
        np.save(file=os.path.join(self.args.exp, "loss_reg.npy"), arr=np.array(self.reg_loss))

    def train(self, data_generator_train, train_patients, val_patients):
        num_train_data = len(train_patients)
        num_val_data = len(val_patients)
        train_seg_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost_img)
        train_reg_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost_transform)
        train_all_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost_all)
        init = tf.global_variables_initializer()

        global_dice = 0.

        self.sess.run(init)

        for epoch in tqdm(range(self.args.num_epochs)):
            print("[x] epoch: %d, training" % epoch)
            num_batch = num_train_data // self.args.batch_size
            epoch_loss = 0.
            losses_seg = 0.
            losses_reg = 0.
            for mini_batch in tqdm(range(num_batch)):
                batch_data = next(data_generator_train)
                batch_data_x = np.concatenate([batch_data['data'], batch_data['prior']], axis=1)

                if epoch < int(self.args.num_epochs * 0.2):
                    _, train_loss = self.sess.run([train_seg_op, self.cost_img],
                                                  feed_dict={self.x: np.transpose(batch_data_x, (0, 2, 3, 4, 1)),
                                                             self.y_gt: np.transpose(batch_data['seg'],
                                                                                     (0, 2, 3, 4, 1)),
                                                             self.lr: self.args.lr,
                                                             self.drop: self.args.dropout_p})

                elif int(self.args.num_epochs) * 0.2 <= epoch < int(self.args.num_epochs) * 0.4:
                    _, train_loss = self.sess.run([train_reg_op, self.cost_transform],
                                                  feed_dict={self.x: np.transpose(batch_data_x, (0, 2, 3, 4, 1)),
                                                             self.y_gt: np.transpose(batch_data['seg'], (0, 2, 3, 4, 1)),
                                                             self.y_template: np.transpose(batch_data['prior'], (0, 2, 3, 4, 1)),
                                                             self.lr: self.args.lr,
                                                             self.drop: self.args.dropout_p,
                                                             self.loss_weight: 0.1})
                else:
                    _, train_loss = self.sess.run([train_all_op, self.cost_all],
                                                  feed_dict={self.x: np.transpose(batch_data_x, (0, 2, 3, 4, 1)),
                                                             self.y_gt: np.transpose(batch_data['seg'], (0, 2, 3, 4, 1)),
                                                             self.y_template: np.transpose(batch_data['prior'], (0, 2, 3, 4, 1)),
                                                             self.lr: self.args.lr * 0.1,
                                                             self.drop: self.args.dropout_p,
                                                             self.loss_weight: 0.1})
                epoch_loss += train_loss

                seg_loss = self.sess.run(self.cost_img,
                                         feed_dict={self.x: np.transpose(batch_data_x, (0, 2, 3, 4, 1)),
                                                    self.y_gt: np.transpose(batch_data['seg'], (0, 2, 3, 4, 1)),
                                                    self.drop: self.args.dropout_p})
                reg_loss = self.sess.run(self.cost_transform,
                                         feed_dict={self.x: np.transpose(batch_data_x, (0, 2, 3, 4, 1)),
                                                    self.y_gt: np.transpose(batch_data['seg'], (0, 2, 3, 4, 1)),
                                                    self.y_template: np.transpose(batch_data['prior'], (0, 2, 3, 4, 1)),
                                                    self.lr: self.args.lr,
                                                    self.drop: self.args.dropout_p,
                                                    self.loss_weight: 0.1})
                losses_seg += seg_loss
                losses_reg += reg_loss

            self.reg_loss.append(losses_reg / num_batch)
            self.seg_loss.append(losses_seg / num_batch)

            print("[x] epoch: %d, average loss: %f, reg_loss: %f, seg_loss: %f"
                    % (epoch, epoch_loss / num_batch, losses_reg / num_batch, losses_seg / num_batch))

            if (epoch) % self.args.validate_epoch == 0:
                print("[x] epoch: %d, validate" % epoch)
                model_results = []
                for mini_batch in range(num_val_data):
                    patient_id = val_patients[mini_batch]

                    data_x = np.load("{}/{}_x.npy".format(self.args.data_path, patient_id), mmap_mode='r')
                    data_y = np.load("{}/{}_y.npy".format(self.args.data_path, patient_id), mmap_mode='r')
                    data_prior = np.load("{}/{}_template.npy".format(self.args.data_path, patient_id), mmap_mode='r')

                    data_x = np.expand_dims(data_x, axis=3)
                    data_x = np.expand_dims(data_x, axis=0)
                    data_y = np.expand_dims(data_y, axis=3)
                    data_y = np.expand_dims(data_y, axis=0)
                    data_prior = np.expand_dims(data_prior, axis=3)
                    data_prior = np.expand_dims(data_prior, axis=0)

                    data = np.concatenate([data_x, data_prior], axis=4)
                    pred_y, warped_pred_y = self.inference_whole_volume(data, data_prior)

                    model_result = self.evaluate_result(np.squeeze(data_y),
                                                        np.squeeze(pred_y),
                                                        patient_id,
                                                        np.squeeze(data))
                    model_results.append(model_result)

                dice, hd, asd, sn, sp = self.save_result(model_results, epoch, pattern="")

                if dice > global_dice:
                    global_dice = dice
                    self.saver.save(self.sess, os.path.join(self.args.exp, 'seg_model.cpkt'))
                print("[x] epoch {}, dice in segmentation = {}, hd = {}, asd = {}, sn = {}, sp = {}".format(epoch, dice, hd, asd, sn, sp))

        self.save_loss_function()

    # def predict(self, batch_data):
    #     batch_pred = self.sess.run(self.y_pred, feed_dict={self.x: np.transpose(batch_data['data'], (0, 2, 3, 4, 1)),
    #                                                        self.drop: 1.})
    #     batch_warped_pred = self.sess.run(self.y_warped,
    #                                       feed_dict={self.x: np.transpose(batch_data['data'], (0, 2, 3, 4, 1)),
    #                                                  self.y_template: np.transpose})
    #     return batch_pred, batch_warped_pred

    def inference_whole_volume(self, data, template, interval_z=1):
        inferenced_volume = np.zeros([1, data.shape[1], data.shape[2], data.shape[3], self.output_channel], dtype=np.float)
        inferenced_time = np.zeros([1, data.shape[1], data.shape[2], data.shape[3], self.output_channel], dtype=np.float)
        warped_inferenced_volume = np.zeros([1, data.shape[1], data.shape[2], data.shape[3], self.output_channel], dtype=np.float)
        for slice_z in range(0, data.shape[3] - self.args.patch_z + 1, interval_z):
            part_pred = self.sess.run(self.y_pred,
                                      feed_dict={self.x: data[:, :, :, slice_z: slice_z + self.args.patch_z, :],
                                                 self.drop: 1.})
            warped_part_pred = self.sess.run(self.y_warped,
                                             feed_dict={self.x: data[:, :, :, slice_z: slice_z + self.args.patch_z, :],
                                                        self.y_template: template[:, :, :, slice_z: slice_z + self.args.patch_z, :],
                                                        self.drop: 1.})
            inferenced_volume[:, :, :, slice_z: slice_z + self.args.patch_z:, :] += part_pred
            warped_inferenced_volume[:, :, :, slice_z: slice_z + self.args.patch_z:, :] += warped_part_pred
            inferenced_time[:, :, :, slice_z: slice_z + self.args.patch_z:, :] += 1
        inferenced_volume = inferenced_volume / inferenced_time
        warped_inferenced_volume = warped_inferenced_volume / inferenced_time
        return inferenced_volume, warped_inferenced_volume

    def evaluate_result(self, gt, y_pred, patient_id, x):
        s = get_pixel_spacing(os.path.join(self.args.dicom_path, patient_id))
        binary_image, auc_roc, auc_pr, dice_coeff, acc, sensitivity, specificity = evaluate_single_image(y_pred, gt)

        # new:  generate largest region
        binary_image = generate_largest_region_threshold(binary_image, ratio=0.02)
        _, auc_roc, auc_pr, dice_coeff, acc, sensitivity, specificity = evaluate_single_image(binary_image, gt)

        try:
            hd, asd = evaluate_singlez_image_distance(binary_image, gt, spacing=s)
        except:
            hd, asd = 10000., 10000.

        result = {'patient_id': patient_id,
                  'bin': binary_image, 'pred': y_pred, 'x': x[:, :, :, 0], 'y': gt, 'prior': x[:, :, :, 1],
                  'acc': acc, 'sn': sensitivity, 'sp': specificity,
                  'dice': dice_coeff, 'auc_roc': auc_roc, 'auc_pr': auc_pr,
                  'hd': hd, 'asd': asd}
        return result

    def save_result(self, evaluated_results, epoch, pattern):
        if not os.path.isdir(self.args.exp + "/%04d" % (epoch)):
            os.makedirs(self.args.exp + "/%04d" % (epoch))

        target = open(self.args.exp + "/%04d/val%s.csv" % (epoch, pattern), "w")
        target.write('patient,auc_roc,auc_pr,dice,acc,sn,sp,hd,asd\n')

        dices = []
        hds = []
        asds = []
        sns = []
        sps = []

        for i, evaluated_result in enumerate(evaluated_results):
            target.write("%s,%f,%f,%f,%f,%f,%f,%f,%f\n" % (evaluated_result['patient_id'],
                                                           evaluated_result['auc_roc'],
                                                           evaluated_result['auc_pr'],
                                                           evaluated_result['dice'],
                                                           evaluated_result['acc'],
                                                           evaluated_result['sn'],
                                                           evaluated_result['sp'],
                                                           evaluated_result['hd'],
                                                           evaluated_result['asd']))
            target.flush()
            dices.append(evaluated_result['dice'])
            hds.append(evaluated_result['hd'])
            asds.append(evaluated_result['asd'])
            sns.append(evaluated_result['sn'])
            sps.append(evaluated_result['sp'])

            np.save(file=self.args.exp + "/%04d/%s_x.npy" % (epoch, evaluated_result['patient_id']), arr=evaluated_result['x'])
            np.save(file=self.args.exp + "/%04d/%s_%sbin.npy" % (epoch, evaluated_result['patient_id'], pattern), arr=evaluated_result['bin']),
            np.save(file=self.args.exp + "/%04d/%s_%spred.npy" % (epoch, evaluated_result['patient_id'], pattern), arr=evaluated_result['pred'])
            np.save(file=self.args.exp + "/%04d/%s_gt.npy" % (epoch, evaluated_result['patient_id']), arr=evaluated_result['y'])

            dv = DataVisualizer([np.squeeze(evaluated_result['x']),
                                 np.squeeze(evaluated_result['prior']),
                                 np.squeeze(evaluated_result['bin']),
                                 np.squeeze(evaluated_result['pred']),
                                 np.squeeze(evaluated_result['y'])],
                                save_path=self.args.exp + "/%04d/%s%s.png" % (epoch, evaluated_result['patient_id'], pattern))
            dv.visualize_np(evaluated_result['x'].shape[2], patch_size=self.args.patch_x)

        target.close()

        return np.mean(np.array(dices)), np.mean(np.array(hds)), np.mean(np.array(asds)), np.mean(np.array(sns)), np.mean(np.array(sps))
