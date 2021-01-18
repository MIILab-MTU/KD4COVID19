import tensorflow as tf

from core.utils.evaluator import evaluate_single_image, evaluate_single_image_distance
from core.utils.helpers import generate_largest_region_threshold

from core.dataset.lung_seg_dataloader import get_generator
from core.dataset.common import get_pixel_spacing

from core.utils.evaluator import threshold_by_otsu
from core.utils.visualize import DataVisualizer

from core.make_figures.make_edges import *
from core.seg_model.models import VNet, count_params

from core.feature_analysis import feature_extractor, feature_analyzer


class SegPriorModelTrainer3D(object):

    def __init__(self, args, sess, input_channel, output_channel):
        self.args = args
        self.input_channel = input_channel
        self.output_channel = output_channel

        self.x = tf.placeholder('float', shape=[None, self.args.patch_x, self.args.patch_y, self.args.patch_z, self.input_channel], name="x")
        self.y_gt = tf.placeholder('float', shape=[None, self.args.patch_x, self.args.patch_y, self.args.patch_z, self.output_channel], name="gt")
        self.lr = tf.placeholder('float', name="lr")
        self.drop = tf.placeholder('float', name="drop")

        with tf.variable_scope("vnet") as scope:
            vnet = VNet(self.args.n_filter, self.x, self.output_channel, self.drop, self.args.l2)
            self.y_pred = vnet.create_model()
        self.cost = self.get_cost()
        self.sess = sess
        self.saver = tf.train.Saver()
        count_params()

    def load_inference_for_feature_analysis(self):
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
                pred_y = self.inference_whole_volume(data_for_input)
                seg_y = threshold_by_otsu(pred_y, flatten=False)

                np.save(file=os.path.join(feature_analyze_path, "{}_pred.npy".format(patient_name)),
                        arr=np.squeeze(seg_y))
                np.save(file=os.path.join(feature_analyze_path, "{}_y.npy".format(patient_name)),
                        arr=np.squeeze(data_y))
                np.save(file=os.path.join(feature_analyze_path, "{}_x.npy".format(patient_name)),
                        arr=np.squeeze(data_x))

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
        fe = feature_extractor.FeatureExtractor(
            data_root=feature_analyze_path,
            csv_save_path=os.path.join(feature_analyze_path, "extracted_features_{}.csv".format(self.args.feature_pattern)),
            class_csv_file=self.args.feature_extract_clf_csv,
            pattern=self.args.feature_pattern)
        fe.extract_features()

        print("[x] feature analysis")
        fa = feature_analyzer.FeatureAnalyzer(
            extracted_csv_path=os.path.join(feature_analyze_path, "extracted_features_{}.csv".format(self.args.feature_pattern)),
            pearson_threshold=self.args.feature_coor_threshold,
            result_save_path=os.path.join(self.args.exp, "feature_extraction_{}".format(self.args.feature_pattern)))
        fa.analyze()

    def get_cost(self, dice_loss=False):

        if dice_loss:
            # dice
            Z, H, W, C = self.y_gt.get_shape().as_list()[1:]
            smooth = 1e-5
            pred_flat = tf.reshape(self.y_pred, [-1, H * W * C * Z])
            true_flat = tf.reshape(self.y_gt, [-1, H * W * C * Z])
            intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
            denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
            loss = -tf.reduce_mean(intersection / denominator)
            return loss
        else:
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_pred, labels=self.y_gt)
            loss = tf.reduce_mean(losses)
            return loss

    def train(self, data_generator_train, train_patients, val_patients):
        num_train_data = len(train_patients)
        num_val_data = len(val_patients)
        train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        init = tf.global_variables_initializer()

        global_dice = 0.

        self.sess.run(init)

        for epoch in tqdm(range(self.args.num_epochs)):
            print("[x] epoch: %d, training" % epoch)
            num_batch = num_train_data//self.args.batch_size
            epoch_loss = 0.
            for mini_batch in tqdm(range(num_batch)):
                batch_data = next(data_generator_train)
                batch_data_x = np.concatenate([batch_data['data'], batch_data['prior']], axis=1)

                _, train_loss = self.sess.run([train_op, self.cost],
                                              feed_dict={self.x: np.transpose(batch_data_x, (0, 2, 3, 4, 1)),
                                                         self.y_gt: np.transpose(batch_data['seg'], (0, 2, 3, 4, 1)),
                                                         self.lr: self.args.lr,
                                                         self.drop: self.args.dropout_p})
                epoch_loss += train_loss

            print("[x] epoch: %d, loss: %f" % (epoch, epoch_loss/num_batch))
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
                    pred_y = self.inference_whole_volume(data)

                    model_result = self.evaluate_result(np.squeeze(data_y),
                                                        np.squeeze(pred_y),
                                                        patient_id,
                                                        np.squeeze(data))
                    model_results.append(model_result)

                dice, hd, asd, sn, sp = self.save_result(model_results, epoch)
                if dice > global_dice:
                    global_dice = dice
                    self.saver.save(self.sess, os.path.join(self.args.exp, 'seg_model.cpkt'))
                print("[x] epoch {}, dice in segmentation = {}, hd = {}, asd = {}, sn = {}, sp = {}".format(epoch, dice, hd, asd, sn, sp))

    def inference_whole_volume(self, data, interval_z=1):
        """
        inference segmentation results on whole slices
        :param data: shape [1, patch_x, patch_y, patch_z, channel]
        :return:
        """
        inferenced_volume = np.zeros([1, data.shape[1], data.shape[2], data.shape[3], self.output_channel], dtype=np.float)
        inferenced_time = np.zeros([1, data.shape[1], data.shape[2], data.shape[3], self.output_channel], dtype=np.float)
        for slice_z in range(0, data.shape[3] - self.args.patch_z + 1, interval_z):
            part_pred = self.sess.run(self.y_pred,
                                      feed_dict={self.x: data[:, :, :, slice_z: slice_z+self.args.patch_z, :],
                                                 self.drop: 1.})
            inferenced_volume[:, :, :, slice_z: slice_z+self.args.patch_z:, :] += part_pred
            inferenced_time[:, :, :, slice_z: slice_z+self.args.patch_z:, :] += 1
        inferenced_volume = inferenced_volume/inferenced_time
        return inferenced_volume

    def evaluate_result(self, gt, y_pred, patient_id, x):
        s = get_pixel_spacing(os.path.join(self.args.dicom_path, patient_id))
        binary_image, auc_roc, auc_pr, dice_coeff, acc, sensitivity, specificity = evaluate_single_image(y_pred, gt)

        # new:  generate largest region
        binary_image = generate_largest_region_threshold(binary_image, ratio=0.02)
        _, auc_roc, auc_pr, dice_coeff, acc, sensitivity, specificity = evaluate_single_image(binary_image, gt)

        try:
            hd, asd = evaluate_single_image_distance(binary_image, gt, spacing=s)
        except:
            hd, asd = 10000., 10000.

        result = {'patient_id': patient_id,
                  'bin': binary_image, 'pred': y_pred, 'x': x[:, :, :, 0], 'y': gt, 'prior': x[:, :, :, 1],
                  'acc': acc, 'sn': sensitivity, 'sp': specificity,
                  'dice': dice_coeff, 'auc_roc': auc_roc, 'auc_pr': auc_pr,
                  'hd': hd, 'asd': asd}
        return result

    def save_result(self, evaluated_results, epoch):
        if not os.path.isdir(self.args.exp + "/%04d" % (epoch)):
            os.makedirs(self.args.exp + "/%04d" % (epoch))

        target = open(self.args.exp + "/%04d/val.csv" % (epoch), "w")
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

            np.save(file=self.args.exp + "/%04d/%s_x.npy" % (epoch, evaluated_result['patient_id']), arr=np.squeeze(evaluated_result['x']))
            np.save(file=self.args.exp + "/%04d/%s_bin.npy" % (epoch, evaluated_result['patient_id']), arr=np.squeeze(evaluated_result['bin']))
            np.save(file=self.args.exp + "/%04d/%s_pred.npy" % (epoch, evaluated_result['patient_id']), arr=np.squeeze(evaluated_result['pred']))
            np.save(file=self.args.exp + "/%04d/%s_gt.npy" % (epoch, evaluated_result['patient_id']), arr=np.squeeze(evaluated_result['y']))

            dv = DataVisualizer([np.squeeze(evaluated_result['x']),
                                 np.squeeze(evaluated_result['prior']),
                                 np.squeeze(evaluated_result['bin']),
                                 np.squeeze(evaluated_result['pred']),
                                 np.squeeze(evaluated_result['y'])],
                                 save_path=self.args.exp + "/%04d/%s.png" % (epoch, evaluated_result['patient_id']))
            dv.visualize_np(evaluated_result['x'].shape[2], patch_size=self.args.patch_x)

        target.close()

        return np.mean(np.array(dices)), np.mean(np.array(hds)), np.mean(np.array(asds)), np.mean(np.array(sns)), np.mean(np.array(sps))

