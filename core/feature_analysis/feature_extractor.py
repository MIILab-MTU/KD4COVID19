import numpy as np
import pandas as pd
import os
import argparse

from glob import glob

import radiomics
import SimpleITK as sitk
import six
from radiomics import featureextractor
from tqdm import tqdm


class FeatureExtractor:

    def __init__(self, data_root, csv_save_path, class_csv_file, pattern):
        self.data_root = data_root
        self.csv_save_path = csv_save_path
        self.pattern = pattern
        self.class_df = pd.read_csv(class_csv_file)

        self.patient_x_files = glob(os.path.join(data_root, "*_x.npy"))
        self.feature_csv_file = open(csv_save_path, "wt")

        self.__init_config__()

    def __init_config__(self):
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('firstorder')
        #extractor.enableFeatureClassByName('shape')
        extractor.enableFeatureClassByName('glcm')
        extractor.enableFeatureClassByName('gldm')
        extractor.enableFeatureClassByName('glrlm')
        extractor.enableFeatureClassByName('glszm')
        extractor.enableFeatureClassByName('ngtdm')

        print('Extraction parameters:\n\t', extractor.settings)
        print('Enabled filters:\n\t', extractor.enabledImagetypes)
        print('Enabled features:\n\t', extractor.enabledFeatures)
        self.extractor = extractor

    def extract_features(self):
        index = 0

        # extract features from predicted numpy arrays
        for patient_x_file in tqdm(self.patient_x_files):
            patient_name = patient_x_file[patient_x_file.rfind("/") + 1: patient_x_file.rfind("_x.npy")]
            print(patient_name)
            data_x = np.load(os.path.join(self.data_root, "{}_x.npy".format(patient_name)))
            data_y = np.load(os.path.join(self.data_root, "{}_{}.npy".format(patient_name, self.pattern)))
            data_y = np.clip(data_y, 0, 1)

            data_x = sitk.GetImageFromArray(data_x)
            data_y = sitk.GetImageFromArray(data_y)

            feature = self.extractor.execute(data_x, data_y, label=1)
            if index == 0:
                index += 1
                keys = []
                keys.append("patient_name")
                for key, value in sorted(six.iteritems(feature)):
                    print('\t', key, ':', value)
                    keys.append(key)
                keys.append("class")

                print("# number of keys {}".format(len(keys)))
                # write csv header
                for idx, key in enumerate(keys):
                    if idx < len(keys) - 1:
                        self.feature_csv_file.write("{},".format(key))
                    else:
                        self.feature_csv_file.write("{}\n".format(key))

            # write features into csv file
            self.feature_csv_file.write("{},".format(patient_name))

            for key, value in sorted(six.iteritems(feature)):
                if isinstance(value, tuple):
                    self.feature_csv_file.write("{}-aaa,".format(str(value[0])))
                    # print(f"write {key}, {str(value[0])}")
                elif isinstance(value, dict):
                    self.feature_csv_file.write("DICT,")
                else:
                    self.feature_csv_file.write("{},".format(value))
                    # print(f"write {key}, {value}")

            row = self.class_df[self.class_df["patient_name"] == patient_name]
            clazz = row.type.values[0]
            self.feature_csv_file.write("{}\n".format(clazz))
            self.feature_csv_file.flush()

        # save extracted features
        self.feature_csv_file.close()

        # remove unused columns
        df = pd.read_csv(self.csv_save_path)
        columns_drop = ["diagnostics_Configuration_EnabledImageTypes", "diagnostics_Configuration_Settings",
                        "diagnostics_Image-original_Dimensionality", "diagnostics_Image-original_Hash",
                        "diagnostics_Image-original_Size", "diagnostics_Image-original_Spacing",
                        "diagnostics_Mask-original_BoundingBox", "diagnostics_Mask-original_CenterOfMass",
                        "diagnostics_Mask-original_CenterOfMassIndex", "diagnostics_Mask-original_Hash",
                        "diagnostics_Mask-original_Size", "diagnostics_Mask-original_Spacing",
                        "diagnostics_Versions_Numpy", "diagnostics_Versions_PyRadiomics",
                        "diagnostics_Versions_PyWavelet", "diagnostics_Versions_Python",
                        "diagnostics_Versions_SimpleITK",
                        "diagnostics_Image-original_Maximum",
                        "diagnostics_Image-original_Mean",
                        "diagnostics_Image-original_Minimum"]

        # save final csv file
        df = df.drop(columns=columns_drop)
        df.to_csv(self.csv_save_path)


'''
python3 feature_extractor.py --data_root=/media/zhaochen/data/covid_seg/exp/seg_reg_cv_3 \
    --feature_extract_clf_csv=/media/zhaochen/data/covid_seg/data/data_sh_all/classification_for_prediction.csv \
    --feature_pattern=pred

python3 feature_extractor.py --data_root=/media/zhaochen/data/covid_seg/exp/seg_reg_cv_3 \
    --feature_extract_clf_csv=/media/zhaochen/data/covid_seg/data/data_sh_all/classification_for_prediction.csv \
    --feature_pattern=y

python3 feature_extractor.py --data_root=/media/zhaochen/data/covid_seg/exp/seg_prior_cv_3 \
    --feature_extract_clf_csv=/media/zhaochen/data/covid_seg/data/data_sh_all/classification_for_prediction.csv \
    --feature_pattern=pred

python3 feature_extractor.py --data_root=/media/zhaochen/data/covid_seg/exp/seg_cv_3 \
    --feature_extract_clf_csv=/media/zhaochen/data/covid_seg/data/data_sh_all/classification_for_prediction.csv \
    --feature_pattern=pred
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default="/media/zhaochen/data/covid_seg/exp/seg_reg_cv_3", type=str)
    parser.add_argument('--feature_extract_clf_csv', type=str, default="/media/zhaochen/data/covid_seg/data/data_sh_all/classification_for_prediction.csv")
    parser.add_argument('--feature_pattern', type=str, default="pred")


    args = parser.parse_args()

    fe = FeatureExtractor(data_root=os.path.join(args.data_root, "feature_analyze"),
                        csv_save_path=os.path.join(args.data_root, "feature_analyze", 
                                                   "extracted_features_{}.csv".format(args.feature_pattern)),
                        class_csv_file=args.feature_extract_clf_csv,
                        pattern=args.feature_pattern)
    fe.extract_features()

    # df_seg_reg_pred = pd.read_csv("/media/zhaochen/data/covid_seg/exp/seg_reg_cv_3/feature_analyze/extracted_features_pred.csv")
    # df_seg_reg_y = pd.read_csv("/media/zhaochen/data/covid_seg/exp/seg_reg_cv_3/feature_analyze/extracted_features_y.csv")
    # df_seg_prior_pred = pd.read_csv("/media/zhaochen/data/covid_seg/exp/seg_prior_cv_3/feature_analyze/extracted_features_pred.csv")
    # df_seg_pred = pd.read_csv("/media/zhaochen/data/covid_seg/exp/seg_cv_3/feature_analyze/extracted_features_pred.csv")

    # print(df_seg_reg_pred.shape)
    # print(df_seg_reg_y.shape)
    # print(df_seg_prior_pred.shape)
    # print(df_seg_pred.shape)


