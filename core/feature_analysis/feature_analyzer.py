from __future__ import print_function
import os
import argparse
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn
import matplotlib.pyplot as plt

import core.feature_analysis.helperML as hf
from ml_covid_clf import *

class FeatureAnalyzer:
    def __init__(self, extracted_csv_path, pearson_threshold, result_save_path):
        self.extracted_csv_path = extracted_csv_path
        self.pearson_threshold = pearson_threshold
        self.result_save_path = result_save_path

    def analyze(self):
        if not os.path.exists(self.result_save_path):
            os.mkdir(self.result_save_path)

        df_ori = pd.read_csv(self.extracted_csv_path, header=0)
        all_cols = df_ori.columns.values.tolist()
        data_x = df_ori[[item for item in all_cols if item not in ['class', 'patient_name', 'Unnamed: 0']]]
        data_y = df_ori['class']

        p005_featureList, p01_featureList = hf.dataFeatureSelection(X=data_x,
                                                                    y=data_y,
                                                                    model='f_classif',
                                                                    write_path=self.result_save_path)
        # Univariate analysis
        print("---------------------------Uni-variate analysis--------------------------------")
        uni_list = p005_featureList

        table = hf.ROC_univariance(uni_list, data_x, data_y, "univariance", self.result_save_path)
        # table_ = table.groupby('AUC').rank(ascending=False)
        table = table.sort_values(by=['AUC'], ascending=False)
        table.to_csv(os.path.join(self.result_save_path, "uni-variance-analysis.csv"))

        data_x = data_x[p005_featureList]
        print("[X] x.shape = {}".format(data_x.shape))
        # pearson analysis to remove highly correlated features

        cor = data_x.astype(float).corr()  # pearson
        features_to_be_remove = []
        for i in range(len(p005_featureList)):
            for j in range(len(p005_featureList)):
                if i > j and np.abs(cor.to_numpy()[i, j]) > self.pearson_threshold:
                    # remove feature
                    if table.ix[i]['AUC'] > table.ix[j]['AUC']:
                        features_to_be_remove.append(j)
                    else:
                        features_to_be_remove.append(i)

        features_to_be_remove = set(features_to_be_remove)
        features_to_be_remove_columns = []
        for e in features_to_be_remove:
            features_to_be_remove_columns.append(p005_featureList[e])

        all_cols = data_x.columns.values.tolist()
        data_x = data_x[[item for item in all_cols if item not in features_to_be_remove_columns]]
        print("[x] data_x after remove highly correlated features, shape = {}".format(data_x.shape))

        cor = data_x.astype(float).corr()

        # plot correlation map
        sns.heatmap(
            cor, linewidths=0.1, square=True, cmap="YlGnBu",
            # mask=mask,
            linecolor='white', annot=False, fmt=".2f", annot_kws={'size': 20}, cbar_kws={"shrink": .5})
        plt.xticks(rotation=60)

        # fix for mpl bug that cuts off top/bottom of seaborn viz
        b, t = plt.ylim()  # discover the values for bottom and top
        b += 0.5  # Add 0.5 to the bottom
        t -= 0.5  # Subtract 0.5 from the top
        plt.ylim(b, t)  # update the ylim(bottom, top) values
        plt.title("correlation")
        plt.savefig(os.path.join(self.result_save_path, 'Figure_Correlation.png'), dpi=200, bbox_inches="tight")
        plt.close()

        print("---------------------------Multivariate analysis---------------------------")
        diy_multivariate_list = data_x.columns.values.tolist()

        # label_list = []
        # for i in range(len(diy_multivariate_list)):
        #     label_list.append("class")
        mva_df = hf.ROC_DIY_multivariance([diy_multivariate_list], ["class"], data_x, data_y, 'multi-variance', self.result_save_path)
        mva_df = mva_df.sort_values(by=['AUC'], ascending=False)
        mva_df.to_csv(os.path.join(self.result_save_path, "multi-variance-analysis.csv"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default="/media/zhaochen/data/covid_seg/exp/seg_reg_cv_3")
    parser.add_argument('--feature_pattern', type=str, default="pred")
    parser.add_argument('--feature_extract_clf_csv', type=str, default="/media/zhaochen/data/covid_seg/data/data_sh_all/classification_for_prediction.csv")
    parser.add_argument('--feature_coor_threshold', type=float, default="0.83")

    args = parser.parse_args()

    feature_analyze_path = os.path.join(args.exp, "feature_analyze")

    fa = FeatureAnalyzer(extracted_csv_path=os.path.join(feature_analyze_path, "extracted_features_{}.csv".format(args.feature_pattern)),
                         pearson_threshold=args.feature_coor_threshold,
                         result_save_path=os.path.join(args.exp, "feature_extraction_{}".format(args.feature_pattern)))
    fa.analyze()