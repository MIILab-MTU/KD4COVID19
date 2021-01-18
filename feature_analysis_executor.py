import argparse
import os
import core.feature_analysis.helperML as hf

from core.feature_analysis.feature_analyzer import FeatureAnalyzer

'''
python3 -W ignore feature_analysis_executor.py --exp=/media/zhaochen/data/covid_seg/exp/seg_reg_cv_3 \
  --feature_pattern=pred \
  --feature_extract_clf_csv=/media/zhaochen/data/covid_seg/data/data_sh_all/classification_for_prediction.csv \
  --feature_coor_threshold=0.83

python3 -W ignore feature_analysis_executor.py --exp=/media/zhaochen/data/covid_seg/exp/seg_reg_cv_3 \
  --feature_pattern=y \
  --feature_extract_clf_csv=/media/zhaochen/data/covid_seg/data/data_sh_all/classification_for_prediction.csv \
  --feature_coor_threshold=0.83

python3 -W ignore feature_analysis_executor.py --exp=/media/zhaochen/data/covid_seg/exp/seg_prior_cv_3 \
  --feature_pattern=pred \
  --feature_extract_clf_csv=/media/zhaochen/data/covid_seg/data/data_sh_all/classification_for_prediction.csv \
  --feature_coor_threshold=0.83

python3 -W ignore feature_analysis_executor.py --exp=/media/zhaochen/data/covid_seg/exp/seg_cv_3 \
  --feature_pattern=pred \
  --feature_extract_clf_csv=/media/zhaochen/data/covid_seg/data/data_sh_all/classification_for_prediction.csv \
  --feature_coor_threshold=0.83

'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default="/media/zhaochen/data/covid_seg/exp/seg_reg_cv_3")
    parser.add_argument('--feature_pattern', type=str, default="y")
    parser.add_argument('--feature_extract_clf_csv', type=str, default="/media/zhaochen/data/covid_seg/data/data_sh_all/classification_for_prediction.csv")
    parser.add_argument('--feature_coor_threshold', type=float, default="0.83")

    args = parser.parse_args()

    feature_analyze_path = os.path.join(args.exp, "feature_analyze")

    fa = FeatureAnalyzer(extracted_csv_path=os.path.join(feature_analyze_path, "extracted_features_{}.csv".format(args.feature_pattern)),
                         pearson_threshold=args.feature_coor_threshold,
                         result_save_path=os.path.join(args.exp, "feature_extraction_{}".format(args.feature_pattern)))
    fa.analyze()
