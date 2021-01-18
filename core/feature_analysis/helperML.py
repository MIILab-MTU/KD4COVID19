#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:27:08 2019
Help functions for CRT Machine Learning Project
@author: zhuo
"""
import os
import numpy as np
import pandas as pd

from random import shuffle
from collections import Counter

from sklearn import metrics
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, cross_val_score

from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression, GenericUnivariateSelect


def dataFeatureSelection(X, y, model='f_classif', write_path='./output data/example/'):
    """ Feature selection
    Parameters
    ----------
    X: Dataframe
        All the features
    y: Dataframe
        label
    data_center: string
        Data center
    model: string
        Check the See also part.
    write_pathe: string
        Directary of the figure

    Return
    ------
    p005_featureList: list
        List of the features that the p-value < 0.05
    p01_featureList: list
        List of the features that the p-value < 0.1

    See also
    --------

    f_classif
        ANOVA F-value between label/feature for classification tasks.

    chi2
        Chi-squared stats of non-negative features for classification tasks.

    SelectPercentile
        Select features based on percentile of the highest scores.

    SelectFpr
        Select features based on a false positive rate test.

    SelectFdr
        Select features based on an estimated false discovery rate.

    SelectFwe
        Select features based on family-wise error rate.

    GenericUnivariateSelect
        Univariate feature selector with configurable mode.

    Reference
    ---------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
    """
    sns.set(font_scale=1, style='ticks', color_codes=True)

    # Create and fit selector

    # selector = SelectKBest(f_classif, k=len(X.columns))
    if model == 'chi2':
        selector = SelectKBest(chi2, k=len(X.columns))
    elif model == 'f_classif':
        selector = SelectKBest(f_classif, k=len(X.columns))
    elif model == 'f_regression':
        selector = SelectKBest(f_regression, k=len(X.columns))
    elif model == 'GenericUnivariateSelect':
        selector = SelectKBest(GenericUnivariateSelect, k=len(X.columns))
    else:
        raise ValueError('The features selection model must be one of ???')

    fit = selector.fit(X, y.values.ravel())
    selector_df = pd.DataFrame({'features': X.columns.values.tolist(), 'pvals': fit.pvalues_}).sort_values(by=['pvals'],
                                                                                                           ascending=False)
    y_pos = np.arange(len(selector_df))

    plt.figure(figsize=(8, 16))
    plt.barh(y_pos, selector_df['pvals'], align='center', alpha=0.5)
    # plot p=0.05 line
    plt.axvline(x=0.05, color='r', label='p=0.05')

    for i, v in enumerate(selector_df['pvals']):
        if v < 0.05:
            plt.text(0.05, i, '  ' + str(np.round(v, 2)), color='b', va='center', fontsize=12)
        else:
            plt.text(v, i, '  ' + str(np.round(v, 2)), color='b', va='center', fontsize=12)

    plt.yticks(y_pos, selector_df['features'])
    plt.xlabel('Significance')
    plt.title(' feature selection')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(write_path, 'Figure' + '_featureSelection'), dpi=200, bbox_inches="tight")
    plt.close()

    # Features table
    print('                 ')
    selector_df.sort_values(by=['features'], ascending=True)

    # the feature list that the p<0.05
    p005_featureList = selector_df[selector_df['pvals'].astype(float) <= 0.05]['features'].tolist()
    print('p<0.05 features:', p005_featureList)
    print()

    # the feature list that the p<0.1
    p01_featureList = selector_df[selector_df['pvals'].astype(float) <= 0.1]['features'].tolist()
    print('p<0.1 features:', p01_featureList)
    print()

    return p005_featureList, p01_featureList


def incrementalAssociation(X, y, data_center, feature_list, label_list, line_list,
                           fig_title='', write_path='./results', output='acc'):
    '''
    Incremental Association

    Parameters
    ----------
    X: dataframe
        Features
    y: dataframe
        Label
    data_center: string
        data center: 'IAEA', 'Nanjing', 'Taiwan'
    feature_list: list
        A list of each bar
    label_list: list
        The name of each bar
    line_list: list
        red line list
    fig_title: string
        figure title
    write_path: string
        write directory
    '''

    sns.set(font_scale=1, style='ticks', color_codes=True)

    ytest = y

    acc_list = []
    llf_list = []
    shape_list = []

    for f_list in feature_list:
        shape_list.append(X[f_list].shape[1])
        df_f_list = sm.add_constant(X[f_list])

        model = sm.Logit(y, df_f_list.astype(float))
        result = model.fit(method='bfgs')
        ypred = result.predict()

        if output == 'acc':
            # accuracy
            acc = metrics.accuracy_score(ytest, ypred.round())
            acc_list.append(acc)
        elif output == 'chi':
            chisq, p = stats.chisquare(ypred, ytest)
            acc_list.append(chisq)
        # log likelihood
        llf = result.llf
        llf_list.append(llf)

    sns.set(font_scale=1, style='white', color_codes=True)

    y_max = max(acc_list) + 0.1
    y_min = min(acc_list) - 0.3

    # plot bar figure
    x = range(len(label_list))
    plt.bar(x, acc_list, color='b', alpha=0.6, width=0.5)
    plt.xticks(x, label_list, rotation=60)
    plt.ylim(y_min, y_max)
    for i, v in enumerate(acc_list):
        plt.text(i, v, '{:.02f}'.format(v), color='b', ha='center', va='bottom')

    # the step of the red line
    h_step = (acc_list[0] - y_min) / len(line_list)

    for i, v in enumerate(line_list):
        dof = np.abs(shape_list[v[0]] - shape_list[v[1]])
        lr, p = lrtest(min(llf_list[v[0]], llf_list[v[1]]), max(llf_list[v[0]], llf_list[v[1]]), dof=dof)

        # the location of the red line
        h = acc_list[0] - h_step * i - 0.05

        plt.plot(v, [h, h], color='r')
        plt.text(np.mean(v), h, 'lr={:.02f}, p={:.03f}'.format(lr, p), color='r', ha='center', va='bottom')

    # plt.title('Incremental association' + fig_title)
    plt.grid(axis='y')
    # plt.show()
    plt.savefig(os.path.join(write_path, 'Figure_' + data_center + '_incrementalAssociation' + fig_title + '.png'),
                dpi=200, bbox_inches="tight")
    plt.close()


##########
# MODELS #
##########

import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier

import scipy.stats
from scipy import stats, interp
import statsmodels.api as sm


def chiSquare(X, y):
    """Chi-square test of independence and homogeneity
    Prames
    ------
    feature: String
        Feature to do the t-test
    df: Dataframe
        The original data

    Return
    ------
    chi2: float
        Chi-square score
    p_val: float
        P-value
    """
    obs = pd.crosstab(X, y.to_numpy().reshape((len(X),))).to_numpy()
    chi2, p_val, dof, exp = stats.chi2_contingency(obs)
    return chi2, p_val


def ROC_univariance(col_list, X, y, fig_title, save_dir):
    """
    Get the ROC univariante analysis figure and table

    Parameters
    ----------
    col_list: List
        the list of the columns' name
    X: Dataframe
        Feature data
    y: Dataframe
        Label data
    fig_title: String
        The name of the figure
    save_dir: string
        The directory of the saved figure and table

    Return
    ------
    data_df: Dataframe
        Summary table

    """
    row_list = []
    label_list = []

    for i, col in enumerate(col_list):
        df_feature = sm.add_constant(X[col])
        model = sm.Logit(y, df_feature.astype(float))
        result = model.fit(method='bfgs')
        # print(result.summary())

        ypred = result.predict()
        # if ytest.isin([0,1]).all():
        #     ypred = ypred.astype(int)

        fpr, tpr, threshold = metrics.roc_curve(y, ypred)
        auc_ = metrics.roc_auc_score(y, ypred)
        label = col.replace('SPECT_pre_', '').replace('ECG_pre_', '')
        label_list.append(label)
        plt.plot(fpr, tpr, label=label + ', auc=' + str(np.round(auc_, 3)))

        tn, fp, fn, tp = confusion_matrix(y, ypred.round()).ravel()
        row = []
        threshold = threshold[np.argmax(tpr - fpr)].round(2)
        cutoff = np.subtract(np.max(df_feature[col]), np.min(df_feature[col]), dtype=np.float32) * threshold + np.min(df_feature[col])
        row.append(np.round(cutoff, 1))  # cut point
        row.append(np.round(tp / (tp + fn), 3))  # sensitivity
        row.append(np.round(tn / (tn + fp), 3))  # specificity
        row.append(auc_.round(3))  # AUC
        OR = np.exp(result.params)[1]
        row.append(np.round(OR, 3))  # OR
        conf = result.conf_int()
        row.append([np.round(np.exp(conf).iloc[1, 0], 3), np.round(np.exp(conf).iloc[1, 1], 3)])  # Confidence interval
        row.append(result.pvalues[1].round(4))
        row_list.append(row)

    # plt.title(fig_title)
    plt.legend(loc=4)
    # plt.show()
    plt.savefig(os.path.join(save_dir, 'Figure_Uni_{}.png'.format(fig_title)),
                dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()

    data_df = pd.DataFrame(row_list,
                           columns=['Cut-off', 'Sensitivity', 'Specificity', 'AUC', 'OR', '95% CI', 'P_value'])
    data_df.insert(0, ' ', label_list)

    print('Uni_{}.png'.format(fig_title))
    # print(data_df)

    tab_title = 'Table_Uni_{}.png'.format(fig_title)
    render_mpl_table(data_df, save_dir, tab_title)
    return data_df


def render_mpl_table(data, save_dir, tab_title, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    plt.savefig(os.path.join(save_dir, 'Table_{}.png'.format(tab_title)), dpi=200, bbox_inches="tight")
    plt.close()

    return ax


def ROC_DIY_multivariance(col_list, label_list, X, y, fig_title, save_dir):
    """
    Get the ROC univariante analysis figure and table
    all the features from col_list to the general_col_list one by one

    Parameters
    ----------
    col_list: List
        the list of the columns' name
    name_list: List
        the list of the labels' name
    X: Dataframe
        Feature data
    y: Dataframe
        Label data
    fig_title: String
        The name of the fig
    save_dir: string
        The directory of the saved figure and table

    Return
    ------
    data_df: Dataframe
        Summary table

    """
    row_list = []


    for i, col in enumerate(col_list):
        df_feature = sm.add_constant(X[col])
        # model = sm.Logit(y, df_feature.astype(float))
        model = sm.GLM(y, df_feature.astype(float))
        result = model.fit(method='IRLS')
        # result = model.fit()
        print("Multi-variate analysis results")
        print(result.summary())
        print("------------------------------- CSV file start ------------------------------- ")
        print(result.summary().as_csv())
        csv_file = open(os.path.join(save_dir, "multi-variate_analysis_summary_{}.csv".format(i)), "wt")
        csv_file.write(result.summary().as_csv())
        csv_file.flush()
        csv_file.close()
        print("------------------------------- CSV file end ------------------------------- ")
        print("\nOR: \n", np.exp(result.params))
        print("conf: \n", np.exp(result.conf_int()))

        ypred = result.predict()

        fpr, tpr, threshold = metrics.roc_curve(y, ypred)
        auc_ = metrics.roc_auc_score(y, ypred)
        label = label_list[i] + ', auc=' + str(np.round(auc_, 2))
        plt.plot(fpr, tpr, label=label)

        ypred[ypred > 0.5] = 1
        ypred[ypred <= 0.5] = 0

        tn, fp, fn, tp = confusion_matrix(y, ypred.round()).ravel()
        row = []
        row.append(threshold[np.argmax(tpr - fpr)].round(2))  # cut point
        row.append(np.round(tp / (tp + fn), 2))  # sensitivity
        row.append(np.round(tn / (tn + fp), 2))  # specificity
        row.append(auc_.round(2))  # AUC
        OR = np.exp(result.params)[1]
        row.append(np.round(OR, 3))  # OR
        conf = result.conf_int()
        row.append([np.round(np.exp(conf).iloc[1, 0], 3), np.round(np.exp(conf).iloc[1, 1], 2)])  # Confidence interval
        row.append(result.pvalues[1].round(4))
        row_list.append(row)

    # plt.title(fig_title)
    plt.legend(loc=4)
    # plt.show()
    plt.savefig(os.path.join(save_dir, 'Figure_Multi-AUC_{}.png'.format(fig_title)), dpi=200, bbox_inches="tight")
    plt.close()

    data_df = pd.DataFrame(row_list, columns=['Cut-off', 'Sensitivity', 'Specificity', 'AUC', 'OR', 'CI', 'P'])
    data_df.insert(0, ' ', label_list)

    tab_title = 'Multi-AUC_{}.png'.format(fig_title)
    render_mpl_table(data_df, save_dir, tab_title)
    return data_df


def plot_univariance_ROC(df_feature, col_name, y, data_center):
    df_feature = sm.add_constant(df_feature)
    model = sm.Logit(y, df_feature.astype(float))
    result = model.fit(method='bfgs')
    # print(result.summary())

    ytest = y
    ypred = result.predict()

    fpr, tpr, _ = metrics.roc_curve(ytest, ypred)
    auc = metrics.roc_auc_score(ytest, ypred)
    label = col_name + ', auc=' + str(np.round(auc, 3))
    plt.plot(fpr, tpr, label=label)
    plt.title(data_center)
    plt.legend(loc=4)
    # plt.show()
    plt.close()


def comparison_multiVariate(x_Scar, x_noScar, y, state_str, data_center):
    x_Scar = sm.add_constant(x_Scar)
    model = sm.Logit(y, x_Scar.astype(float))
    result = model.fit(method='bfgs')
    # print(result.summary())

    y_test = y
    y_pred_Scar = result.predict()

    fpr_Scar, tpr_Scar, _ = metrics.roc_curve(y_test, y_pred_Scar)
    auc_Scar = metrics.roc_auc_score(y_test, y_pred_Scar)
    label_Scar = state_str + ', auc=' + str(np.round(auc_Scar, 3))
    plt.plot(fpr_Scar, tpr_Scar, label=label_Scar)

    x_noScar = sm.add_constant(x_noScar)
    model = sm.Logit(y, x_noScar.astype(float))
    result = model.fit()
    # print(result.summary())

    y_pred_noScar = result.predict()

    fpr_noScar, tpr_noScar, _ = metrics.roc_curve(y_test, y_pred_noScar)
    auc_noScar = metrics.roc_auc_score(y_test, y_pred_noScar)
    label_noScar = 'no_' + state_str + ', auc=' + str(np.round(auc_noScar, 3))
    plt.plot(fpr_noScar, tpr_noScar, label=label_noScar)
    plt.title(data_center)
    plt.legend(loc=4)
    plt.show()


def nprange(x, axis=0):
    return np.max(x, axis=axis) - np.min(x, axis=axis)


def plot_all_X(X, y, y_col, features, cols=2, X_val=None, y_val=None):
    """Multi-variate analysis
    Do a multi-variate analysis of
    Parameters
    ----------
    X: dataframe
        The original X_train data with every feature
    y: dataframe
        The original y_train data, labels
    y_col: string
        the name of the ylabel that will be shown on the figure
    features: list
        The features' columns of X
    cols: int, default 2
        The number of the columns of the subplots
    X_val: dataframe, default None
        validation set X
    y_val: dataframe, default None
        validation set y

    Explain
    -------
    Red line: x^2
    Blue line: x^3
    Green line: x^4

    Mean absolute error (MAE) to see if benefit gained from highter degree polonomials came from overfitting or actual improvements
    """
    plot_lm_2 = plt.figure(2)
    plot_lm_2.set_figheight(12)
    plot_lm_2.set_figwidth(12)

    flen = len(features)
    # cols = 2
    rows = int(np.ceil(flen / cols))

    # Will hold the R^2 values for LR and PR
    # R^2 is for training set, MAE is for test set
    r2scores = []
    maescores = []

    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 5, rows * 3), \
                            subplot_kw={'xticks': [], 'yticks': []})

    fig.subplots_adjust(left=0.03, right=0.97, hspace=0.3, wspace=0.05)

    for ax, interp_method in zip(axs.flat, features):
        ax.set_title(str(interp_method))
        ax.set_xlabel(str(interp_method))
        ax.set_ylabel(y_col)
        ax.scatter(X[interp_method], y)

        xrng = nprange(X[interp_method])
        yrng = nprange(y)
        xmin = X[interp_method].min(axis=0)
        xmax = X[interp_method].max(axis=0)
        ymin = np.min(y)
        ymax = np.max(y)

        ax.axis([xmin, xmax, ymin, ymax])

        xticks = [xmin, xmin + (0.5 * xrng), xmax]
        yticks = [ymin, ymin + (0.5 * yrng), ymax]
        ax.set_xticks(xticks, minor=False)
        ax.set_yticks(yticks, minor=False)

        # Sort X values for non-linear graphing
        X_sorted = X[interp_method].sort_values()

        # Record the R^2 for fitting (1) LR, (2) PR (X**2), (3) PR (X**3)
        # Linear Regression
        lr_model = LinearRegression()
        lr_fit = lr_model.fit(X[[interp_method]], y)
        lr_predict = lr_model.predict(X[[interp_method]])

        # best fit curve :
        # p2 (X**2) in red
        z2 = np.polyfit(X[interp_method], y, 2)
        p2 = np.poly1d(z2)
        ax.plot(X_sorted, p2(X_sorted), 'r-')
        # p3 (X**3) in blue
        z3 = np.polyfit(X[interp_method], y, 3)
        p3 = np.poly1d(z3)
        ax.plot(X_sorted, p3(X_sorted), 'b-')
        # p4 (X**3) in green
        z4 = np.polyfit(X[interp_method], y, 4)
        p4 = np.poly1d(z4)
        ax.plot(X_sorted, p4(X_sorted), 'g-')

        r2scores.append([interp_method, "%.03f" % (r2_score(y, lr_predict)),
                         "%.03f" % (r2_score(y, p2(X[interp_method]))),
                         "%.03f" % (r2_score(y, p3(X[interp_method]))),
                         "%.03f" % (r2_score(y, p4(X[interp_method])))])
        if X_val is not None and y_val is not None:
            lr_eval = lr_model.predict(X_val[[interp_method]])
            maescores.append([interp_method, "%.03f" % (mean_absolute_error(y_val, lr_eval)),
                              "%.03f" % (mean_absolute_error(y_val, p2(X_val[interp_method]))),
                              "%.03f" % (mean_absolute_error(y_val, p3(X_val[interp_method]))),
                              "%.03f" % (mean_absolute_error(y_val, p4(X_val[interp_method])))])

    plt.tight_layout()
    plt.show()

    for i in range(len(r2scores)):
        print("\n" + r2scores[i][0] + " Linear R^2: %s, X^2 R^2: %s, X^3 R^2, %s, X^4 R^2, %s" % (r2scores[i][1],
                                                                                                  r2scores[i][2],
                                                                                                  r2scores[i][3],
                                                                                                  r2scores[i][4]))
        if X_val is not None and y_val is not None:
            print("(Eval) " + maescores[i][0] + " Linear mae: %s, X^2 mae: %s, X^3 mae, %s, X^4 mae, %s" % (
            maescores[i][1],
            maescores[i][2], maescores[i][3], maescores[i][4]))


def lrtest(llmin, llmax, dof):
    """Likelihood ratio test
    Parameters
    ----------
    llmin: float
        the min of the log likelihood
    llmax: float
        the max of the log likelihood
    dof: int
        degree of freedom

    Return
    ------
    lr: float
        likelihood ratio
    p:  float
        p-value
    """
    lr = 2 * (llmax - llmin)
    p = stats.chi2.sf(lr, dof)  # llmax has dof more than llmin
    return lr, p


####################
# MACHINE LEARNING #
####################
from imblearn import over_sampling
from sklearn.ensemble import ExtraTreesClassifier


def inforGain(X, y, write_path='./results'):
    """ featureImportance
    Parameters
    ----------
    X: Dataframe
        All the features
    y: Dataframe
        label
    write_pathe: string
        Directary of the figure

    Return
    ------
    etc_df: dataframe
        ranking data frame
    """
    from sklearn.tree import DecisionTreeClassifier

    sns.set(font_scale=1, style='ticks', color_codes=True)

    print('Information Gain feature selection:')

    # Create and fit model
    # feature extraction
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(X, y.values.ravel())

    etc_df = pd.DataFrame({
        'features': X.columns.values.tolist(),
        'scores': model.feature_importances_}).sort_values(by=['scores'], ascending=False)

    # append ranking index
    rank = np.arange(1, len(X.columns) + 1)
    etc_df['ranking'] = rank
    etc_df = etc_df.reset_index(drop=True)

    return etc_df


def univariateSelection(X, y, model='f_classif', write_path='./results'):
    """ univariateSelection
    Parameters
    ----------
    X: Dataframe
        All the features
    y: Dataframe
        label
    model: string
        Check the See also part.
    write_pathe: string
        Directary of the figure

    Return
    ------
    p005_featureList: list
        List of the features that the p-value < 0.05
    p01_featureList: list
        List of the features that the p-value < 0.1

    Figure and table(cols_name, p, score)

    Reference
    ---------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
    """
    sns.set(font_scale=1, style='ticks', color_codes=True)

    print('Univariate feature selection:')
    # Create and fit selector
    selector = SelectKBest(f_classif, k=len(X.columns))

    fit = selector.fit(X, y.values.ravel())
    selector_df = pd.DataFrame({
        'features': X.columns.values.tolist(),
        'pvals': fit.pvalues_}).sort_values(by=['pvals'], ascending=True)

    # append ranking index
    rank = np.arange(1, len(X.columns) + 1)
    selector_df['ranking'] = rank
    selector_df = selector_df.reset_index(drop=True)

    return selector_df


def featureImportance(X, y, write_path='./results'):
    """ featureImportance
    Parameters
    ----------
    X: Dataframe
        All the features
    y: Dataframe
        label
    write_pathe: string
        Directary of the figure

    Return
    ------
    etc_df: dataframe
        ranking data frame
    """
    sns.set(font_scale=1, style='ticks', color_codes=True)

    print('Tree-based feature selection:')

    # Create and fit model
    # feature extraction
    model = ExtraTreesClassifier(n_estimators=10)
    model.fit(X, y.values.ravel())

    etc_df = pd.DataFrame({
        'features': X.columns.values.tolist(),
        'scores': model.feature_importances_}).sort_values(by=['scores'], ascending=False)

    # append ranking index
    rank = np.arange(1, len(X.columns) + 1)
    etc_df['ranking'] = rank
    etc_df = etc_df.reset_index(drop=True)

    return etc_df


def featureSelection(method='info', X=None, y=None, write_path='./results'):
    '''Feature selection method
    Parameters
    ----------
    method: string
        feature selection method
    X: dataframe
        All the features
    y: Dataframe
        label
    write_pathe: string
        Directary of the figure

    Return
    ------
    df_rank: dataframe
        dataframe of the ranking results
    '''
    selectN = 30
    type_list = []

    if method == 'info':
        df_rank = inforGain(X, y, write_path=write_path)
        fig_title = 'Figure_featureSelection_informationGain'
    elif method == 'uni':
        df_rank = univariateSelection(X, y)
        fig_title = 'Figure_featureSelection_Univariante'
    elif method == 'tree':
        df_rank = featureImportance(X, y)
        fig_title = 'Figure_featureSelection_tree-based'
    else:
        raise Exception('Feature selection method error!')
    df_rank.columns = ["features", "scores", "ranking"]

    for idx, row in df_rank.iterrows():
        # Add label columns
        if 'ECG' in row['features']:
            type_list.append('ECG')
        elif 'SPECT' in row['features']:
            if 'PSD' in row['features'] or 'PBW' in row['features']:
                type_list.append('Phase')
            else:
                type_list.append('SPECT')
        elif 'Echo' in row['features']:
            type_list.append('Echo')
        else:
            type_list.append('Clinic')

    df_rank['Type'] = type_list

    # Set all variables
    feature_name = df_rank.iloc[:selectN, 0].iloc[::-1]  # Feature name: 1st column
    scores = df_rank.iloc[:selectN, 1].iloc[::-1]  # Scores: 2nd column
    rank = df_rank.iloc[:selectN, 2].iloc[::-1]  # Rank: 3rd column
    type_ = df_rank.iloc[:selectN, 3].iloc[::-1]  # Type: 4th column
    num = np.arange(len(rank))

    color = type_.to_list()
    color = [c.replace('ECG', 'r').replace('Phase', 'y') for c in color]
    color = [c.replace('SPECT', 'b').replace('Echo', 'g') for c in color]
    color = [c.replace('Clinic', 'c') for c in color]

    plot_barh_fig(num, scores, feature_name, color, fig_title='',
                  file_name=fig_title, file_dir=write_path)

    return df_rank


def plot_barh_fig(y, width, y_name, color, fig_title, file_name, file_dir):
    '''Plot barh figure with different colors

    '''

    sns.set(font_scale=1.5, style='ticks', color_codes=True)

    plt.figure(figsize=(8, 16))
    plt.barh(y, width, align='center', alpha=0.5, color=color)
    for i, v in enumerate(width):
        plt.text(v, i, '  ' + str(np.round(v, 4)), color='k', va='center', fontsize=12)
    plt.yticks(y, y_name)
    # plt.xlabel(xlabel)
    plt.title(fig_title)
    # plt.legend(loc='upper right')
    min_ = np.min(width)
    max_ = np.max(width)
    plt.xlim([min_, max_ + (max_ - min_) / 5])
    # plt.ylim([-3, 3])

    bwith = 3
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.xaxis.set_ticks_position('top')

    plt.savefig(os.path.join(file_dir, file_name),
                dpi=200, bbox_inches="tight")
    plt.show()


def ML_overSamplingTrainningSet(X, y, ML_model, overSamp_model, score='accuracy', write_dir='./resutls',
                                fig_title='ROC', random_state=0, n_splits=5):
    """ Machine learning
    Parameters
        ----------
        X: Dataframe
            Features
        y: Dataframe
            Labels
        ML_model: String
            "svm" : SVM
            "rf" : Random forest
            "Ada" : AdaBoost
        overSamp_model: String
            Over sampling model
            "random" : random over sampling
            "smote" : SMOTE
            "adasyn" : ADASYN
        score: string, default 'accuracy'
            Defining model evaluation rules.
            For example: 'accuracy', 'recall', 'f1', 'f1_micro', 'f1_macro', 'roc_auc'
            For more details, please check the reference
        fig_title: string, default 'ROC'
            The title of the ROC figure

        Reference
        ---------
        [1] https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html
        [2] https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    """
    sns.set(style='ticks', color_codes=True, font_scale=1)

    # Set over sampling method
    if overSamp_model == 'random':
        overSamp = over_sampling.RandomOverSampler(sampling_strategy='all', random_state=random_state)
    elif overSamp_model == 'smote':
        overSamp = over_sampling.SMOTE(sampling_strategy='all', random_state=random_state)
    elif overSamp_model == 'adasyn':
        overSamp = over_sampling.ADASYN(sampling_strategy='all', random_state=random_state)
    elif overSamp_model == 'kmeans_smote':
        overSamp = over_sampling.KMeansSMOTE(sampling_strategy='all', random_state=random_state)
    else:
        raise Warning("The input model is not valid!")

    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=1 / n_splits, random_state=random_state)
    fig_title = str(fig_title).replace('\', \'', ' + ').replace('SPECT_pre_', '').replace('ECG_pre_', '')
    fig_title = fig_title.replace('\'', '').replace('[', '').replace(']', '').replace('\\', '')

    # Set different tuned_parameters/clf for different ML models
    if ML_model == "svm":
        tuned_parameters = {
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'C': np.logspace(-2, 10, 13),
            'gamma': np.logspace(-9, 3, 13),
        }
        clf = GridSearchCV(SVC(max_iter=10000), tuned_parameters, scoring=score, cv=cv,
                           iid=False, n_jobs=-1)

    elif ML_model == "rf":
        tuned_parameters = {
            # 'bootstrap': [True, False],
            'n_estimators': [1, 2, 4, 8],
            'max_depth': np.linspace(1, 16, 16, endpoint=True),
            'min_samples_split': np.linspace(0.1, 1.0, 10, endpoint=True),
            'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True),
            'max_features': ['log2', 'auto', 'sqrt'],
        }
        clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, scoring=score, cv=cv,
                           iid=False, n_jobs=-1)
    elif ML_model == "Ada":
        tuned_parameters = {
            'n_estimators': range(10, 100, 10),
            'learning_rate': np.logspace(-3, 10, 5),
            # 'algorithm': ['SAMME', 'SAMME.R']
        }
        clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters, scoring=score, cv=cv,
                           iid=False, n_jobs=-1)
    else:
        raise Warning('Please input effective model name')

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    checkDict = {}

    report_df = pd.DataFrame(columns=['acc', 'prec', 'recall', 'f1', 'sensitivity', 'specificity',
                                      'sensitivity_train', 'specificity_train'])

    fig, ax = plt.subplots()

    for i, (train, test) in enumerate(cv.split(X, y)):
        print('Start fold {}'.format(i))
        X_train = X.iloc[train]
        X_train.drop(columns=['ID'], inplace=True)
        y_train = np.ravel(y.iloc[train])

        X_test = X.iloc[test]
        y_ID = X_test['ID'].values
        X_test.drop(columns=['ID'], inplace=True)
        y_test = np.ravel(y.iloc[test])

        print('Oversampling #{}'.format(i))
        X_resampled, y_resampled = overSamp.fit_resample(X_train, y_train)

        # Train model by the best parameters
        print('Training #{} ...'.format(i))
        clf.fit(X_resampled, y_resampled)

        # Call predict on the estimator with the best found parameters.
        y_true, y_pred = y_test, clf.predict(X_test)
        y_train_true, y_train_pred = y_train, clf.predict(X_train)
        col_list = X_test.columns.values.tolist()
        d = {
            'ID_{}'.format(i): y_ID,
            'y_true_{}'.format(i): y_true,
            'y_pred_{}'.format(i): y_pred
        }
        for j in range(len(col_list)):
            d.update({
                '{}_{}'.format(col_list[j], i): X_test[col_list[j]].values,
            })
        checkDict.update(d)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn)  # sensitivity
        spec = tn / (tn + fp)  # specificity

        tn1, fp1, fn1, tp1 = confusion_matrix(y_train_true, y_train_pred).ravel()
        sens1 = tp1 / (tp1 + fn1)  # sensitivity
        spec1 = tn1 / (tn1 + fp1)  # specificity

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        report_df = report_df.append({
            'acc': np.round(acc, 3),
            'prec': np.round(prec, 3),
            'recall': np.round(recall, 3),
            'f1': np.round(f1, 3),
            'sensitivity': np.round(sens, 3),
            'specificity': np.round(spec, 3),
            'sensitivity_train': np.round(sens1, 3),
            'specificity_train': np.round(spec1, 3),
        }, ignore_index=True)

        # # Matrix figure
        # ax_ = plt.axes()
        # y_tick_labels = ['Actual 0', 'Actual 1']
        # x_tick_labels = ['Predicted 0', 'Predicted 1']
        # ax_.set_title('Confusion Matrix (1 = response, 0= no response)', fontsize=12)
        # sns.heatmap(confusion_matrix(y_true, y_pred), linewidths=0.1, square=True,
        #             cmap="YlGnBu", annot=True, xticklabels = x_tick_labels,
        #             yticklabels = y_tick_labels, ax=ax_)

        # # fix for mpl bug that cuts off top/bottom of seaborn viz
        # b, t = plt.ylim() # discover the values for bottom and top
        # b += 0.5 # Add 0.5 to the bottom
        # t -= 0.5 # Subtract 0.5 from the top
        # plt.ylim(b, t) # update the ylim(bottom, top) values
        # plt.show()

        # ROC
        viz = metrics.plot_roc_curve(clf, X_test, y_test,
                                     name='ROC fold {}'.format(i),
                                     alpha=0.3, lw=1, ax=ax)
        interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        print()

    print(report_df)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=fig_title)
    ax.legend(loc="lower right", prop={'size': 'small'})
    # plt.show(fig)
    plt.savefig(os.path.join(write_dir, '{}.png'.format(fig_title)), dpi=200, bbox_inches="tight")
    plt.show()

    render_mpl_table(report_df, save_dir=write_dir, tab_title='Table_ROCreport')

    pd.DataFrame(data=checkDict).to_csv(os.path.join(write_dir, 'check_outliers.csv'))
    return checkDict


def ML_with_overSampling(X, y, ML_model, overSamp_model, score='accuracy', write_dir='./results',
                         fig_title='ROC', random_state=0):
    """ Simple SVM modell with over sampling

        Parameters
        ----------
        X: Dataframe
            Features
        y: Dataframe
            Labels
        ML_model: String
            "svm" : SVM
            "rf" : Random forest
            "Ada" : AdaBoost
        overSamp_model: String
            Over sampling model
            "random" : random over sampling
            "smote" : SMOTE
            "adasyn" : ADASYN
        score: string, default 'accuracy'
            Defining model evaluation rules.
            For example: 'accuracy', 'recall', 'f1', 'f1_micro', 'f1_macro', 'roc_auc'
            For more details, please check the reference
        fig_title: string, default 'ROC'
            The title of the ROC figure

        Reference
        ---------
        [1] https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html
        [2] https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    """
    if overSamp_model == 'random':
        overSamp = over_sampling.RandomOverSampler(sampling_strategy='all', random_state=random_state)
    elif overSamp_model == 'smote':
        overSamp = over_sampling.SMOTE(sampling_strategy='all', random_state=random_state)
    elif overSamp_model == 'adasyn':
        overSamp = over_sampling.ADASYN(sampling_strategy='all', random_state=random_state)
    elif overSamp_model == 'kmeans_smote':
        overSamp = over_sampling.KMeansSMOTE(sampling_strategy='all', random_state=random_state)
    else:
        raise Warning("The input model is not valid!")

    X_resampled, y_resampled = overSamp.fit_resample(X, y)

    ind_list = [i for i in range(len(y_resampled))]
    shuffle(ind_list)
    X_shuffled = X_resampled.iloc[ind_list, :]
    y_shuffled = y_resampled.iloc[ind_list,]

    fig_title = str(fig_title).replace('\', \'', ' + ').replace('SPECT_pre_', '').replace('ECG_pre_', '')
    fig_title = fig_title.replace('\'', '').replace('[', '').replace(']', '').replace('\\', '')

    if ML_model == "svm":
        tuned_parameters = {
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'C': np.logspace(-2, 10, 13),
            'gamma': np.logspace(-9, 3, 13),
        }
        training_simple_model(model_name='SVM', tp=tuned_parameters,
                              X=X_shuffled, y=y_shuffled, score=score, fig_title=fig_title)

    elif ML_model == "rf":
        tuned_parameters = {
            'bootstrap': [True, False],
            'n_estimators': range(10, 100, 10),
            'min_samples_split': range(2, 10, 2),
            # 'min_samples_leaf': [1, 2, 4],
            'max_features': ['log2', 'auto', 'sqrt'],
        }
        training_simple_model(model_name='RF', tp=tuned_parameters,
                              X=X_shuffled, y=y_shuffled, score=score, write_dir=write_dir, fig_title=fig_title)
    elif ML_model == "Ada":
        tuned_parameters = {
            'n_estimators': range(10, 100, 10),
            'learning_rate': np.logspace(-3, 10, 5),
            # 'algorithm': ['SAMME', 'SAMME.R']
        }
        training_simple_model(model_name='Ada', tp=tuned_parameters,
                              X=X_shuffled, y=y_shuffled, score=score, write_dir=write_dir, fig_title=fig_title)
    else:
        raise Warning("The inputed machine learning model is not valid!")


def training_simple_model(model_name, tp, X, y, score='accuracy', write_dir='./resutls',
                          fig_title='ROC', random_state=0):
    """simple RF model
    5-fold cross validation to find the best parameters based on the whole dastaset
    5-fold cross validation to get the mean of the results based on the whole dataset

    Parameters
    ----------
    model_name: string
        model name: 'SVM', 'RF'
    tp: dictionary
        tuned parameters
    X: dataframe
        data's features
    y: dataframe
        data's label
    score: string, default 'accuracy'
        Defining model evaluation rules
        For more details, please check the reference
    fig_title: string, default ROC
            The title of the ROC figure

    Reference
    ---------
    https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    """
    report_df = pd.DataFrame(columns=['acc', 'prec', 'recall', 'f1'])

    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=random_state)

    print("# Tuning hyper-parameters for ", score)
    print()

    if model_name == 'RF':
        clf = GridSearchCV(RandomForestClassifier(), tp,
                           scoring=score, cv=cv, iid=False)
    elif model_name == 'SVM':
        clf = GridSearchCV(SVC(max_iter=10000), tp,
                           scoring=score, cv=cv, iid=False)
    elif model_name == 'Ada':
        clf = GridSearchCV(AdaBoostClassifier(), tp,
                           scoring=score, cv=cv, iid=False)
    else:
        raise Warning('Please input effective model name')

    clf.fit(X, y)

    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print()

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()

    for i, (train, test) in enumerate(cv.split(X, y)):
        X_train = X.iloc[train]
        y_train = np.ravel(y.iloc[train])
        X_test = X.iloc[test]
        y_test = np.ravel(y.iloc[test])

        # Train model by the best parameters
        clf.fit(X_train, y_train)

        # Results of each fold
        y_true, y_pred = y_test, clf.predict(X_test)
        # print(classification_report(y_true, y_pred))
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        report_df = report_df.append({'acc': acc, 'prec': prec, 'recall': recall, 'f1': f1},
                                     ignore_index=True)

        # # Matrix figure
        # ax_ = plt.axes()
        # y_tick_labels = ['Actual 0', 'Actual 1']
        # x_tick_labels = ['Predicted 0', 'Predicted 1']
        # ax_.set_title('Confusion Matrix (1 = response, 0= no response)', fontsize=12)
        # sns.heatmap(confusion_matrix(y_true, y_pred), linewidths=0.1, square=True,
        #             cmap="YlGnBu", annot=True, xticklabels = x_tick_labels,
        #             yticklabels = y_tick_labels, ax=ax_)

        # # fix for mpl bug that cuts off top/bottom of seaborn viz
        # b, t = plt.ylim() # discover the values for bottom and top
        # b += 0.5 # Add 0.5 to the bottom
        # t -= 0.5 # Subtract 0.5 from the top
        # plt.ylim(b, t) # update the ylim(bottom, top) values
        # plt.show()

        # ROC
        viz = metrics.plot_roc_curve(clf, X_test, y_test,
                                     name='ROC fold {}'.format(i),
                                     alpha=0.3, lw=1, ax=ax)
        interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        print()

    print(report_df)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=fig_title)
    ax.legend(loc="lower right", prop={'size': 'small'})
    # plt.show(fig)
    plt.savefig(os.path.join(write_dir, '{}.png'.format(fig_title)), dpi=200, bbox_inches="tight")
    plt.show()


########
# PLOT #
########

import six


def plot_df_table(data, col_width=3.0, row_height=0.625, font_size=14,
                  header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                  bbox=[0, 0, 1, 1], header_columns=0,
                  ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax


####################
# TEXTURE ANALYSIS #
####################

from sklearn.decomposition import PCA


def merge_df(df1, df2, key_str, left_suf, right_suf):
    '''
    Merge two dataframe based on the same key words
    '''
    df_cb = pd.merge(df1, df2, sort=key_str, on=key_str, how='outer',
                     suffixes=(left_suf, right_suf))
    df_dupli = df_cb[df_cb.duplicated([key_str], keep=False)]
    print('duplicated rows\' size: ', df_dupli.shape[0])
    print('combined df with duplicated rows shape: ', df_cb.shape)
    df_cb = df_cb.drop_duplicates([key_str], keep='first')
    print('combined df without duplicated rows shape: ', df_cb.shape)
    return df_cb


def PCA_featureSelection(df, num_components, txtFeature_cols):
    # PCA - Nonalization

    # Normalize by standard scaler
    df[txtFeature_cols] = preprocessing.StandardScaler().fit_transform(df[txtFeature_cols])

    # boxplot = df[num_cols].boxplot(figsize=(16,8))
    # boxplot.set_xticklabels(df[num_cols].columns.values, rotation=60, fontdict={"fontsize":12})

    # PCA
    # Reference: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    # https://imbalanced-learn.readthedocs.io/en/stable/auto_examples/over-sampling/plot_comparison_over_sampling.html#sphx-glr-auto-examples-over-sampling-plot-comparison-over-sampling-py
    # Feature extraction
    pca = PCA(n_components=3)
    # hoose the minimum number of principal components
    # such that 95% of the variance is retained.
    # pca = PCA(.95)
    fit = pca.fit(df[txtFeature_cols])  # fitting PCA
    # summarize components
    print("Explained Variance: %s" % fit.explained_variance_ratio_)
    # print(fit.components_)

    pca_txtFeatures = pca.transform(df[txtFeature_cols])
    df['PCA_0'] = pca_txtFeatures[:, 0]
    df['PCA_1'] = pca_txtFeatures[:, 1]
    df['PCA_2'] = pca_txtFeatures[:, 2]
    pca_cols = ['PCA_0', 'PCA_1', 'PCA_2']
    return df, pca_cols


##################
# READ .MAT FILE #
##################

def read_mat(dir_mat_file):
    """Read the polarmaps from .MAT file
    Parames
    -------
    dir_mat_file: string
        The directory of the .MAT file
    return
    ------
    df_polarMap: dataframe
        The dataframe contaning the polarmaps of ID, contraction_phase_polarmap,
        brightening_polarmap, and perfusion_polarmap.
    """

    # mat_path = '../output data/polarmaps/polarMaps_pre/'
    mat_files = []
    mat_files = [x for x in os.listdir(dir_mat_file) if x.endswith(".mat")]
    print('The number of the mat_files: ', len(mat_files))

    # all patients
    polarMap_list = []
    for i in mat_files:
        mat = scipy.io.loadmat(dir_mat_file + i)
        patient_i = []
        patient_i.append(mat['ECTBdata']['ID'][0][0][0])
        patient_i.append(mat['ECTBdata']['contraction_phase_polarmap'][0][0])
        patient_i.append(mat['ECTBdata']['brightening_polarmap'][0][0])
        patient_i.append(mat['ECTBdata']['perfusion_polarmap'][0][0])
        polarMap_list.append(patient_i)

    df_polarMap = pd.DataFrame(polarMap_list, columns=['ID', 'phase_polarmap',
                                                       'brightening_polarmap',
                                                       #                                                   'perfusion_polarmap',
                                                       'perfusion_polarmap'])
    print(df_polarMap.shape)
    print('Should be 353')

    df_polarMap = df_polarMap.sort_values(by=['ID'])
    return df_polarMap


def ROC_cv_multivariance(col_list, label_list, X, y, fig_title, save_dir, cv_n):
    """
    Get the ROC univariante analysis figure and table
    all the features from col_list to the general_col_list one by one

    Parameters
    ----------
    col_list: List
        the list of the columns' name
    name_list: List
        the list of the labels' name
    X: Dataframe
        Feature data
    y: Dataframe
        Label data
    fig_title: String
        The name of the fig
    save_dir: string
        The directory of the saved figure and table

    Return
    ------
    data_df: Dataframe
        Summary table

    """

    sns.set(font_scale=1, style='ticks', color_codes=True)
    cv = StratifiedShuffleSplit(n_splits=cv_n, test_size=1 / cv_n, random_state=0)

    row_list = []

    for i, col in enumerate(col_list):
        data = sm.add_constant(X[col])
        auc_arr = np.array([])
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        sens_cv = []
        spec_cv = []
        acc_cv = []

        # cv
        for j, (train, test) in enumerate(cv.split(data, y)):
            X_train = data.iloc[train]
            y_train = np.ravel(y.iloc[train])

            X_test = data.iloc[test]
            y_test = np.ravel(y.iloc[test])

            # sm calculate
            model = sm.Logit(y_train, X_train.astype(float))
            result = model.fit(method='bfgs')

            y_pred = result.predict(X_test)

            # confusion_matrix , sensitivity, specificity, and accurancy
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            sens = tp / (tp + fn)  # sensitivity
            spec = tn / (tn + fp)  # specificity
            acc = accuracy_score(y_test, y_pred)
            sens_cv.append(sens)
            spec_cv.append(spec)
            acc_cv.append(acc)

            # tpr, fpr for ROC curve
            fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
            auc = metrics.roc_auc_score(y_test, y_pred)
            auc_arr = np.append(auc_arr, auc)

            interp_tpr = interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            # aucs.append(auc)

            print()

        # calculate mean_tpr, mean_fpr for each y label
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        mean_auc = np.mean(auc_arr)
        std_auc = np.std(auc_arr)

        label = r'%s Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (label_list[i], np.round(mean_auc, 2), np.round(std_auc, 2))
        plt.plot(mean_fpr, mean_tpr, label=label)

        # get the statistical table
        row = []

        row.append(np.round(np.mean(sens_cv), 2))  # sensitivity
        row.append(np.round(np.mean(spec_cv), 2))  # specificity
        row.append(np.round(np.mean(acc_cv), 2))  # Accurancy
        row_list.append(row)

    plt.title('ROC curve of {}'.format(fig_title))
    plt.legend(loc=4)
    plt.savefig(os.path.join(save_dir, 'Figure_{}.png'.format(fig_title)), dpi=200, bbox_inches="tight")
    plt.show()

    data_df = pd.DataFrame(row_list, columns=['Sensitivity', 'Specificity', 'Accurancy'])
    data_df.insert(0, ' ', label_list)

    tab_title = 'Table_{}.png'.format(fig_title)
    render_mpl_table(data_df, save_dir, tab_title)
    return data_df
