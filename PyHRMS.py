import pandas as pd
import matplotlib.pyplot as plt
import time
import pymzml
import scipy.signal
from numpy import *
from numba import jit
from numba.typed import List
from glob import glob
import numpy as np
import scipy.interpolate as interpolate
import multiprocessing as mp
from molmass import Formula
import os
import json
import sys


# %config InlineBackend.figure_format = 'retina'


def peak_picking(df1, ms_error=50, threshold=15):
    '''
    Perform peak picking for a whole LC-MS file, and return the result.
    :param df1: LC-MS dataframe, genrated by the function gen_df()
    :param ms_error: The ms difference between two selected masses (for extraction), this parameter may not affect the final result, but 50 is recommended.
    :param threshold: This parameter is used for the function of peak_finding(eic, threhold)
    :return:
    '''
    index = ms_locator(df1, ms_error)  ### 获得ms locator
    start_t = time.time()
    RT = np.array(df1.columns)
    l = len(index)
    num = 0
    num_p = 0
    for i in range(l - 1):
        df2 = df1.iloc[index[i]:index[i + 1]]
        a = np.array(df2).T  ### 将dataframe转换成np.array
        if len(a[0]) != 0:  ### 判断切片结果是否为0
            extract_c = a.sum(axis=1)
            peak_index, left, right = peak_finding(extract_c, threshold)  ## 关键函数，峰提取
            if len(peak_index) != 0:  ### 判断是否找到峰
                df3 = df2[df2.columns[peak_index]]
                rt = np.round(RT[peak_index], 2)
                intensity = np.round(np.array(df3.max().values), 0)
                mz = np.round(np.array(df3.idxmax().values), 4)
                name = 'peak' + str(num_p)
                locals()[name] = np.array([rt, mz, intensity]).T
                num_p += 1
        p = round(num / l * 100, 1)
        print(f' \r finding peaks...{p}%                   ', end='')  ### 可以切换成百分比
        num += 1
    data = []
    for i in range(num_p - 1):
        name = 'peak' + str(i)
        data.append(locals()[name])
    peak_info = np.concatenate(data)
    peak_info_df = pd.DataFrame(data=peak_info, columns=['rt', 'mz', 'intensity'])
    return peak_info_df


def ms_locator(df1, ppm=50):
    '''
    For pick picking, selecting a series of mass locators for 50-1000.
    :param df1: LC-MS dataframe, genrated by the function gen_df()
    :param ppm: the mass difference between two locators
    :return: mass locators
    '''
    @jit(nopython=True)
    def find_locator(list1, error):
        locators = []
        locator = list1[0]
        for i in range(len(list1)):
            if list1[i] > locator:
                locators.append(i)
                locator *= (1 + error * 1e-6)
        return locators

    ms_list = list(df1.index)
    typed_a = List()
    [typed_a.append(x) for x in ms_list]
    locators = find_locator(typed_a, ppm)
    return locators


def sep_scans(path, company):
    '''
    To separate scan for MS1, MS2 and lockspray. Only supported for Waters .raw and Agilent .d file
    :param path: The path for mzML files
    :return: ms1, ms2 and lockspray
    '''
    if company == 'Waters':
        a = time.time()
        print('\r Reading files...             ', end="")
        run = pymzml.run.Reader(path)
        ms1, ms2 = [], []
        lockspray = []
        for scan in run:
            if scan.id_dict['function'] == 1:
                ms1.append(scan)
            if scan.id_dict['function'] == 2:
                ms2.append(scan)
            if scan.id_dict['function'] == 3:
                lockspray.append(scan)
        b = time.time()
        time1 = round(b - a, 2)
        print(f'\r Reading files finished! Total time: {time1} s           ', end='')
        return ms1, ms2, lockspray
    elif company == 'Agilent':
        a = time.time()
        print('\r Reading files...             ', end="")
        run = pymzml.run.Reader(path)
        ms1, ms2 = [], []
        for i, scan in enumerate(run):
            if scan.ms_level == 1:
                ms1.append(scan)
            else:
                ms2.append(scan)
        b = time.time()
        time1 = round(b - a, 2)
        print(f'\r Reading files finished! Total time: {time1} s           ', end='')
        
        return ms1, ms2


def peak_finding(eic, threshold=15):
    '''
    finding peaks in a single extracted chromatogram,and return peak index, left valley index, right valley index.
    :param eic: extracted ion chromatogram data; e.g., [1,2,3,2,3,1...]
    :param threshold: define the noise level for a peak, 6 is recommend
    :return:peak index, left valley index, right valley index.
    '''
    peaks, _ = scipy.signal.find_peaks(eic, width=2)
    prominence = scipy.signal.peak_prominences(eic, peaks)
    peak_prominence = prominence[0]
    left = prominence[1]
    right = prominence[2]
    ### peak_picking condition 1: value of peak_prominence must be higher than
    len_pro = len(peak_prominence)
    if len(peak_prominence) == 0:
        peak_index, left, right = np.array([]), np.array([]), np.array([])
    else:
        median_1 = np.median(peak_prominence)  ### 获得中位数的值
        index_pos2 = where(prominence[0] > threshold * median_1)[0]
        peak_index = peaks[index_pos2]
        left = left[index_pos2]
        right = right[index_pos2]
    return peak_index, left, right



def extract(df1, mz, error=50):
    '''
    Extracting chromatogram based on mz and error.
    :param df1: LC-MS dataframe, genrated by the function gen_df()
    :param mz: Targeted mass for extraction.
    :param error: mass error for extraction
    :return: rt,eic
    '''
    low = mz * (1 - error * 1e-6)
    high = mz * (1 + error * 1e-6)
    low_index = argmin(abs(df1.index.values - low))
    high_index = argmin(abs(df1.index.values - high))
    df2 = df1.iloc[low_index:high_index]
    rt = df1.columns.values
    if len(np.array(df2)) == 0:
        intensity = np.zeros(len(df1.columns))
    else:
        intensity = np.array(df2).T.sum(axis=1)
    return rt, intensity  ### 只返回RT和EIC



def gen_df_to_centroid(ms1, ms_round=4):
    '''
    Convert mzml data to a dataframe in centroid mode.
    :param ms1: ms scan list generated by the function of sep_scans(), or directed from pymzml.run.Reader(path).
    :return: A Dataframe
    '''
    t1 = time.time()
    l = len(ms1)
    num = 0
    print('\r Generating dataframe...             ', end="")
    ###将所有的数据转换成centroid格式，并将每个scan存在一个独立的变量scan(n)中
    for i in range(l):
        name = 'scan' + str(i)
        peaks, _ = scipy.signal.find_peaks(ms1[i].i.copy())
        locals()[name] = pd.Series(data=ms1[i].i[peaks], index=ms1[i].mz[peaks].round(ms_round),
                                   name=round(ms1[i].scan_time[0], 3))
        t2 = time.time()
        total_t = round(t2 - t1, 2)
        p = round(num / l * 100, 2)
        print(f'\r Reading each scans：{total_t} s, {num}/{l}, {p} %     ', end="")
        num += 1
    ### 将所有的变量汇总到一个列表中
    data = []
    for i in range(l):
        name = 'scan' + str(i)
        data.append(locals()[name])
    t3 = time.time()
    ## 开始级联所有数据
    print('\r Concatenating all the data...                   ', end="")
    df1 = pd.concat(data, axis=1)
    df2 = df1.fillna(0)
    t4 = time.time()
    t = round(t4 - t1, 2)
    print(f'\r Concat finished, Consumed time: {t} s            ', end='')
    return df2


def gen_df_raw(ms1, ms_round=4):
    '''
    Convert mzml data to a dataframe in profile mode.
    :param ms1: ms scan list generated by the function of sep_scans(), or directed from pymzml.run.Reader(path).
    :return: A Dataframe
    '''
    t1 = time.time()
    l = len(ms1)
    num = 0
    print('\r Generating dataframe...             ', end="")
    ###将每个scan存在一个独立的变量scan(n)中
    for i in range(l):
        name = 'scan' + str(i)
        locals()[name] = pd.Series(data=ms1[i].i, index=ms1[i].mz.round(ms_round), name=round(ms1[i].scan_time[0], 3))
        t2 = time.time()
        total_t = round(t2 - t1, 2)
        p = round(num / l * 100, 2)
        print(f'\r Reading each scans：{total_t} s, {num}/{l}, {p} %', end="")
        num += 1
    ### 将所有的变量汇总到一个列表中
    data = []
    for i in range(l):
        name = 'scan' + str(i)
        data.append(locals()[name])
    t3 = time.time()
    ## 开始级联所有数据
    print('\r Concatenating all the data...                             ', end="")
    df1 = pd.concat(data, axis=1)
    df2 = df1.fillna(0)
    t4 = time.time()
    t = round(t4 - t1, 2)
    print(f'\r Concat finished, Consumed time: {t} s                     ', end='')
    return df2


def B_spline(x, y):
    '''
    Generating more data points for a mass peak using beta-spline based on x,y
    :param x: mass coordinates
    :param y: intensity
    :return: new mass coordinates, new intensity
    '''
    t, c, k = interpolate.splrep(x, y, s=0, k=4)
    N = 300
    xmin, xmax = x.min(), x.max()
    new_x = np.linspace(xmin, xmax, N)
    spline = interpolate.BSpline(t, c, k, extrapolate=False)
    return new_x, spline(new_x)


def cal_bg(data):
    '''
    :param data: data need to calculate the background
    :return: background value
    '''
    if len(data) > 5:
        Median = median(data)
        Max_value = max(data)
        STD = std(data)
        Mean = mean(data)
        if Median == 0:
            bg = Mean + STD
        elif Mean <= Median * 5:
            bg = Max_value
        elif Mean > Median * 5:
            bg = Median
    else:
        bg = 1000000
    return bg + 1


def peak_checking_plot(df1, mz, rt1, Type='profile', path=None):
    '''
    Evaluating/visulizing the extracted mz
    :param df1: LC-MS dataframe, genrated by the function gen_df()
    :param mz: Targetd mass for extraction
    :param rt1: expected rt for peaks
    :return:
    '''

    fig = plt.figure(figsize=(12, 4))
    ### 检查色谱图ax
    ax = fig.add_subplot(121)
    rt, eic = extract(df1, mz, 50)
    rt2 = rt[where((rt > rt1 - 2) & (rt < rt1 + 2))]
    eic2 = eic[where((rt > rt1 - 2) & (rt < rt1 + 2))]
    ax.plot(rt2, eic2)
    ax.set_xlabel('Retention Time(min)', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    peak_index = np.argmin(abs(rt - rt1))
    peak_height = max(eic[peak_index - 2:peak_index + 2])
    ax.scatter(rt1, peak_height * 1.05, c='r', marker='*', s=50)
    ##计算背景
    bg_left, bg_right = cal_bg(eic2[:50]), cal_bg(eic2[-50:])
    rt3 = rt2[:50]
    rt4 = rt2[-50:]
    bg1 = zeros(50) + bg_left
    bg2 = zeros(50) + bg_right
    ax.plot(rt3, bg1)
    ax.plot(rt4, bg2)
    SN1 = round(peak_height / bg_left, 1)
    SN2 = round(peak_height / bg_right, 1)
    ax.set_title(f'SN_left:{SN1},         SN_right:{SN2}')
    ax.set_ylim(top=peak_height * 1.1, bottom=-peak_height * 0.05)

    ### 检查质谱图ax1
    ax1 = fig.add_subplot(122)
    width = 0.02
    spec = spec_at_rt(df1,rt1)  ## 提取到特定时间点的质谱图
    new_spec = target_spec(spec, mz, width=0.04)
    
    if Type == 'profile':
        mz_obs, error1, mz_opt, error2, resolution = evaluate_ms(new_spec, mz)
        ax1.plot(new_spec)
        ax1.bar(mz, max(new_spec.values), color='r', width=0.0005)
        ax1.bar(mz_opt,max(new_spec.values), color='g', width=0.0005)
        ax1.text(min(new_spec.index.values)+0.005, max(new_spec.values)*0.8, 
             f'mz_obs: {mz_obs},{error1} \n mz_opt:{mz_opt}, {error2}')
    else:
        ax1.bar(mz1, max(new_spec.values), width=0.0002)

    
    ax1.set_title(f'mz_exp: {mz}')
    ax1.set_xlabel('m/z', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_xlim(mz - 0.04, mz + 0.04)

    if path == None:
        pass
    else:
        plt.savefig(path, dpi=1000)
        plt.close('all')



        



def peak_alignment(files_excel, rt_error=0.1, mz_error=0.015):
    '''
    Generating peaks information with reference mz/rt pair
    :param files_excel: files for excels of peak picking and peak checking;
    :param rt_error: rt error for merge
    :param mz_error: mz error for merge
    :return: Export to excel files
    '''
    print('\r Generating peak reference...        ', end='')
    peak_ref = gen_ref(files_excel, rt_error=rt_error, mz_error=mz_error)
    j = 1
    for file in files_excel:
        print(f'\r for {j} files, generating alignment results...', end='')
        peak_p = pd.read_excel(file, index_col='Unnamed: 0').loc[:, ['rt', 'mz']].values
        peak_df = pd.read_excel(file, index_col='Unnamed: 0')
        new_all_index = []
        for i in range(len(peak_p)):
            rt1, mz1 = peak_p[i]
            index = np.where((peak_ref[:, 0] <= rt1 + rt_error) & (peak_ref[:, 0] >= rt1 - rt_error)
                             & (peak_ref[:, 1] <= mz1 + mz_error) & (peak_ref[:, 1] >= mz1 - mz_error))
            new_index = str(peak_ref[index][0][0]) + '_' + str(peak_ref[index][0][1])
            new_all_index.append(new_index)
        peak_df['new_index'] = new_all_index
        peak_df = peak_df.set_index('new_index')
        peak_df.to_excel(file.replace('.xlsx', '_alignment.xlsx'))
        j += 1


def database_evaluation(database, i, df1, df2, path=None):
    '''
    :param database: excel file containing compounds' information
    :param i:  the index for a row in excel
    :param df1: ms1 dataframe
    :param df2: ms2 dataframe
    :return:
    '''
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    formula = database.loc[i, 'Molecular Formula']
    rt_exp = database.loc[i, 'rt']
    mz_exp = round(database.loc[i, 'Positive'], 4)
    f = Formula(formula)
    a = f.spectrum()
    mz_iso, i_iso = np.array([a for a in a.values()]).T
    i_iso = i_iso / i_iso[0] * 100
    mz_iso += 1.00727647  ## 加个H
    frag_exp = float(database.loc[i, 'Fragment1'])

    fig = plt.figure(figsize=(14, 8))

    ### 1. 对MS1进行色谱峰提取
    ax1 = fig.add_subplot(231)
    ax1.set_title('MS1 extracted chromatogram', fontsize=10)
    rt1, eic1 = extract(df1, mz_exp, 50)
    ax1.plot(rt1, eic1)

    ax1.set_ylabel('Intensity', fontsize=12)
    peak_index = np.argmin(abs(rt1 - rt_exp))
    rt_obs = round(rt1[peak_index - 20:peak_index + 20][np.argmax(eic1[peak_index - 20:peak_index + 20])], 2)  ###找到峰的位置

    peak_height = max(eic1[peak_index - 2:peak_index + 2])
    ax1.scatter(rt_exp, peak_height * 1.05, c='r', marker='*', s=50)
    left1 = rt_exp - 1
    right1 = rt_exp + 1

    ax1.text(rt_obs + 0.05, peak_height * 0.7, f'RT_obs:{rt_obs} min \n RT_exp:{rt_exp} min')

    ax1.set_xlim(left=left1, right=right1)
    ax1.set_ylim(top=peak_height * 1.1, bottom=-peak_height * 0.05)

    ### 2. MS1质谱图质量评估
    ax2 = fig.add_subplot(232)
    ax2.set_title('Isotope check/MS1', fontsize=10)
    ind = np.argmin(abs(df1.columns.values - rt_exp))
    mz2, i2 = df1.iloc[:, ind].index, df1.iloc[:, ind].values

    peaks, _ = scipy.signal.find_peaks(i2)
    index1 = peaks[np.argmin(abs(mz2[peaks] - (mz_exp - 1)))]
    index2 = peaks[np.argmin(abs(mz2[peaks] - (mz_exp + 5)))]
    index_obs = peaks[np.argmin(abs(mz2[peaks] - mz_exp))]
    mz_obs = mz2[index_obs]
    height_obs = i2[index_obs]  ## 一定要放在之前
    mz2, i2 = mz2[index1:index2], i2[index1:index2]

    ax2.plot(mz2, i2)
    ax2.bar(mz_iso, -(height_obs / 100) * i_iso, width=0.03, color=['r'], zorder=2)
    ax2.text(mz_iso[0], -(height_obs / 100) * i_iso[0], round(mz_iso[0], 4))
    ax2.text(mz_iso[1], -(height_obs / 100) * i_iso[1], round(mz_iso[1], 4))
    ax2.text(mz_iso[2], -(height_obs / 100) * i_iso[2], round(mz_iso[2], 4))

    ### 3. 检查MS1质谱准确性
    ax3 = fig.add_subplot(233)
    ax3.set_title('Accurate mz check/MS1', fontsize=10)
    index3 = argmin(abs(mz2 - mz_exp)) - 12
    index4 = argmin(abs(mz2 - mz_exp)) + 12
    mz3, i3 = mz2[index3:index4], i2[index3:index4]
    mz3_opt, i3_opt = B_spline(mz3, i3)


    
    
    
    ax3.plot(mz3, i3, marker='o')  ### 原始数据
    ax3.plot(mz3_opt, i3_opt, c='r', lw=0.5)  ### 优化数据
    ax3.bar(mz_exp, height_obs, width=0.001, color=['g', 'b'])  ## 理论质量
    obs_error = round((mz_obs - mz_exp) / mz_exp * 1000000, 1)
    opt_error = round((mz_opt - mz_exp) / mz_exp * 1000000, 1)
    ax3.text(mz_obs - 0.05, height_obs * 0.8,
             f'  obs: {round(mz_obs, 4)} \n  obs_err: {obs_error} \n  opt: {round(mz_opt, 4)} \n  opt_err: {opt_error}')
    ax3.text(mz_obs + 0.01, height_obs * 1, f'exp: {mz_exp}')

    ## 4. 检查DFI 提取色谱图
    if np.isnan(frag_exp):
        print(f'frag_exp is not given, current frag_exp: {frag_exp}')
    else:
        ax4 = fig.add_subplot(234)
        ax4.set_title('mass spectrum/DFI', fontsize=10)
        ax4.set_xlabel('Retention Time(min)', fontsize=12)

        rt4, eic4 = extract(df2, frag_exp, 50)  ##提取改成df2
        ax4.plot(rt4, eic4)
        ax4.set_ylabel('Intensity', fontsize=12)
        peak_index = np.argmin(abs(rt4 - rt_exp))
        peak_height4 = max(eic4[peak_index - 2:peak_index + 2])
        ax4.scatter(rt_exp, peak_height4 * 1.05, c='r', marker='*', s=50)
        left4 = rt_exp - 1
        right4 = rt_exp + 1
        ax4.set_xlim(left=left4, right=right4)
        ax4.text(rt_exp + 0.05, peak_height4 * 0.7, f' RT_exp:{rt_exp} min')
        ax4.set_ylim(top=peak_height4 * 1.1, bottom=-peak_height4 * 0.05)

        ### 5. 检查DFI同位素峰
        ax5 = fig.add_subplot(235)
        ax5.set_title('DFI check', fontsize=10)
        ax5.set_xlabel('m/z', fontsize=12)
        index5 = np.argmin(abs(df2.columns.values - rt_obs))
        mz5, i5 = df2.iloc[:, index5].index.values, df2.iloc[:, index5].values
        index5_1 = np.argmin(abs(mz5 - frag_exp))
        height_obs = i5[index5_1]
        index5_2 = np.argmin(abs(mz5 - (frag_exp - 1)))
        index5_3 = np.argmin(abs(mz5 - (frag_exp + 4)))
        mz5, i5 = mz5[index5_2:index5_3], i5[index5_2:index5_3]
        ax5.plot(mz5, i5)
        frag_iso = mz_iso - (mz_iso - frag_exp)[0]
        ax5.bar(frag_iso + 0.1, (height_obs / 100) * i_iso, width=0.03, color=['m'], zorder=2)
        ax5.text(frag_iso[0] + 0.1, (height_obs / 100) * i_iso[0], round(frag_iso[0], 4))
        ax5.text(frag_iso[1] + 0.1, (height_obs / 100) * i_iso[1], round(frag_iso[1], 4))
        ax5.text(frag_iso[2] + 0.1, (height_obs / 100) * i_iso[2], round(frag_iso[2], 4))

        ### 6. 检查DFI质谱峰
        ax6 = fig.add_subplot(236)
        ax6.set_title('Accurate DFI check', fontsize=10)
        ax6.set_xlabel('m/z', fontsize=12)
        index6_1 = argmin(abs(mz5 - frag_exp)) - 12
        index6_2 = argmin(abs(mz5 - frag_exp)) + 12
        mz6, i6 = mz5[index6_1:index6_2], i5[index6_1:index6_2]
        frag_obs = round(mz6[np.argmax(i6)], 4)
        mz6_opt, i6_opt = B_spline(mz6, i6)
        frag_opt = mz6_opt[np.argmax(i6_opt)]

        ax6.plot(mz6, i6, marker='o')  ### 原始数据
        ax6.plot(mz6_opt, i6_opt, c='r', lw=0.5)  ### 优化数据
        ax6.bar(frag_exp, height_obs, width=0.001, color=['g'])

        obs_error = round((frag_obs - frag_exp) / mz_exp * 1000000, 1)
        opt_error = round((frag_opt - frag_exp) / mz_exp * 1000000, 1)
        ax6.text(frag_obs - 0.04, height_obs * 0.8,
                 f'  obs: {round(frag_obs, 4)} \n  obs_err: {obs_error} \n  opt: {round(frag_opt, 4)} \n  opt_err: {opt_error}')
        ax6.text(frag_obs + 0.01, height_obs * 1, f'exp: {frag_exp}')
        
    if path ==None:
        pass
    else:   
        fig.savefig(path, dpi=1000)
        plt.close(fig)

def peak_checking(peak_df, df1, error=50,
                  i_threshold=500, SN_threshold=5):
    '''
    Processing extracted peaks, remove those false positives.
    :param peak_df: Extracted peaks generated by the function of peak_picking
    :param df1: LC-MS dataframe, genrated by the function gen_df()
    :param error: For the function of extract(df,mz, error)
    :param i_threshold: filter peaks with intensity<i_threshold
    :param SN_threshold: filter peaks with sn<SN_threshold
    :return:
    '''
    final_result = pd.DataFrame()
    n = 0
    peak_num = len(peak_df['rt'])
    SN_all_left, SN_all_right, area_all, cor_all_mz, cor_all_i = [], [], [], [], []
    for i in range(peak_num):
        mz = peak_df.iloc[i]['mz']
        rt = peak_df.iloc[i]['rt']
        ### 第一步：处理色谱峰
        rt_e, eic_e = extract(df1, mz, error=error)
        peak_index = np.argmin(abs(rt_e - rt))  ## 找到特定时间点的索引
        rt_left = rt - 0.2
        rt_right = rt + 0.2
        peak_index_left = np.argmin(abs(rt_e - rt_left))
        peak_index_right = np.argmin(abs(rt_e - rt_right))
        mz_all, intensity_t = df1.iloc[:, peak_index].index.values, df1.iloc[:, peak_index].values  ## 提取特定时间的质谱峰
        try:
            peak_height = max(eic_e[peak_index - 2:peak_index + 2])
            other_peak = max(eic_e[peak_index - 5:peak_index + 5])
        except:
            peak_height = 1
            other_peak = 3
        rt_t, eic_t = rt_e[peak_index_left:peak_index_right], eic_e[peak_index_left:peak_index_right]
        try:
            area = scipy.integrate.simps(eic_e[peak_index - 40:peak_index + 40])
        except:
            area = scipy.integrate.simps(eic_e)
        if other_peak - peak_height > 1:
            bg_left, bg_right = 10000000, 10000000
        else:
            bg_left = cal_bg(eic_t[:50])
            bg_right = cal_bg(eic_t[-50:])

        SN_left = round(peak_height / bg_left, 1)
        SN_right = round(peak_height / bg_right, 1)
        SN_all_left.append(SN_left)
        SN_all_right.append(SN_right)
        area_all.append(area)

        ### 第二步：处理质谱峰
        mz_l, mz_h = mz - 0.005, mz + 0.005
        mz_all, intensity_t = df1.iloc[:, peak_index].index.values, df1.iloc[:, peak_index].values
        mz_index = argmin(abs(mz_all - mz))
        mz_width = 20
        mz_, i_ = mz_all[mz_index - mz_width:mz_index + mz_width], intensity_t[mz_index - mz_width:mz_index + mz_width]
        peaks, _ = scipy.signal.find_peaks(i_)
        peak_mz, peak_i = mz_[peaks], i_[peaks]
        peak_mz, peak_i = peak_mz[(peak_mz >= mz_l) & (peak_mz <= mz_h)], peak_i[(peak_mz >= mz_l) & (peak_mz <= mz_h)]
        if len(peak_mz) == 0:
            cor_mz, cor_i = 0, 1
        else:
            cor_mz = round(peak_mz[argmax(peak_i)], 4)
            cor_i = round(peak_i[argmax(peak_i)], 0)
        cor_all_mz.append(cor_mz)
        cor_all_i.append(cor_i)
        n += 1
        print(f'\r Processing peaks...{n}/{peak_num}                        ', end='')
    final_result['SN_left'] = SN_all_left
    final_result['SN_right'] = SN_all_right
    final_result['area'] = list(map(int, area_all))
    final_result['mz'] = cor_all_mz
    final_result['intensity'] = cor_all_i
    final_result['rt'] = peak_df['rt']
    ### 筛选条件，峰强度> i_threshold; 左边和右边SN至少一个大于SN_threshold
    final_result = final_result[(final_result['intensity'] > i_threshold) &
                                ((final_result['SN_left'] > SN_threshold) | (final_result['SN_right'] > SN_threshold))]
    final_result = final_result.loc[:, ['rt', 'mz', 'intensity', 'SN_left', 'SN_right', 'area']].sort_values(
        by='intensity').reset_index(drop=True)

    return final_result


def spec_at_rt(df1, rt):
    '''
    :param df1: LC-MS dataframe, genrated by the function gen_df()
    :param rt:  rentention time for certain ms spec
    :return: ms spec
    '''
    index = argmin(abs(df1.columns.values - rt))
    spec = df1.iloc[:, index]
    return spec









def concat_alignment(files_excel):
    '''
    Concatenate all data and return
    :param files_excel: excel files
    :param mode: selected 'area' or 'intensity' for each sample
    :return: dataframe
    '''
    align = []
    data_to_concat = []
    for i in range(len(files_excel)):
        if 'alignment' in files_excel[i]:
            align.append(files_excel[i])
    for i in range(len(align)):
        name = 'data' + str(i)
        locals()[name] = pd.read_excel(align[i], index_col='Unnamed: 0')
        data_to_concat.append(locals()[name])
    final_data = pd.concat(data_to_concat, axis=1)
    return final_data



def formula_to_distribution(formula, adducts='+H', num=3):
    '''
    :param formula: molecular formula, e.g., ‘C13H13N3’
    :param adducts: ion adducts, '+H', '-H'
    :return: mz_iso, i_iso (np.array)
    '''
    f = Formula(formula)
    a = f.spectrum()
    mz_iso, i_iso = np.array([a for a in a.values()]).T
    i_iso = i_iso / i_iso[0] * 100
    if adducts == '+H':
        mz_iso += 1.00727647
    elif adducts == '-H':
        mz_iso -= 1.00727647
    mz_iso = mz_iso.round(4)
    i_iso = i_iso.round(1)
    return mz_iso[:num], i_iso[:num]


def add_ms_values(new_spec):
    '''
    :param new_spec: the spectrum (pandas.Series)
    :return:  peak_mz,peak_i
    '''
    peaks, _ = scipy.signal.find_peaks(new_spec.values)
    peak3 = new_spec.iloc[peaks].sort_values().iloc[-3:]
    peak_mz = peak3.index.values
    peak_i = peak3.values
    return peak_mz, peak_i


def multi_process(file, company):
    ms1, *_ = sep_scans(file, company)
    df1 = gen_df_raw(ms1)
    peak_all = peak_picking(df1)
    peak_selected = peak_checking(peak_all, df1)
    peak_selected.to_excel(file.replace('.mzML', '.xlsx'))


def KMD_cal(mz_set, group='Br/H'):
    if '/' in group:
        g1, g2 = group.split('/')
        f1, f2 = Formula(g1), Formula(g2)
        f1, f2 = f1.spectrum(), f2.spectrum()
        f1_value, f2_value = [x for x in f1.values()][0][0], [x for x in f2.values()][0][0]
        values = [abs(f1_value - f2_value), round(abs(f1_value - f2_value), 0)]
        KM = mz_set * (max(values) / min(values))
        KMD_set = KM - np.floor(KM)

        print(f1_value, f2_value)
        print(min(values), max(values))
        print(values)
    else:
        g1 = Formula(group)
        f1 = g1.spectrum()
        f1_value = [x for x in f1.values()][0][0]
        KM = mz_set * (int(f1_value) / f1_value)
        KMD_set = KM - np.floor(mz_set)
    return KMD_set


def sep_result(result, replicate=4, batch=5):
    a = 0
    sep_result = []
    for i in range(batch):
        name = 'b' + str(i)
        sep_result.append(result[result.columns[a:a + replicate]])
        a += replicate

    return sep_result



def peak_checking_area(ref_all,df1,name='area'):
    '''
    Based on referece pairs, extract all peaks and integrate the peak area.
    :param ref_all: all referece pairs (dataframe)
    :param df1: LC-MS dataframe, genrated by the function gen_df()
    :param name: name for area
    :return: peak_ref (dataframe)
    '''
    area_all = []
    peak_index = np.array(ref_all['rt'].map(lambda x:str(round(x,2))).str.cat(ref_all['mz'].map(lambda x:str(round(x,4))),sep = '_'))
    num = len(ref_all)
    for i in range(num):
        rt,mz =ref_all.loc[i,['rt','mz']]
        rt1,eic1 = extract(df1,mz,50)
        rt_ind = argmin(abs(rt1-rt))
        left = argmin(abs(rt1-(rt-0.2)))
        right = argmin(abs(rt1-(rt+0.2)))
        rt_t,eic_t = rt1[left:right],eic1[left:right]
        area = round(scipy.integrate.simps(eic_t,rt_t),0)
        area_all.append(area)
        print(f'\r {i}/{num}', end = '')
    peak_ref = pd.DataFrame(area_all,index = peak_index,columns = [name])
    return peak_ref


def JsonToExcel(path):
    with open(path,'r',encoding='utf8')as fp:
        json_data = json.load(fp)
    Inchikey,precursor,frag,formula,smiles = [],[],[],[],[]
    num = len(json_data)
    for i in range(num):
        try:
            cmp_info = json_data[i]['compound'][0]['metaData']
            Inchikey.append([x['value'] for x in cmp_info if x['name']=='InChIKey'][0])
            formula.append([x['value'] for x in cmp_info if x['name']=='molecular formula'][0])
            precursor.append([x['value'] for x in cmp_info if x['name']=='total exact mass'][0])
            smiles.append([x['value'] for x in cmp_info if x['name']=='SMILES'][0])
        except:
            Inchikey.append(None)
            formula.append(None)
            precursor.append(None)
            smiles.append(None)
        frag.append(r'{' + json_data[i]['spectrum'].replace(' ',',') + r'}')    
        print(f'\r {round(i/num*100,2)}%',end='')
    database = pd.DataFrame(np.array([Inchikey,precursor,frag,formula,smiles]).T,
                            columns = ['Inchikey','Precursor','Frag','Formula','Smiles'])
    return database


def evaluate_ms(new_spec,mz_exp):
    peaks,_ = scipy.signal.find_peaks(new_spec.values)
    mz_obs = new_spec.index.values[peaks][argmin(abs(new_spec.index.values[peaks]-mz_exp))]
    x, y = B_spline(new_spec.index.values, new_spec.values)
    peaks,_ = scipy.signal.find_peaks(y)
    max_index = peaks[argmin(abs(x[peaks]-mz_exp))]
    half_height = y[peak_index]/2
    mz_left = x[:max_index][argmin(abs(y[:max_index] - half_height))]
    mz_right = x[max_index:][argmin(abs(y[max_index:] - half_height))]
    resolution = int(mz_obs / (mz_right - mz_left))
    mz_opt = round(mz_left+(mz_right - mz_left)/2, 4)
    mz_opt_ref = round(x[max_index],4)
    
    if abs(mz_opt-mz_opt_ref)/mz_exp*1000000 <10:
        final_mz_opt = mz_opt
    else:
        final_mz_opt = mz_opt_ref
        
    error1 = round((mz_obs -mz_exp)/mz_exp*1000000,1)
    error2 = round((final_mz_opt -mz_exp)/mz_exp*1000000,1)
    return mz_obs,error1,final_mz_opt,error2,resolution




def target_spec(spec, target_mz, width=0.04):
    '''
    :param spec: spec generated from function spec_at_rt()
    :param target_mz: target mz for inspection
    :param width: width for data points
    :return: new spec and observed mz
    '''
    index = argmin(abs(spec.index.values-target_mz))
    index_left =argmin(abs(spec.index.values-(target_mz-width)))
    index_right =argmin(abs(spec.index.values-(target_mz+width)))
    new_spec = spec.iloc[index_left:index_right]
    return new_spec


def gen_ref(files_excel, mz_error=0.015, rt_error=0.1):
    '''
    For alignment, generating a reference mz/rt pair
    :param files_excel: excel files path for extracted peaks
    :return: mz/rt pair reference
    '''
    data = []
    for i in range(len(files_excel)):
        name = 'peaks' + str(i)
        locals()[name] = pd.read_excel(files_excel[i], index_col='Unnamed: 0').loc[:, ['rt', 'mz']].values
        data.append(locals()[name])
        print(f'\r Reading excel files... {i}/{len(files_excel)}                   ', end="")
    print(f'\r Concatenating all peaks...                 ', end='')
    pair = np.concatenate(data, axis=0)
    peak_all_check = pair
    peak_ref = []
    while len(pair) > 0:
        rt1, mz1 = pair[0]
        index1 = np.where((pair[:, 0] <= rt1 + rt_error) & (pair[:, 0] >= rt1 - rt_error)
                          & (pair[:, 1] <= mz1 + mz_error) & (pair[:, 1] >= mz1 - mz_error))
        peak = np.mean(pair[index1], axis=0).tolist()
        peak = [round(peak[0],2),round(peak[1],4)]
        pair = np.delete(pair, index1, axis=0)
        peak_ref.append(peak)
        print(f'\r  {len(pair)}                        ', end='')

    peak_ref2 = np.array(peak_ref)
    
    ### 检查是否有漏的
    peak_lost = []
    for peak in peak_all_check:
        rt1, mz1 = peak
        check = np.where((peak_ref2[:, 0] <= rt1 + rt_error) & (peak_ref2[:, 0] >= rt1 - rt_error)
                         & (peak_ref2[:, 1] <= mz1 + mz_error) & (peak_ref2[:, 1] >= mz1 - mz_error))
        if len(check[0]) == 0:
            peak_lost.append([rt1, mz1])
    peak_lost=np.array(peak_lost)
    while len(peak_lost) > 0:
        rt1, mz1 = peak_lost[0]
        index1 = np.where((peak_lost[:, 0] <= rt1 + rt_error) & (peak_lost[:, 0] >= rt1 - rt_error)
                          & (peak_lost[:, 1] <= mz1 + mz_error) & (peak_lost[:, 1] >= mz1 - mz_error))
        peak = np.mean(peak_lost[index1], axis=0).tolist()
        peak = [round(peak[0],2),round(peak[1],4)]
        peak_lost = np.delete(peak_lost, index1, axis=0)
        peak_ref.append(peak)
        print(f'\r  {len(pair)}                        ', end='')
    
    return np.array(peak_ref)


if __name__ == '__main__':
    pass
#     path = r'D:\TOF-Ms DATA\HYH-MZML\混合实验\*.mzML'
#     files_mzml = glob(path)
#     pool = Pool(processes = 5)
#     for file in files_mzml:
#         pool.apply_async(multi_process,args=(file,))
#     print('Finished')
#     pool.close()
#     pool.join()
