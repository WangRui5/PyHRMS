import pandas as pd
from bisect import bisect_left
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, RadioButtons
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
from matplotlib.ticker import FuncFormatter
import pyisopach
#%config InlineBackend.figure_format = 'retina'


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
    if len(peak_prominence) != 0:
        median_1 = np.median(peak_prominence)  ### 获得中位数的值
        index_pos2 = []
        for i in range(len_pro):
            if peak_prominence[i] > median_1 * threshold:
                index_pos2.append(i)
        peak_index = peaks[index_pos2]
        left = left[index_pos2]
        right = right[index_pos2]
    else:
        peak_index, left, right = np.array([]), np.array([]), np.array([])
    return peak_index, left, right


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


def sep_scans(path):
    '''
    To separate scan for MS1, MS2 and lockspray. Only supported for Waters .raw data
    :param path: The path for mzML files
    :return: ms1, ms2 and lockspray
    '''
    a = time.time()
    print('\r Reading files...             ', end="")
    run = pymzml.run.Reader(path)
    ms1 = []
    ms2 = []
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
    print(f'\r Reading files finished! {time1} s        ', end='')
    return ms1, ms2, lockspray


def extract(df1, mz, error):
    '''
    Extracting chromatogram based on mz and error.
    :param df1: LC-MS dataframe, genrated by the function gen_df()
    :param mz: Targeted mass for extraction.
    :param error: mass error for extraction
    :return: rt,eic
    '''
    low = mz * (1 - error * 1e-6)
    high = mz * (1 + error * 1e-6)
    low_index = bisect_left(df1.index, low)
    high_index = bisect_left(df1.index, high)
    df2 = df1.iloc[low_index:high_index]
    rt = df1.columns
    if len(np.array(df2)) == 0:
        intensity = np.zeros(len(df1.columns))
    else:
        intensity = np.array(df2).T.sum(axis=1)
    return rt, intensity  ### 只返回RT和EIC


def gen_df_centroid(ms1):
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
        locals()[name] = pd.Series(data=ms1[i].i[peaks], index=ms1[i].mz[peaks], name=ms1[i].scan_time[0])
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


def gen_df_profile(ms1):
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
        peaks, _ = scipy.signal.find_peaks(ms1[i].i.copy())
        locals()[name] = pd.Series(data=ms1[i].i, index=ms1[i].mz, name=ms1[i].scan_time[0])
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


def digMs(path):
    '''
    Visualize the LC-MS data
    :param path: the path for the LC-MS mzML file
    :return:
    '''
    def ms_show(labels):
        ax.cla()  ### 清除之前的数据
        x = datapool[labels][0]
        y = datapool[labels][1]  ### 定义x, y的值，以labels为索引，找到datapool里对应的rt和tic
        art, = ax.plot(x, y, marker='o', markersize=0.5)  # 绘制上面的子图(质谱图)
        art.set_picker(True)
        art.set_pickradius(10)  ### 用于设置拾取器使用的轴的深度

        index1 = scipy.signal.find_peaks(y, distance=150, height=2 * mean(y))[0]
        index1 = index1[:5]
        for i in index1:
            ax.text(s=np.round(x[i], 2), x=x[i], y=y[i])
        ax.set_ylabel('Intensity')
        ax.set_xlabel('Retention Time(min)')

        ### 色谱图与质谱图交互
        fig.canvas.draw()
        fig.canvas.mpl_connect('pick_event', click)  # 交互操作函数
        global label
        label = labels

    def click(event):
        '''
        show ms
        :param event:
        :return:
        '''
        # if event.artist != art:
        #     return
        ind = event.ind[0]
        x = scanpool[label][ind].mz
        y = scanpool[label][ind].i
        ax1.cla()  # 清除之前所绘制图形
        ax1.plot(x, y)  # 绘制横向柱状图
        ### 在图的右上角显示质谱图时间
        rt_show = f'RT: {np.round(scanpool[label][ind].scan_time[0], 2)} min'
        ms_show = np.round(x, 4)
        ax1.set_title(f'{rt_show}', loc='right')

        index = scipy.signal.find_peaks(y.copy(), distance=50, height=max(max(y.copy()) / 10, 1000))[0]
        index = index[:5]
        for i in index:
            ax1.text(s=ms_show[i], x=x[i], y=y[i])
            # plt.text(s =ms_show[i],x =x[i], y = y[i])

        ax1.set_xlabel('m/z')
        ax1.set_ylabel('Intensity')
        fig.canvas.draw()

    ms1, ms2, locksp = sep_scans(path)
    rt1, rt2, rt3, tic1, tic2, tic3 = [], [], [], [], [], []

    for scan in ms1:
        rt1.append(scan.scan_time[0])
        tic1.append(scan.TIC)

    for scan in ms2:
        rt2.append(scan.scan_time[0])
        tic2.append(scan.TIC)
    for scan in locksp:
        rt3.append(scan.scan_time[0])
        tic3.append(scan.TIC)

    # 声明一个变量用来定义使用的数据类别（i.e., ms1, ms2, lockspray）
    global label
    label = 'ms1'

    ### 将数据存在字典中
    datapool = {'ms1': [rt1, tic1], 'ms2': [rt2, tic2], 'lockspray': [rt3, tic3]}
    scanpool = {'ms1': ms1, 'ms2': ms2, 'lockspray': locksp}

    ### 创建画布
    fig, (ax, ax1) = plt.subplots(2, 1, figsize=(10, 10))

    ### 设置按钮位置和大小
    ax_MS = plt.axes([0.80, 0.79, 0.1, 0.09])
    ms_button = RadioButtons(ax_MS, ['ms1', 'ms2', 'lockspray'], [False, False, True], activecolor='b')

    ### 设置峰提取
    index1 = scipy.signal.find_peaks(tic1, distance=50, height=mean(tic1))[0]
    index1 = index1[:100]
    for i in index1:
        ax.text(s=np.round(rt1[i], 2), x=rt1[i], y=tic1[i])
    ax.set_ylabel('Intensity')
    ax.set_xlabel('Retention Time(min)')

    # -----button 控件------

    ms_button.on_clicked(ms_show)
    plt.show()


def extract_vis(df1):
    '''
    Visualize the extracted chromatogram
    :param df1: LC-MS dataframe, genrated by the function gen_df()
    :return:
    '''
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    rt1 = df1.columns
    tic1 = np.array(df1).T.max(axis=1)
    ax.plot(rt1, tic1, lw=1, color='r')

    def submit2(mass):
        data = mass.split(',')
        if len(data) == 2:
            mass, error = float(data[0]), int(data[1])
        elif len(data) == 1:
            mass, error = float(data[0]), 10
        else:
            mass, error = 200, 20
        mz = float(mass)
        ax.cla()
        rt, EIC = extract(df1, mz, error)
        ax.plot(rt, EIC)  ### 重新作图
        x_peak_index = scipy.signal.find_peaks(EIC, distance=50, height=max(max(EIC.copy()) / 10, 2000))[0]  ## 定义峰的范围
        if len(x_peak_index) != 0:
            index_list = x_peak_index.tolist()
            for i in index_list:
                f = int(i)
                w = 30  ## 设置峰宽
                ax.fill_between(rt[(f - w):(f + w)], EIC[f - w], EIC[(f - w):(f + w)], color='b', alpha=0.3)
                area = scipy.integrate.simps(EIC[(f - w):(f + w)], rt[(f - w):(f + w)])
                ### 获取标记峰的位置
                x_peak = rt[x_peak_index]
                y_peak = EIC[x_peak_index] * 1.07  # 给rt和area的位置
                y_peak2 = EIC[x_peak_index] * 1.03  # 给标记*的位置
                ax.plot(x_peak, y_peak2, '*')
                rt1 = round(x_peak[0], 2)
                area1 = int(area)
                ### 保留时间，峰面积放在峰上吗
                ax.text(x_peak[0], y_peak, f'RT: {rt1} min \narea: {area1} ')

    axbox = fig.add_axes([0.2, 0.05, 0.2, 0.075])
    text_box = TextBox(axbox, "Extracted m/z:  ")
    text_box.on_submit(submit2)
    plt.show()


def evaluate_ms(df1, mz, rt):
    '''
    Evalute the peaks extracted based on mz and rt
    :param df1: LC-MS dataframe, genrated by the function gen_df()
    :param mz: target mz for extraction
    :param rt: rt for expected peaks
    :return: rt_find, mz_find, mz_error1, corr_mz, mz_error2, intensity_find
    '''
    if (rt is nan) or (mz is nan):
        rt_find, mz_find, mz_error1, corr_mz, mz_error2, intensity_find = 'error', 'error', 'error', 'error', 'error', 'error'
    else:
        ### Step2 针对MS1，找到特定时间点的质谱图
        mz_rt = bisect_left(df1.columns, rt)
        mz_spec = df1.iloc[:, mz_rt]
        mz1 = mz_spec.index
        intensity1 = mz_spec.values
        peaks, _ = scipy.signal.find_peaks(intensity1)

        index = np.argmin(abs(mz1[peaks] - mz))

        mz_index = peaks[index]
        mz_find = round(mz1[mz_index], 4)
        mz_error1 = round((mz_find - mz) / mz * 1000000, 2)
        rt_find = round(df1.columns[mz_rt], 2)
        intensity_find = intensity1[mz_index]

        ### 对找到的质谱做矫正，仅供参考；
        mz_width = 12
        mz_for_corr = mz1[mz_index - mz_width:mz_index + mz_width]
        i_for_corr = intensity1[mz_index - mz_width:mz_index + mz_width]
        beta_mz, beta_i = B_spline(mz_for_corr, i_for_corr)
        corr_mz = round(beta_mz[np.argmax(beta_i)], 4)
        mz_error2 = round((corr_mz - mz) / mz * 1000000, 2)
    return beta_mz, beta_i, mz,rt
    #return rt_find, mz_find, mz_error1, corr_mz, mz_error2, intensity_find


def evaluate_peak(df1, mz, rt):
    '''
    Evalute the extracted peak based on mz and rt
    :param df1: LC-MS dataframe, genrated by the function gen_df()
    :param mz: target mass for extraction
    :param rt: expected rt for peak
    :return: rt, i, sn, area
    '''
    if (rt is nan) or (mz is nan):
        peak_rt, peak_i, peak_SN, peak_area = 'error', 'error', 'error', 'error'
    else:
        ### Step1 针对MS1，找到色谱图；
        rt1, eic1 = extract(df1, mz, 50)  ### 检查是否有母体色谱峰；
        #### 看提取的是否有峰
        rt_left = bisect_left(rt1, rt - 0.3)
        rt_right = bisect_left(rt1, rt + 0.3)
        peak_rt1 = rt1[rt_left:rt_right]
        peak_eic1 = eic1[rt_left:rt_right]
        peaks, _ = scipy.signal.find_peaks(peak_eic1)
        if len(peaks) != 0:
            index = np.argmax(peak_eic1[peaks])
            peak_index = peaks[index]
            peak_rt = round(peak_rt1[peak_index], 2)
            peak_i = peak_eic1[peak_index]
            peak_bg = np.median(peak_eic1)
            if peak_bg != 0:
                peak_SN = round(peak_i / peak_bg, 0)
            else:
                peak_SN = peak_i
            peak_area = round(scipy.integrate.simps(peak_eic1), 0)
        else:
            peak_rt, peak_i, peak_SN, peak_area = 'error1', 'error1', 'error1', 'error1'
    return peak_rt, peak_i, peak_SN, peak_area


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


def peak_checking_plot(df1, mz, rt1):
    '''
    Evaluating/visulizing the extracted mz
    :param df1: LC-MS dataframe, genrated by the function gen_df()
    :param mz: Targetd mass for extraction
    :param rt1: expected rt for peaks
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rt, eic = extract(df1, mz, 50)
    ax.plot(rt, eic)
    ax.set_xlabel('Retention Time(min)', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    peak_index = np.argmin(abs(rt - rt1))
    peak_height = max(eic[peak_index - 2:peak_index + 2])
    ax.scatter(rt1, peak_height * 1.05, c='r', marker='*', s=50)
    left1 = rt1 - 1
    right1 = rt1 + 1
    ax.set_xlim(left=left1, right=right1)
    ax.set_ylim(top=peak_height * 1.1, bottom=-peak_height * 0.05)


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
    peak_num = len(peak_df['rt'])
    n = 0
    SN_all = []
    area_all = []
    for i in range(peak_num):
        mz = peak_df.iloc[i]['mz']
        rt_1 = peak_df.iloc[i]['rt']
        intensity = int(peak_df.iloc[i]['intensity'])
        rt, eic = extract(df1, mz, error=error)
        peak_index = np.argmin(abs(rt - rt_1))  ##计算一下peak height
        try:
            peak_height = max(eic[peak_index - 2:peak_index + 2])
            other_peak = max(eic[peak_index - 5:peak_index + 5])
        except:
            peak_height = 1
            other_peak = 3
        rt_index = np.where((rt > rt_1 - 1) & (rt < rt_1 + 1))
        eic = eic[rt_index]
        area = scipy.integrate.simps(eic)
        if other_peak - peak_height > 1:
            bg = 10000000
        else:
            if np.median(eic) == 0:
                bg = 1
            else:
                bg = np.median(eic)

        SN = round(peak_height / bg, 1)
        SN_all.append(SN)
        area_all.append(area)
        n += 1
        print(f'\r {n}/{peak_num}                        ', end='')
    peak_df['SN'] = SN_all
    peak_df['area'] = list(map(int, area_all))
    final_df = peak_df[(peak_df['intensity'] > i_threshold)
                       & (peak_df['SN'] > SN_threshold)]
    final_df = final_df.sort_values(by='intensity', ascending=True)
    result_all = final_df.reset_index(drop=True)

    #### 对找到的峰进行筛选，去除重复的值

    rt_error = 0.1
    mz_error = 0.005
    final_list = pd.DataFrame()
    num = 0
    total_num = len(result_all)
    while len(result_all) != 0:
        rt, mz = result_all.iloc[0]['rt'], result_all.iloc[0]['mz']
        align = result_all[(result_all['rt'] < rt + rt_error) & (result_all["rt"] > rt - rt_error)
                           & (result_all["mz"] > mz - mz_error) & (
                                   result_all["mz"] < mz + mz_error )]
        final_list = final_list.append(align.iloc[0])
        result_all = result_all.drop(align.index)
        print(f'\r Starting alignment...   {len(result_all)}             ', end='')
    final_list = final_list.reset_index(drop=True)
    final_list = final_list[['rt', 'mz', 'intensity', 'SN', 'area']]
    return final_list


def gen_ref(files_excel,mz_error=0.005,rt_error = 0.1):
    '''
    For alignment, generating a reference mz/rt pair
    :param files_excel: excel files path for extracted peaks
    :return: mz/rt pair reference
    '''
    for i in range(len(files_excel)):
        name = 'peaks' + str(i)
        locals()[name] = pd.read_excel(files_excel[i], index_col='Unnamed: 0').loc[:,['rt','mz']].values
        print(f'\r Reading excel files... {i}/{len(files_excel)}                   ', end="")
    data = []
    for i in range(len(files_excel)):
        name = 'peaks' + str(i)
        data.append(locals()[name])
    print(f'\r Concatenating all peaks...                 ',end = '')
    pair = np.concatenate(data, axis=0)
    peak_all_check = pair
    peak_ref = []
    while len(pair) > 0:
        rt1, mz1 = pair[0]
        index1 = np.where((pair[:, 0] <= rt1 + rt_error) & (pair[:, 0] >= rt1 - rt_error)
                          & (pair[:, 1] <= mz1 + mz_error ) & (pair[:, 1] >= mz1  - mz_error))
        peak = np.mean(pair[index1], axis=0).tolist()
        pair = np.delete(pair, index1, axis=0)
        peak_ref.append(peak)
        print(f'\r  {len(pair)}                        ', end='')
    
    peak_ref2 = np.array(peak_ref)
    for peak in peak_all_check:
        rt1,mz1 = peak
        check = np.where((peak_ref2[:, 0] <= rt1 + rt_error) & (peak_ref2[:, 0] >= rt1 - rt_error)
                      & (peak_ref2[:, 1] <= mz1 + mz_error ) & (peak_ref2[:, 1] >= mz1  - mz_error))
        if len(check[0]) == 0:
            peak_ref.append([rt1,mz1])
    return np.array(peak_ref)
    


    

    
    
def peak_alignment(files_excel,rt_error = 0.1, mz_error = 0.005):
                  ):
    '''
    Generating peaks information with reference mz/rt pair
    :param files_excel: files for excels of peak picking and peak checking;
    :param rt_error: rt error for merge
    :param mz_error: mz error for merge
    :return: Export to excel files
    '''
    peak_ref = gen_ref(files_excel)
    for file in files_excel:
        peak_p = pd.read_excel(file, index_col='Unnamed: 0').loc[:,['rt','mz']].values
        peak_df = pd.read_excel(file, index_col='Unnamed: 0')
        new_all_index = []
        for i in range(len(peak_p)):
            rt1,mz1 = peak_p[i]
            index = np.where((peak_ref[:, 0] <= rt1 + rt_error) & (peak_ref[:, 0] >= rt1 - rt_error)
                              & (peak_ref[:, 1] <= mz1 + mz_error ) & (peak_ref[:, 1] >= mz1  - mz_error))
            new_index = str(round(peak_ref[index][0][0],2)) +'_'+str(round(peak_ref[index][0][1],4))
            new_all_index.append(new_index)
        peak_df['new_index'] = new_all_index
        peak_df =peak_df.set_index('new_index')
        peak_df.to_excel(file.replace('.xlsx','_alignment.xlsx'))
        
        
def concat_alignment(files_excel,mode = 'area'):
    '''
    Concatenate all data and return 
    :param files_excel: excel files
    :param mode: selected 'area' or 'intensity' for each sample 
    :return: dataframe
    '''
    align = []
    for i in range(len(files_excel)):
        if 'alignment' in files_excel[i]:
            align.append(files_excel[i])
    for i in range(len(align)):
        i_name = os.path.split(align[i])[-1].split('.')[0] +'_i'
        a_name =  os.path.split(align[i])[-1].split('.')[0] +'_a'
        data = pd.read_excel(align[i],index_col='new_index')
        data[i_name] = data['intensity']
        data[a_name] = data['area']
        data = data[~data.index.duplicated(keep='last')]
        name = 'data' + str(i)
        if mode =='area':
            locals()[name] = data.loc[:,[a_name]]
        elif mode == 'intensity':
            locals()[name] = data.loc[:,[i_name]]
    data_to_concat = []
    for i in range(len(align)):
        name = 'data' + str(i)
        data_to_concat.append(locals()[name])
    final_data = pd.concat(data_to_concat, axis=1)    
    return final_data



def database_evaluation(database, i, df1, df2):
    '''
    :param database: excel file containing compounds' information
    :param i:  the index for a row in excel
    :param df1: ms1 dataframe
    :param df2: ms2 dataframe
    :return:
    '''
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    formula = database.loc[i, 'formula']
    rt_exp = database.loc[i, 'RT']
    mz_exp = round(database.loc[i, 'Positive'], 4)
    mol = pyisopach.Molecule(formula)
    mz_iso, i_iso = mol.isotopic_distribution(charge=0)
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
    ax2.bar(mz_iso + 0.1, (height_obs / 100) * i_iso, width=0.03, color=['r'], zorder=2)
    ax2.text(mz_iso[0] + 0.1, (height_obs / 100) * i_iso[0], round(mz_iso[0], 4))
    ax2.text(mz_iso[1] + 0.1, (height_obs / 100) * i_iso[1], round(mz_iso[1], 4))
    ax2.text(mz_iso[2] + 0.1, (height_obs / 100) * i_iso[2], round(mz_iso[2], 4))

    ### 3. 检查MS1质谱准确性
    ax3 = fig.add_subplot(233)
    ax3.set_title('Accurate mz check/MS1', fontsize=10)
    index3 = argmin(abs(mz2 - mz_exp)) - 12
    index4 = argmin(abs(mz2 - mz_exp)) + 12
    mz3, i3 = mz2[index3:index4], i2[index3:index4]
    mz3_opt, i3_opt = B_spline(mz3, i3)
    mz_opt = mz3_opt[np.argmax(i3_opt)]
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

        
if __name__ == '__main__':
    pass
