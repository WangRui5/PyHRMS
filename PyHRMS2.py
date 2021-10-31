from rdkit.Chem import Draw
from rdkit import Chem
from matplotlib.offsetbox import DrawingArea,OffsetImage,AnnotationBbox,TextArea
import matplotlib.image as mpimg


def database_1st_plot(df1,mz,smi=None,name = None, formula = None,CAS = None, path=None):
    fig = plt.figure(figsize = (12,8))
    
    ### 图1 色谱图
    ax1 = fig.add_subplot(221)
    rt,eic = extract(df1,mz,50)
    ax1.plot(rt,eic)
    peak_index,left,right = peak_finding(eic)
    if len(peak_index) ==0:
        rt_obs,peak_h = array([0]),array([0])
    else:
        rt_obs,peak_h = rt[peak_index],eic[peak_index]
        rt_obs = rt_obs[argmax(peak_h)]
    peak_h =max(peak_h)
    ax1.text(rt_obs,peak_h,f'{rt_obs.round(2)}')
    ax1.set_xlabel('Retention Time(min)')
    ax1.set_ylabel('Intensity')    
    
    ### 图2 质谱图
    ax2 = fig.add_subplot(222)
    rt1 = rt[argmax(eic)]
    spec = spec_at_rt(df1,rt1)
    new_spec,mz_obs=target_spec(spec,mz,width = 600)
    ax2.plot(new_spec)
    ax2.set_title(f'RT: {rt_obs}')
    ms_peak_height = max(new_spec.values)
    ## 添加质谱数值
    peak_mz, peak_i = add_ms_values(new_spec)
    ax2.text(peak_mz[-1],peak_i[-1], round(peak_mz[-1], 4))
    ax2.text(peak_mz[-2],peak_i[-2], round(peak_mz[-2], 4))
    
    
    if formula == None:
        pass
    else:
        mz_iso1, i_iso1 = formula_to_distribution(formula,adducts = '+H') ## 添加同位素峰
        mz_iso2, i_iso2 = formula_to_distribution(formula,adducts = '-H')
        ax2.bar(mz_iso1,-i_iso1*ms_peak_height/100,width = 0.02,color = 'r')
        ax2.bar(mz_iso2,-i_iso2*ms_peak_height/100,width = 0.02,color = 'g')
    
    ax2.set_xlabel('m/z')
    ax2.set_ylabel('Intensity')
    
    ax2.text(mz_iso1[0], -i_iso1[0]*ms_peak_height/100, round(mz_iso1[0], 4)) ## 添加同位素峰标记
    ax2.text(mz_iso1[1], -i_iso1[1]*ms_peak_height/100, round(mz_iso1[1], 4))
    ax2.text(mz_iso2[0], -i_iso2[0]*ms_peak_height/100, round(mz_iso1[0], 4))
    ax2.text(mz_iso2[1], -i_iso2[1]*ms_peak_height/100, round(mz_iso1[1], 4))
    
    
    ### 图3 质谱图 显示分辨率
    ax3 = fig.add_subplot(223)
    new_spec,mz_obs = target_spec(spec,mz,width=15)
    ax3.plot(new_spec)
    ax3.bar(mz,max(new_spec.values),width =0.001,color='r')
    ax3.set_xlabel('m/z')
    ax3.set_ylabel('Intensity')
    x,y = B_spline(new_spec.index.values,new_spec.values)
    ax3.plot(x,y,lw=0.5,c='g')
    mz_obs,error1,mz_opt,error2,mz_res= evaluate_ms(spec,mz)
    ax3.text(min(new_spec.index.values),max(new_spec.values)*0.5,
             f' mz_obs:{mz_obs}, {error1} \n mz_opt: {mz_opt}, {error2} \n Resolution: {mz_res}')
    
     ### 图4 显示分子信息
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    ### 将分子图片放入box
    if smi == 'Not Found':
        pass
    else:
        mol = Chem.MolFromSmiles(smi)
        img = Draw.MolToImage(mol,size = (300,300),kekulize=True)
        imagebox = OffsetImage(img,zoom=0.5) ## 将图片放在OffsetBox容器中
        imagebox2 = AnnotationBbox(imagebox,(0.6,0.6),frameon=False)  ##使用AnnotationBbox添加到画布中
        ax4.add_artist(imagebox2)  ##最后用ax.add_artist(ab)应用加入到画布中
        ax4.text(0.3,0.2,f' Name:{name} \n Formula: {formula} \n CAS: {CAS} \n mz_exp: {mz}',zorder =15)
    if path ==None:
        pass
    else:
        fig.savefig(path,dpi=600)
        plt.close('all')
