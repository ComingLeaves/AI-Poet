import sys
import os
import json
import re
import numpy as np

def _parseRawData(author=None,constrain=None,src='./chinese-poetry/json/simplified',category="poet.tang"):
    """
    处理json文件，返回诗歌内容列表
    :param author: 作者名字
    :param constrain: 长度限制
    :param src: 文件存放路径
    :param category: 类别，有poet.song和poet.tang
    
    :return:data :list['每首古诗内容'] 
    """
    def sentenceParse(para):
        #para 繁体古诗，包括换型字，如：生摘琵琶酸，“琵琶”做“枇杷”
        #re.subn()用来匹配和替换内容
        result,number = re.subn(u"(.*)","",para)
        result,number = re.subn(u"{.*}","",para)
        result,number = re.subn(u"《.*》","",para)
        result,number = re.subn(u"《.*》","",para)
        result,number = re.subn(u"[\]\[]","",para)
        r = ""
        for s in result:
            if s not in set('0123456789-'):
                r += s
        r,number = re.subn(u"。。 ",u"。 ",r)
        return r

    def handleJson(file):
        rst = []
        data = json.loads(open(file).read())
        for poetry in data:
            pdata = ""

    return None

#获取numpy压缩包的数据
def get_data(opt):
    """
    
    :param opt:配置选项，Config对象
    :return: word2ix:dict ,每个字对应的序号，形如u'月'->100
    :return: ix2word:dict,每个序号对应的字，形如'100'->'月'
    :return: data:numpy数组，每一行是一首诗对应的字的下标 
    
    """
    if os.path.exists(opt.pickle_path):
        data = np.load(opt.pickle_path,allow_pickle=True)
        data,word2ix,ix2word = data['data'],data['word2ix'].item(),data['ix2word'].item()
        return data,word2ix,ix2word