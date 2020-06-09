import os
import sys

import ipdb
import torch as t
import tqdm
from torch import nn
from torchnet import meter

from data import get_data
from model import PoetryModel
from utils import Visualizer

class Config(object):
    data_path = 'data/' #诗歌的文本文件存放路径
    pickle_path = 'tang.npz' #预处理好的二进制诗歌集
    author = None #只学习某位作者的诗歌
    constrain = None
    category = 'poet.tang' #诗歌类别，唐诗还是宋词
    lr = 1e-3
    weight_decay = 1e-4
    use_gpu = True
    epoch = 20
    batch_size = 128  # 一次训练所取的样本数
    maxlen = 125 #诗歌最大长度
    plot_every = 20 #每20个batch可视化一次
    use_env  = True #是否使用visdom可视化
    env = 'poetry' #visdom env
    max_gen_len = 200 #生成诗歌最大长度
    debug_file = '/tmp/debugp'
    model_path = None #预训练模型路径

    #生成诗歌相关配置
    prefix_words = '细雨鱼儿出，微风燕子斜。' #控制诗歌生成的格式与语境,默认值
    start_words = '闲云潭影日悠悠' #诗歌以此开头
    acrostic = False #是否藏头诗
    model_prefix = 'checkpoints/tang' #模型保存路径

opt = Config()

def train(**kwargs):
    for k,v in kwargs.items():
        setattr(opt,k,v)

    vis = Visualizer(env=opt.env)

    #获取数据
    data,word2ix,ix2word = get_data(opt)
    data = t.from_numpy(data)
    dataloader = t.utils.data.DataLoader(data,batch_size=opt.batch_size,shuffle=True,num_workers=1)

    #模型定义
    model = PoetryModel(len(word2ix),128,256)
    optimizer = t.optim.Adam(model.parameters(),lr=opt.lr)
    criterion = nn.CrossEntropyLoss()

    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))

    if opt.use_gpu:
        model.cuda()
        criterion.cuda()
    loss_meter = meter.AverageValueMeter()

    for epoch in range(opt.epoch):
        loss_meter.reset()
        for li,data_ in tqdm.tqdm(enumerate(dataloader)):
            #训练
            data_ = data_.long().transpose(1,0).contiguous()
            if opt.use_gpu: data_ = data_.cuda()
            optimizer.zero_grad()
            ##输入和目标错开
            input_,target = Variable(data_[:-1,:]),Variable(data_[1:,:])
            output,_ = model(input_)
            loss = criterion(output,target.view(-1))
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.data[0])

            # 可视化
            if (1+ii)%opt.plot_every==0:

                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                vis.plot('loss',loss_meter.value()[0])
                #诗歌原文
                poetrys = [[ix2word[_word] for _word in data_[:,-iii]] for _iii in range(data_.size(1))][:16]
                vis.text('</br>'.join([''.join(poetry) for poetry in poetrys]),win=u'origin_poem')

                gen_poetries = []
                #分别以这几个字作为诗歌的第一个字，生成8首诗
                for word in list(u'春江花月夜凉如水'):
                    gen_poetry = ''.join(generate(model,word,ix2word,word2ix))
                    gen_poetries.append(gen_poetry)
                vis.text('</br>'.join([''.join(poetry) for poetry in gen_poetries]),win=u'gen_poem')

        t.save(model.state_dict(),'%s_%s.pth' % (opt.model_prefix,epoch))

#接词生成
def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    给定几个词，根据这几个词接着生成一首完整的诗歌
    start_words：u'春江潮水连海平'
    比如start_words 为 春江潮水连海平，可以生成：

    """

    results = list(start_words)
    start_word_len = len(start_words)
    # 手动设置第一个词为<START>
    input = t.Tensor([word2ix['<START>']]).view(1, 1).long()
    if opt.use_gpu: input = input.cuda()
    hidden = None

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)

        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return results

#藏头生成
def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    生成藏头诗
    start_words : u'深度学习'
    生成：
        深木通中岳，青苔半日脂。
        度山分地险，逆浪到南巴。
        学道兵犹毒，当时燕不移。
        习根通古岸，开镜出清羸。
    """
    results = []
    start_word_len = len(start_words)
    input = (t.Tensor([word2ix['<START>']]).view(1, 1).long())
    if opt.use_gpu: input = input.cuda()
    hidden = None

    index = 0  # 用来指示已经生成了多少句藏头诗
    # 上一个词
    pre_word = '<START>'

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = (input.data.new([word2ix[word]])).view(1, 1)

    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]

        if (pre_word in {u'。', u'！', '<START>'}):
            # 如果遇到句号，藏头的词送进去生成

            if index == start_word_len:
                # 如果生成的诗歌已经包含全部藏头的词，则结束
                break
            else:
                # 把藏头的词作为输入送入模型
                w = start_words[index]
                index += 1
                input = (input.data.new([word2ix[w]])).view(1, 1)
        else:
            # 否则的话，把上一次预测是词作为下一个词输入
            input = (input.data.new([word2ix[w]])).view(1, 1)
        results.append(w)
        pre_word = w
    return results

#使用fire模块，一命令行方式运行相应方法
def gen(**kwargs):
    """
    提供命令行接口，用以生成相应的诗
    """
    for k, v in kwargs.items():
        setattr(opt, k, v)

    #加载数据与模型
    data, word2ix, ix2word = get_data(opt)
    model = PoetryModel(len(word2ix), 128, 256);
    map_location = lambda s, l: s
    state_dict = t.load(opt.model_path, map_location=map_location)
    model.load_state_dict(state_dict)

    if opt.use_gpu:
        model.cuda()

    # python2和python3 字符串兼容
    if sys.version_info.major == 3:
        if opt.start_words.isprintable():
            start_words = opt.start_words
            prefix_words = opt.prefix_words if opt.prefix_words else None
        else:
            start_words = opt.start_words.encode('ascii', 'surrogateescape').decode('utf8')
            prefix_words = opt.prefix_words.encode('ascii', 'surrogateescape').decode(
                'utf8') if opt.prefix_words else None
    else:
        start_words = opt.start_words.decode('utf8')
        prefix_words = opt.prefix_words.decode('utf8') if opt.prefix_words else None

    start_words = start_words.replace(',', u'，') \
        .replace('.', u'。') \
        .replace('?', u'？')

    gen_poetry = gen_acrostic if opt.acrostic else generate
    result = gen_poetry(model, start_words, ix2word, word2ix, prefix_words)
    result = ''.join(result)
    return result

#测试
# if __name__ == '__main__':
#
#     # import fire
#     #
#     # fire.Fire()
#     #python main.py gen - -model - path = 'checkpoints/tang_199.pth' - -pickle - path = 'tang.npz' - -start - words = '深度学习' - -prefix - words = '江流天地外，山色有无中。' - -acrostic = True - -nouse - gpu
#     setting = {'model_path':'checkpoints/tang_199.pth','pickle_path':'tang.npz','start_words':'深度学习','prefix_words':'可怜九月初三夜，露似珍珠月似弓。','acrostic':True,'nouse_gpu':'True'}
#     gen(**setting)
