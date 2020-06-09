
class Config(object):
    data_path = 'data/' #诗歌的文本文件存放路径
    pickle_path = 'tang.npz' #预处理好的二进制诗歌集
    author = None #只学习某位作者的诗歌
    constrain = 'poet.tang' #诗歌类别，唐诗还是宋词
    lr = 1e-3
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

