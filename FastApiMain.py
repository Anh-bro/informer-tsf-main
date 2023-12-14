import csv

from pydantic import BaseModel
from fastapi import FastAPI
import time

from exp.exp_informer import Exp_Informer
from utils.tools import dotdict

args = dotdict()

args.model = 'informer'  # model of experiment, options: [informer, informerstack, informerlight(TBD)]
args.data = 'custom'  # data
args.root_path = './data/CustomDataSet/'  # root path of data file
args.data_path = 'dust.csv'  # data file
args.features = 'S'  # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
args.target = 'value'  # target feature in S or MS task
args.freq = 't'  # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
args.checkpoints = './checkpoints'  # location of model checkpoints

# 96 48 48
args.seq_len = 64  # input sequence length of Informer encoder
args.label_len = 32  # start token length of Informer decoder
args.pred_len = 32  # prediction sequence length
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

# 4 4 1
# 1 1 1
args.enc_in = 1  # encoder input size
args.dec_in = 1  # decoder input size
args.c_out = 1  # output size
args.factor = 5  # probsparse attn factor
args.padding = 0  # padding type
args.d_model = 256  # dimension of model
args.n_heads = 4  # num of heads
args.e_layers = 2  # num of encoder layers
args.d_layers = 1  # num of decoder layers
args.d_ff = 256  # dimension of fcn in model
args.dropout = 0.05  # dropout
args.attn = 'prob'  # attention used in encoder, options:[prob, full]
args.embed = 'timeF'  # time features encoding, options:[timeF, fixed, learned]
args.activation = 'gelu'  # activation
args.distil = True  # whether to use distilling in encoder
args.output_attention = False  # whether to output attention in ecoder

args.batch_size = 32
args.learning_rate = 0.00001
args.loss = 'rmse'
args.lradj = 'type1'
args.use_amp = False  # whether to use automatic mixed precision training
args.inverse = False

args.num_workers = 0
args.train_epochs = 10000
args.patience = 3
args.des = 'exp'

# args.use_gpu = True if torch.cuda.is_available() else False
args.use_gpu = True
args.gpu = 0

args.use_multi_gpu = False
args.devices = '0,1,2,3'

args.detail_freq = args.freq
args.freq = args.freq[-1:]

app = FastAPI()

class Item(BaseModel):
    dust_data: list

@app.post("/items/")
async def create_item(item: Item):
    dustList=item.dust_data
    #将dustList列表保存为csv文件 第一列为当前时间时间,列名是Date 第二列为数据，列名为Value
    with open('dust.csv', 'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["date", "value"])
        for i in range(len(dustList)):
            writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), dustList[i]])
    exp=Exp_Informer(args)
    setting="dust_informer_custom_ftS_sl64_ll32_pl32_dm256_nh4_el2_dl1_df256_atprob_fc5_ebtimeF_dtTrue_exp"
    print(exp.predict_custom(setting, True))

    return exp.predict_custom(setting, True).tolist()



