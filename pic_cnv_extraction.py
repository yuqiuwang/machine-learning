# -*- coding: utf-8 -*-
"""
Author  : yuqiuwang
Mail    : yuqiuwang929@gmail.com
Created : 2018/7/22 11:28
"""


import re
import os
import sys
import optparse

import tensorflow as tf
from scipy import misc


# 读取一张染色体图片 800X1280
# 根据CNV坐标，切割出CNV的位置，并压缩为 300X300 像素的图片
# 用法 python pic_cnv_extract.py -i ./sample1_chr4.png -o ./sample1_4_37015354_37098608.png -p chr4,37015354,37098608


class RefError(Exception):
    # 定义了一个异常处理类
    def __init__(self,err='reference error!'):
        Exception.__init__(self,err)


class CnvExtract:

    # 该类通过CNV的起始、终止位置，定位在散点图中的位置
    # 通过main函数，返回像数的横坐标

    def __init__(self, chr_num, start, end, hg_length="hg19"):
        self.chr_num = chr_num
        self.start = start
        self.end = end

        if hg_length == "hg18":
            self.hg_length = {"chr1": 247249719, "chr2": 242951149, "chr3": 199501827, "chr4": 191273063,
                            "chr5": 180857866, "chr6": 170899992, "chr7": 158821424, "chr8": 146274826,
                            "chr9": 140273252, "chr10": 135374737, "chr11": 134452384, "chr12": 132349534,
                            "chr13": 114142980, "chr14": 106368585, "chr15": 100338915, "chr16": 88827254,
                            "chr17": 78774742, "chr18": 76117153, "chr19": 63811651, "chr20": 62435964,
                            "chr21": 46944323, "chr22": 49691432, "chrx": 154913754, "chry": 57772954}

        elif hg_length == "hg19":
            self.hg_length = {"chr1": 249250621, "chr2": 243199373, "chr3": 198022430, "chr4": 191154276,
                            "chr5": 180915260, "chr6": 171115067, "chr7": 159138663, "chr8": 146364022,
                            "chr9": 141213431, "chr10": 135534747, "chr11": 135006516, "chr12": 133851895,
                            "chr13": 115169878, "chr14": 107349540, "chr15": 102531392, "chr16": 90354753,
                            "chr17": 81195210, "chr18": 78077248, "chr19": 59128983, "chr20": 63025520,
                            "chr21": 48129895, "chr22": 51304566, "chrx": 155270560, "chry": 59373566}

        else:
            raise RefError("You should choose hg19 or hg18!")

    def __call__(self, *args, **kwargs):
        return self.main()

    def check_chr(self):
        my_chr = str(self.chr_num).lower()
        if re.findall(r'chr',my_chr):
            pass
        else:
            my_chr = "chr"+my_chr
        if re.findall(r'chr[xy]|chr\d{1,2}', my_chr): pass
        else:
            raise RefError("Chromosome should input as chr1 or 1!")
        if my_chr == "chr23":
            my_chr = "chrx"
        elif my_chr == "chr24":
            my_chr = "chry"
        return self.hg_length[my_chr]

    def main(self):
        # 返回切割位置的坐标
        chr_length = self.check_chr()
        new_start = int(1180 * int(self.start) / chr_length)
        new_end = int(1180 * int(self.end) / chr_length)
        return new_start, new_end


class PicCovert:

    # 该类依赖tensorflow、scipy包
    # 读取康孕染色体散点图，截取CNV区域，缩放后保存为新的图片

    def __init__(self, pic_path, save_path, start, end, save_size=(300, 300)):
        self.pic_path = pic_path
        self.save_path = save_path
        self.save_size = save_size
        self.start = start
        self.end = end

    def __call__(self, *args, **kwargs):
        return self.save_pic()

    def load_pic(self):
        with tf.Session() as sess:
            image_raw_data = tf.gfile.FastGFile(self.pic_path,'rb').read()
            img_data = tf.image.decode_jpeg(image_raw_data)
            img_data = img_data.eval().reshape(800, 1280, 3)
            #img_data = img_data[100:660, 50:1230, :]
            img_data = img_data[100:660, :, :]
            img_data = img_data[:, self.start+50:self.end+70, :]
            img_data = tf.image.resize_images(img_data, self.save_size)
            img_data = img_data.eval()
        return img_data

    def save_pic(self):
        misc.imsave(self.save_path, self.load_pic())


if __name__ == "__main__":
    prog_base = os.path.split(sys.argv[0])[1]
    parser = optparse.OptionParser()
    parser.add_option("-i", "--path", action="store", type="string", dest="pic_path", help="picture path")
    parser.add_option("-o", "--input", action="store", type="string", dest="save_path", help="output_path")
    parser.add_option("-p", "--position", action="store", type="string", dest="pos_info",
                      help="chr position info. chr1,10001,60001")
    parser.add_option("-r", "--refinfo", action="store", type="string", dest="hg_ref",
                      help="hg version, surpport hg18 or hg19, default: hg19")
    parser.add_option("-s", "--size", action="store", type="string", dest="save_size",
                      help="picture save size, default: 300,300")
    (options, args) = parser.parse_args()
    if options.pic_path is None or options.pos_info is None:
        print(prog_base + ": error: missing required command-line argument.")
        parser.print_help()
        sys.exit(0)

    if options.save_path is None:
        Save_path = options.pic_path + ".convert.png"
    else:
        Save_path = options.save_path

    Chr_num, Start, End = options.pos_info.split(",")
    if options.hg_ref is None:
        Start, End = CnvExtract(Chr_num, Start, End)()
    else:
        Start, End = CnvExtract(Chr_num, Start, End, options.hg_ref.lower())()

    if options.save_size is None:
        PicCovert(options.pic_path, Save_path, Start, End)()
    else:
        Save_size = options.save_size.split(",")
        Save_size = tuple([int(x) for x in Save_size])
        PicCovert(options.pic_path, Save_path, Start, End, Save_size)()
