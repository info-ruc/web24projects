from PIL import Image
import os
import json
from tqdm import tqdm

def create_data_lists(train_folders, test_folders, min_size, output_folder):
    """
    创建训练集和测试集列表文件.
        参数 train_folders: 训练文件夹集合; 各文件夹中的图像将被合并到一个图片列表文件里面
        参数 test_folders: 测试文件夹集合; 每个文件夹将形成一个图片列表文件
        参数 min_size: 图像宽、高的最小容忍值
        参数 output_folder: 最终生成的文件列表,json格式
    """
    print("\n正在扫描图片,检验图片完整性并创建文件列表.. 请耐心等待.\n")
    train_images = list()
    for d in train_folders:
        for root,dirs,files in tqdm( os.walk(d) ):
            for name in files :
                img_path = os.path.join( root , name )
                try :
                    img = Image.open( img_path , mode = 'r' ) 
                    img.load()
                    if img.width >= min_size and img.height >= min_size:
                        train_images.append(img_path)
                except Exception as e:
                    print( e , ", 文件路径为: " , img_path )
                    continue
    print("训练集中共有 %d 张图像\n" % len(train_images))
    with open(os.path.join(output_folder, 'train_images.json'), 'w') as j:
        json.dump(train_images, j)

    for d in test_folders:
        test_images = list()
        test_name = d.split("/")[-1]
        for root,dirs,files in tqdm( os.walk(d) ):
            for name in files :
                img_path = os.path.join( root , name )
                try :
                    img = Image.open( img_path , mode = 'r' ) 
                    img.load()
                    if img.width >= min_size and img.height >= min_size:
                        test_images.append(img_path)
                except Exception as e:
                    print( e , ", 文件路径为: " , img_path )
                    continue
        print("在测试集 %s 中共有 %d 张图像\n"%(test_name, len(test_images)))
        with open(os.path.join(output_folder, test_name + '_test_images.json'),'w') as j:
            json.dump(test_images, j)

    print("生成完毕。训练集和测试集文件列表已保存在 %s 下\n" % output_folder)


if __name__ == '__main__':
    create_data_lists(train_folders=['/mnt/data/izumihanako/MyCourses/WebInfoProcess/data/TrainSet'],
                      test_folders=['/mnt/data/izumihanako/MyCourses/WebInfoProcess/data/TestSet/·85'],
                      min_size=100,
                      output_folder='/mnt/data/izumihanako/MyCourses/WebInfoProcess/data')
