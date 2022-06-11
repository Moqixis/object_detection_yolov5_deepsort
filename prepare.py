import os
import cv2
import xml.etree.ElementTree as ET

 
def main():
    # JPG文件的地址
    img_path = '/data/anhui-ai/lhq/VOCdevkit/VOC2007/JPEGImages/'
    # XML文件的地址
    anno_path = '/data/anhui-ai/lhq/VOCdevkit/VOC2007/Annotations/'
    # 存结果的文件夹
    cut_path = '/data/anhui-ai/lhq/VOCdevkit/VOC2007/train/'
    if not os.path.exists(cut_path):
        os.makedirs(cut_path)
    imagelist = os.listdir(img_path)
    for image in imagelist:
        image_pre, ext = os.path.splitext(image)
        img_file = img_path + image
        img = cv2.imread(img_file)
        xml_file = anno_path + image_pre + '.xml'
 
        tree = ET.parse(xml_file)
        root = tree.getroot()
        obj_i = 0
        for obj in root.iter('object'):
            obj_i += 1
            cls = obj.find('name').text
            xmlbox = obj.find('bndbox')
            b = [int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)),
                 int(float(xmlbox.find('ymax').text))]
            img_cut = img[b[1]:b[3], b[0]:b[2], :]
            path = os.path.join(cut_path, cls)
            # 目录是否存在,不存在则创建
            mkdirlambda = lambda x: os.makedirs(x) if not os.path.exists(x) else True
            mkdirlambda(path)
            # 可能是负样本，加了try判断，只算入了正样本
            try:
                cv2.imwrite(os.path.join(cut_path, cls, '{}_{:0>2d}.jpg'.format(image_pre, obj_i)), img_cut)
            except:
                continue
            print("&&&&")
 
 
if __name__ == '__main__':
    main()
