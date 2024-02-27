import numpy as np
import cv2
import json
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch


class sam_process:  # 这个类定义了使用Segment anything模型对图片进行分割处理的方法
    def __init__(self, img_path):
        self.img_path = img_path

    def mask2rle(self, img):
        # 这个函数是为了将SAM对图片进行分割处理后生成的每一个遮罩信息使用RLE编码发送到前端（使用RLE是为了无损传输）
        '''
        img: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        pixels = img.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    def trans_anns(self, anns):
        # 所有的遮罩信息进入这个函数，然后通过上面的mask2rle对每一个遮罩信息处理再整合
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=False)
        list = []
        index = 0
        # 对每个注释进行处理
        for ann in sorted_anns:
            bool_array = ann['segmentation']
            # 将boolean类型的数组转换为int类型
            int_array = bool_array.astype(int)
            # 转化为RLE格式
            rle = self.mask2rle(int_array)
            list.append({"index": index, "mask": rle})
            index += 1
        return list

    def process(self):
        # 调用这个方法可以调用SAM模型对图片进行分割，返回值是经过RLE编码的全部遮罩信息
        image = cv2.imread(self.img_path)

        # sam 模型路径
        sam_checkpoint = 'sam_vit_b_01ec64.pth'
        # 根据下载的模型，设置对应的类型
        model_type = "vit_b"

        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(image)
        # 处理sam返回的图层信息
        mask_list = self.trans_anns(masks)

        mask_obj = {
            "height": image.shape[0],
            "width": image.shape[1],
            "mask_list": mask_list
        }
        torch.cuda.empty_cache()
        return mask_obj
