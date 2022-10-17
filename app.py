from upcunet_v3 import RealWaifuUpScaler
import argparse
import gradio as gr
import time
import logging
import os
from PIL import ImageOps
import numpy as np
import math

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    return parser.parse_args()




def greet(input_img, input_model_name, input_tile_mode):
    # if input_img.size[0] * input_img.size[1] > 256 * 256:
    #     y = int(math.sqrt(256*256/input_img.size[0]*input_img.size[1]))
    #     x = int(input_img.size[0]/input_img.size[1]*y)
    #     input_img = ImageOps.fit(input_img, (x, y))
    input_img = np.array(input_img)
    if input_model_name not in model_cache:
        t1 = time.time()
        upscaler = RealWaifuUpScaler(input_model_name[2], ModelPath + input_model_name, half=False, device="cpu")
        t2 = time.time()
        logger.info(f'load model time, {t2 - t1}')
        model_cache[input_model_name] = upscaler
    else:
        upscaler = model_cache[input_model_name]
        logger.info(f'load model from cache')

    start = time.time()
    result = upscaler(input_img, tile_mode=input_tile_mode)
    end = time.time()
    logger.info(f'input_model_name, {input_model_name}')
    logger.info(f'input_tile_mode, {input_tile_mode}')
    logger.info(f'input shape, {input_img.shape}')
    logger.info(f'output shape, {result.shape}')
    logger.info(f'speed time, {end - start}')
    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s")
    logger = logging.getLogger()
    args = parse_args()
    ModelPath = "weights_v3/"
    model_cache = {}

    input_model_name = gr.inputs.Dropdown(os.listdir(ModelPath), default="up2x-latest-denoise2x.pth", label='选择model')
    input_tile_mode = gr.inputs.Dropdown([0, 1, 2, 3, 4], default=2, label='选择tile_mode')
    input_img = gr.inputs.Image(label='image', type='pil')

    inputs = [input_img, input_model_name, input_tile_mode]
    outputs = "image"
    iface = gr.Interface(fn=greet,
                         inputs=inputs,
                         outputs=outputs,
                         allow_screenshot=False,
                         allow_flagging='never',
                         examples=[['test-img.jpg', "up2x-latest-denoise2x.pth", 2]],
                         article='[https://github.com/bilibili/ailab/tree/main/Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN)<br>'
                                 '感谢b站开源的项目，图片过大会导致内存不足，所有我将图片裁剪小，想体验大图片的效果请自行前往上面的链接。<br>'
                                 '修改bbb'
                                 'The large image will lead to memory limit exceeded. So I crop and resize image. '
                                 'If you want to experience the large image, please go to the link above.')
    iface.launch(
      enable_queue=args.enable_queue
    )
