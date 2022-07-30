# %%
import base64
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import re
import torch

from plotly.subplots import make_subplots
from tqdm.auto import tqdm
from transformers import DPTFeatureExtractor, DPTForDepthEstimation, DPTModel

# %%


def decode_image(src):
    """
    解码图片
    :param src: 图片编码
        eg:
            src="data:image/gif;base64,R0lGODlhMwAxAIAAAAAAAP///
                yH5BAAAAAAALAAAAAAzADEAAAK8jI+pBr0PowytzotTtbm/DTqQ6C3hGX
                ElcraA9jIr66ozVpM3nseUvYP1UEHF0FUUHkNJxhLZfEJNvol06tzwrgd
                LbXsFZYmSMPnHLB+zNJFbq15+SOf50+6rG7lKOjwV1ibGdhHYRVYVJ9Wn
                k2HWtLdIWMSH9lfyODZoZTb4xdnpxQSEF9oyOWIqp6gaI9pI1Qo7BijbF
                ZkoaAtEeiiLeKn72xM7vMZofJy8zJys2UxsCT3kO229LH1tXAAAOw=="

    :return: str 保存到本地的文件名
    """
    # 1、信息提取
    result = re.search(
        "data:image/(?P<ext>.*?);base64,(?P<data>.*)", src, re.DOTALL)
    if result:
        ext = result.groupdict().get("ext")
        data = result.groupdict().get("data")

    else:
        raise Exception("Do not parse!")

    # 2、base64解码
    img = base64.urlsafe_b64decode(data)

    # 3、二进制文件保存
    filename = "{}.{}".format('latest', ext)
    with open(filename, "wb") as f:
        f.write(img)

    return filename

# %%


feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
down_sample = 2


def _hex_rgb(rgb):
    return '#' + ''.join([hex(e).replace('x', '')[-2:] for e in rgb])


def analysis_image(file_name):
    #
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)[:, :, :3]
    print('Image shape is {}'.format(img.shape))

    #
    inputs = feature_extractor(img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_depth = outputs.predicted_depth
    predicted_depth = predicted_depth.squeeze().cpu().numpy()
    depth_map = cv2.resize(predicted_depth, (img.shape[1], img.shape[0]))

    #
    points = []
    down_sample = max(1, int(np.max(depth_map.shape[:2]) / 1000))

    #
    fig_depth = px.imshow(depth_map[::-down_sample, ::down_sample])
    # fig_depth.update_layout(
    #     # width=400,
    #     margin={'t': 10},
    # )
    print('Done with depth')

    #
    for x in tqdm(range(0, depth_map.shape[0], down_sample), '{}'.format(down_sample)):
        for y in range(0, depth_map.shape[1], down_sample):
            # points.append((x/down_sample, y/down_sample,
            #                max(0, (depth_map[x, y]-20) * 10), _hex_rgb(img[x, y])))
            points.append((x/down_sample, y/down_sample,
                           depth_map[x, y] * 4, _hex_rgb(img[x, y])))

    table = pd.DataFrame(points, columns=['x', 'y', 'depth', 'color'])
    table['size'] = 1

    #
    fig_cloud = px.scatter_3d(table, x='x', y='y', z='depth',
                              size='size', size_max=down_sample * 2)

    fig_cloud.data[0]['marker']['color'] = table['color']
    fig_cloud.data[0]['marker']['line'] = dict(width=0)

    # fig_cloud.update_layout(
    #     # width=800,
    #     margin={'t': 10},
    #     scene=dict(
    #         aspectmode='data',
    #     )
    # )

    print('Done with cloud')

    fig1 = make_subplots(rows=1, cols=2, specs=[
                         [{'type': 'scene'}, {'type': 'xy'}]])
    fig1.add_trace(fig_cloud.data[0], row=1, col=1)
    fig1.add_trace(fig_depth.data[0], row=1, col=2)

    fig1.update_layout({'margin': {'t': 5}, 'scene': {'aspectmode': 'data'}})

    return fig_depth, fig_cloud, fig1

# %%
