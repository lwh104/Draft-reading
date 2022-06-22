# @Author: Li
# @FileName: calculate_fps.py
# @Time: 2022/1/30 12:41
import paddle

from paddleseg.models import U2Netp
from paddleseg.models import UNet
from paddleseg.models import CA_all_U2NetP
import numpy as np

# from paddleseg.models.ca_unet import CA_UNet

unet = UNet(num_classes=3)
u2netp = U2Netp(num_classes=3)
# unet_ca = CA_UNet(num_classes=3)
u2netp_ca = CA_all_U2NetP(num_classes=3)


