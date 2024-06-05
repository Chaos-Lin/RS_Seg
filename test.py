from models import *
from data_loader import *
from torchvision.utils import save_image
from pathlib import Path
import torch

# 只需要修改下面几个变量即可
super_epoch = 40
dataset_name = 'water'
model_name = 'unetplusplus'
picture_num = 104

# image_name = f'{dataset_name}{picture_num}_image.png'
image_name = "5.jpg"
weight_path = f'model_save_dir/{model_name}/{dataset_name}/{model_name}-{dataset_name}-{super_epoch}.pth'
args = {
    'model_name': model_name,
    'dataset_name': dataset_name
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = AMIO(args).to(device)
if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path))
    print('successful load weight')
else:
    print('not successful load weight')

# image_path = f'../Dataset/{dataset_name}/JPEGImages'
image_path = "C:\\Users\\Chaos Lin\\Desktop"
# segment_path = f'../Dataset/{dataset_name}/SegmentationClass'

input_image = os.path.join(image_path, image_name)
# segment_image = os.path.join(segment_path, image_name)

img = keep_image_size_open(input_image)
img = transform(img).to(device)
img_data = torch.unsqueeze(img, dim=0)

out = net(img_data)
out = torch.squeeze(out)
threshold = 0.5
out = torch.where(out > threshold, torch.tensor(1.0), torch.tensor(0.0))

# seg_img = keep_image_size_open(segment_image)
# seg_img = transform(seg_img).to(device)

img_result = torch.stack([img,
                          # seg_img,
                          out],dim=0)
save_dir = f'result/{model_name}/{dataset_name}'
Path(save_dir).mkdir(parents=True, exist_ok=True)
save_file = os.path.join(save_dir, f'{image_name}')
save_image(img_result,save_file)