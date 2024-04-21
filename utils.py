from PIL import Image

def keep_image_size_open(path,size=(256,256)):
    img = Image.open(path)
    # 打开图片
    temp = max(img.size)
    # 获取长边
    mask = Image.new('RGB',(temp,temp),(0,0,0))
    # 创建一个长边的四角形
    mask.paste(img,(0,0))
    # 将图片粘到四边形上
    mask = mask.resize(size)
    # 再把图片缩放
    return mask
