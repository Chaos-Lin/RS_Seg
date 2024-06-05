import os
import random

# 分三级目录，如A/B/a.jpg
# input_path为一级目录
def creat_filelist(input_path, classes):
    # 创建三级目录
    # index 一定是str类型，不可以为int
    dir_image1 = []  # 二级目录
    file_list = []  # 三级目录
    for index, name in enumerate(classes):
        print('index', index)
        index_str = str(index)
        dir_image1_temp = input_path + '\\' + name + '\\'
        for dir2 in os.listdir(dir_image1_temp):
            dir_image2_temp = os.path.join(dir_image1_temp, dir2)
            dir_image2_temp1 = dir_image2_temp +' ' + index_str
            file_list.append(dir_image2_temp1)

    return dir_image1, file_list


def creat_txtfile(output_path, file_list):
    with open(output_path, 'w') as f:
        for list in file_list:
            print(list)
            f.write(str(list) + '\n')


def main():
    dir_image0 = r'D:\Search\RS\Dataset\UCMerced\Images'
    dir_image1 = os.listdir(dir_image0)
    classes = dir_image1
    print(classes)
    dir_list, file_list = creat_filelist(dir_image0, classes)
    print(file_list[0:3])
    output_path = r'D:\Search\RS\Dataset\UCMerced'
    random.shuffle(file_list)
    train_ratio = 0.7
    val_ratio = 0.2
    # test_ratio = 0.1
    train_size = int(train_ratio * len(file_list))
    val_size = int(val_ratio * len(file_list))

    train_set = file_list[:train_size]
    eval_set = file_list[train_size:train_size + val_size]
    test_set = file_list[train_size + val_size:]

    train_path = os.path.join(output_path, 'train.txt')
    test_path = os.path.join(output_path, 'test.txt')
    eval_path = os.path.join(output_path, 'eval.txt')

    creat_txtfile(train_path, train_set)
    creat_txtfile(test_path, test_set)
    creat_txtfile(eval_path, eval_set)



if __name__ == '__main__':
    main()