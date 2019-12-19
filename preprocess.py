import re
import jieba
import codecs
import os


# remove chinese characters
def clean_str(string):
    string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


# get dataset
def get_data(original_path, save_path='all_email.txt'):
    files = os.listdir(original_path)
    for file in files:
        if os.path.isdir(original_path + '/' + file):
            get_data(original_path + '/' + file, save_path=save_path)
        else:
            email = ''
            f = codecs.open(original_path + '/' + file, 'r', 'gbk', errors='ignore')
            for line in f:
                line = clean_str(line)
                email += line
            f.close()
            f = open(save_path, 'a', encoding='utf8')
            email = [word for word in jieba.cut(email) if word.strip() != '']
            f.write(' '.join(email) + '\n')


# get labels
def get_label(original_path, save_path='all_email.txt'):
    f = open(original_path, 'r')
    label_list = []
    for line in f:
        if line[0] == 's':
            label_list.append('0')
        elif line[0] == 'h':
            label_list.append('1')

    f = open(save_path, 'w', encoding='utf8')
    f.write('\n'.join(label_list))
    f.close()


print('Storing emails...')
get_data('../', save_path='all_email.txt')
print('Storage accomplished')
print('Storing labels...')
get_label('full/index', save_path='label.txt')
print('Storage accomplished')
