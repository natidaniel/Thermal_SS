import os



def read_image_list(file_name):
    '''
    Parameters:
    -   file name: a txt file with a list of img's to be used in the following format 
        {SPLIT_FOLDER(train/val etc.)   ABCD1234(IMAGE_PREFIX)  jpg(file extension)}
    RETURNS:
    - A list of SPLIT_FOLDER for each image in the list
    - A list of IMAGE_PREFIX for each image in the list
    '''
    num_lines = sum(1 for _ in open(file_name))
    splits = [None] * num_lines
    files = [None] * num_lines
    extensions = [None] * num_lines
    with open(file_name, 'r') as f:
        for i, line in enumerate(f):
            values = line.split(sep='\t')
            splits[i] = values[0].rstrip()
            files[i] = values[1].rstrip()
            extensions[i] = values[2].rstrip()
    return (splits, files, extensions)



def create_image_list(data_dir, split_dir, file_extension, file_name):
    '''
    Parameters:
    -   data_dir: the path to the main data directory (e.g. 'c:/data')
    -   split_dir: the name of the directory within the data_dir that contains the images of the split (e.g. 'train/rgb')
    -   file_extension: the extension of the image files (e.g. 'jpg') 
    -   file name: a txt file to which a list of img's will be written in the following format 
        {SPLIT_FOLDER(train/val etc.)   ABCD1234(IMAGE_PREFIX)  jpg(file extension)}
    '''
    img_dir = os.path.join(data_dir, split_dir)
    prefix_list = [file for file in os.listdir(img_dir) if file.endswith(file_extension)]
    with open(file_name,'w') as f:
        for i in range(0, len(prefix_list)):
            prefix = os.path.splitext(prefix_list[i])[0]
            fmt_str = "{:5}\t{:40}\t{:100}\n"
            str = fmt_str.format(split_dir, prefix, file_extension)
            f.write(str)



if __name__ == '__main__':
    data_dir = ''
    split_dir = ''
    file_extension = ''
    file_name = ''
    create_image_list(data_dir=data_dir, split_dir=split_dir, file_extension=file_extension, file_name=file_name)