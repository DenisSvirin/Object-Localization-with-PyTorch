imopr

def extract_info_from_xml(image, info):
    """
    image: path to the image
    info: path to the xml file of the image
    ------
    return: filename, width, height,
     class_num, x_min, y_min, x_max, y_max
    """
    file = minidom.parse(info)

    height, width = cv2.imread(image).shape[:2]

    xmin = file.getElementsByTagName("xmin")
    x1 = float(xmin[0].firstChild.data)

    ymin = file.getElementsByTagName("ymin")
    y1 = float(ymin[0].firstChild.data)

    xmax = file.getElementsByTagName("xmax")
    x2 = float(xmax[0].firstChild.data)

    ymax = file.getElementsByTagName("ymax")
    y2 = float(ymax[0].firstChild.data)

    class_name = file.getElementsByTagName("name")[0].firstChild.data
    class_num = 1 if class_name == "cat" else 0

    files = file.getElementsByTagName("filename")
    filename = files[0].firstChild.data

    return filename, width, height, class_num, x1, y1, x2, y2

def xml_to_csv(image_directory, info_directory):
    """
    return: DataFrame with all data recieved from
    extract_info_from_xml function
    """
    info_files = os.listdir(info_directory)
    img_files = os.listdir(image_directory)

    data = []
    for _img_file, _info_file in zip(img_files, info_files):
        img_path = image_dir + "/" + _img_file
        info_path = info_dir + "/" + _info_file

        values = extract_info_from_xml(img_path, info_path)

        data.append(values)

    columns = [
        "filename",
        "width",
        "height",
        "class_num",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]

    df = pd.DataFrame(data, columns=columns)

    return df
