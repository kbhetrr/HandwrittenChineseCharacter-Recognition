import struct
from codecs import decode
from scipy.misc import toimage
import numpy as np
import PIL.ImageOps


class Reader:
    def load_gnt_file(self, filename):
        """
        Load characters and images from a given GNT file.
        :param filename: The file path to load.
        :return: (image: Pillow.Image.Image, character) tuples
        """
        with open(filename, "rb") as f:
            while True:
                packed_length = f.read(4)
                if packed_length == b'':
                    break
                length = struct.unpack("<I", packed_length)[0]
                raw_label = struct.unpack(">cc", f.read(2))
                width = struct.unpack("<H", f.read(2))[0]
                height = struct.unpack("<H", f.read(2))[0]
                photo_bytes = struct.unpack("{}B".format(height * width), f.read(height * width))
                label = decode(raw_label[0] + raw_label[1], encoding="gb2312")
                image = toimage(np.array(photo_bytes).reshape(height, width))

                yield image, label

    def read_gnt_image(self, path):
        data = self.load_gnt_file(path)
        data_list = []
        while True:
            try:
                image, label = next(data)
                image = image.resize((56, 56)) # 이미지 크기 조절
                #image = PIL.ImageOps.invert(image) # 색상 반전
                data_list.append((image, label))
            except StopIteration:
                break
        return data_list