# Original Author       : Ghasem Abdi, ghasem.abdi@yahoo.com
# File Last Update Date : April 15, 2022

import os
import re
import glob
import shutil
import imghdr
import natsort
import image_slicer
from PIL import Image

def prepare_tiles(fnames):
    i = 0
    tiles = []
    for fname in fnames:
        pos = image_slicer.get_image_column_row(fname)
        img = Image.open(fname)

        count, position_xy = 0, [0, 0]
        for a, b in zip(pos, img.size):
            position_xy[count] = a * b
            count += 1
        
        tiles.append(image_slicer.Tile(image=img, position=pos, number=i+1, coords=position_xy, filename=fname))
        i += 1
    return tiles

class chunk_data:
    def __init__(self, number_tiles: int):
        super(chunk_data, self).__init__()
        self.number_tiles = number_tiles

    def chunk(self, data_root, save_directory=None):
        if save_directory is None:
            save_directory = os.path.join(data_root, 'chunk_data')
        
        if os.path.isdir(save_directory):
            shutil.rmtree(path=save_directory)
            
        shutil.copytree(src=data_root, dst=save_directory, \
            ignore=(lambda data_root, files: [f for f in files if os.path.isfile(os.path.join(data_root, f))]))

        filenames = glob.glob(pathname=os.path.join(data_root, '**/*'), recursive=True)
        for filename in filenames:
            try:
                if imghdr.what(file=filename):
                    image_slicer.save_tiles(tiles=image_slicer.slice(filename=filename, number_tiles=self.number_tiles, \
                        save=False), prefix=os.path.splitext(os.path.basename(filename))[0], directory=os.path.join(save_directory, \
                            os.path.dirname(os.path.relpath(filename, data_root))), format=os.path.splitext(os.path.basename(filename))[1].replace('.', ''))
            
            except:
                pass
        return save_directory
    
    def dechunk(self, data_root, pattern=re.compile('_\d{2}_\d{2}\.'), save_directory=None, del_chunk=True):
        if save_directory is None: 
            save_directory = data_root
            
        fnames = natsort.natsorted(glob.glob(os.path.join(data_root, '*.*')))

        for i in range(0, len(fnames), self.number_tiles):
            tiles = prepare_tiles(fnames[i:i+self.number_tiles])
            image = image_slicer.join(tiles)
            image.save(os.path.join(save_directory, pattern.sub('.', os.path.basename(fnames[i]))))

        [os.remove(fname) for fname in fnames if del_chunk]
        return save_directory