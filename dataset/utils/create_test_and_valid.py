import os
import math
import random
import shutil

ROOT_DIR = 'C:/Users/Luke/Desktop/coding/unmix_guitar_separation/dataset/data/'
GUITAR = os.listdir(ROOT_DIR+'guitar')
INTERFERERS = os.listdir(ROOT_DIR+'interferers')

# Move 20% of guitar files to valid set
r = math.floor(len(GUITAR) * 0.2)
for i in range(r):
    random_file_index = math.floor(random.random() * len(GUITAR))
    shutil.move(ROOT_DIR+'guitar/'+GUITAR[random_file_index], ROOT_DIR+'/valid/guitar/'+GUITAR[random_file_index])
    GUITAR = os.listdir(ROOT_DIR+'guitar')

# Move remaining guitar files to test
for file in GUITAR:
    shutil.move(ROOT_DIR+'guitar/'+file, ROOT_DIR+'/train/guitar/'+file)


# Move 20% of interferer files to valid set
r = math.floor(len(INTERFERERS) * 0.2)
for i in range(r):
    random_file_index = math.floor(random.random() * len(INTERFERERS))
    shutil.move(ROOT_DIR+'interferers/'+INTERFERERS[random_file_index], ROOT_DIR+'/valid/interferers/'+INTERFERERS[random_file_index])
    INTERFERERS = os.listdir(ROOT_DIR+'interferers')

# Move remaining interferer files to test
for file in INTERFERERS:
    shutil.move(ROOT_DIR+'interferers/'+file, ROOT_DIR+'/train/interferers/'+file)