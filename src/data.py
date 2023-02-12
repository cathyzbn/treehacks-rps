from optparse import OptionParser
import os
import cv2
import numpy as np
from glob import glob
from PIL import Image
from helpers import key_equal, get_hand_landmark


NUM_IMAGES = 100

def get_data(path, data_type, start):
    path = path + data_type + '/'
    os.makedirs(path, exist_ok=True)

    print(f"Path: {path}")
    print("Press c to continue, q to quit")

    execute = False
    n = 0
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and n < NUM_IMAGES:
        success, image = cap.read()
        if not success:
            continue
        
        key = cv2.waitKey(1)
        if key_equal(key, 'c'):
            execute = True
        elif key_equal(key, 'q'):
            break

        cv2.imshow(f'Data gathering for {data_type}', image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if execute:
            image_file = Image.fromarray(image)
            image_file.save(f"{path}{n+start}.png")
            n += 1

    print(f"{n} images saved!")
    cap.release()


def get_data_from_file(path):
    data = dict()
    for t in ['rock', 'paper', 'scissors']:
        data[t] = (glob(f"{path}{t}/*.png"))

    n_samples = len(data['paper']) + len(data['scissors']) + len(data['rock'])
    X = np.zeros((n_samples, 21*3))
    y = np.zeros(n_samples)

    i = 0
    temp_count = 0
    num_images = []
    for label, t in enumerate(['paper', 'scissors', 'rock']):
        for d in data[t]:
            lm = get_hand_landmark(d)
            if lm is not None:
                X[i] = lm
                y[i] = label
                i += 1
                temp_count += 1
            num_images.append(temp_count)
            print(f"{temp_count} valid {t} images")
        temp_count = 0

    X = X[:i, :]
    y = y[:i]
    return X, y

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-t", "--type", dest="type", default=None, 
                      type='string', help='type of data to generate')
    parser.add_option('-s', '--start', dest='start', default=0,
                      type='int', help='data filename start number')
    
    option, args = parser.parse_args()
    if option.type is None:
        parser.error("Please add a type")
    if option.type not in ['rock', 'paper', 'scissors']:
        parser.error('Input valid data type')

    path = 'data/'
    
    get_data(path, option.type, option.start)
