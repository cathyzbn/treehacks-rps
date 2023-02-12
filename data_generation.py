# Copyright @2021 Ruining Li. All Rights Reserved.

from optparse import OptionParser
import os
import cv2
from PIL import Image
import keyboard

def main():
    parser = OptionParser()
    (options, args) = parser.parse_args()
    for type_ in ['paper', 'scissors', 'rock']:
        path = "dataset/" + type_
        os.makedirs(path, exist_ok=True)
        print("Ready to generate data in dataset/" + type_ + " folder.")
        print("Press the Enter key to start taking pictures")
        print("or press the Esc key to quit.")

        cap = cv2.VideoCapture(0)
        start_taking_pictures = False
        num_images_taken = 0
        NUM_IMAGES_REQUIRED = 100
        while cap.isOpened() and num_images_taken < NUM_IMAGES_REQUIRED:
            success, image = cap.read()
            if not success:
                continue
            cv2.imshow('Camera', image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if not start_taking_pictures and keyboard.is_pressed('enter'):
                start_taking_pictures = True
            if start_taking_pictures:
                image_file = Image.fromarray(image)
                image_file.save(path + "/" + str(num_images_taken + (0 if len(args) == 0 else int(args[0]))) + ".png")
                num_images_taken += 1
            if cv2.waitKey(5) & 0xFF == 27:
                break

        print(str(num_images_taken), "images saved!")
        cap.release()

if __name__ == "__main__":
    main()
