"""
    Author:         Aiden Stevenson Bradwell
    Date:           2021-11-17
    Affiliation:    University of Ottawa, Ottawa, Ontario, Canada

    Content:
        This file contains a method which takes an image as an input, and exports the detected numbers within it.
        The user can use this method by executing the file (using python command)
        and providing the name of the image in question (WITHOUT FILE EXTENSIONS OF .png or .jpg

"""

# STEP 0: Disable TensorFlow Notifications
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2 as cv2
import numpy as np


def get_4_largest(cont):
    if len(cont) < 4:
        raise ValueError("Catch error, not enough bounding boxes found")

    all_c = []
    for c in cont:
        area = cv2.contourArea(c)
        all_c.append([area, c])

    sorted_contours = sorted(all_c, key=lambda x: x[0])
    largest_areas = np.array(sorted_contours, dtype=object)[:, 1][-4:]

    furthest_left = []
    for c in largest_areas:
        (x, y, w, h) = cv2.boundingRect(c)
        furthest_left.append([x, c])

    sorted_left = np.array(sorted(furthest_left, key=lambda x: x[0]), dtype=object)[:, 1]

    return sorted_left


def draw_bounding_boxes(frame, conts):
    for cont in conts:
        (x, y, w, h) = cv2.boundingRect(cont)
        cv2.rectangle(frame, (x, y), (x + w + 5, y + h + 5), (0, 255, 0), 3)

    return frame


def seperate_numbers(frame, conts):
    """
        This method takes an image, and the location of the 4 numbers within it (in the form of contours)
        The method then takes each number separately and sends it to a helper method to detect which number it is.

    :param frame: Frame to be processed
    :param conts: Contours of found numbers
    :return: Value of the numbers in the image
    """

    # STEP 9: Take the bounding boxes of each number
    (x1, y1, w1, h1) = cv2.boundingRect(conts[0])
    (x2, y2, w2, h2) = cv2.boundingRect(conts[1])
    (x3, y3, w3, h3) = cv2.boundingRect(conts[2])
    (x4, y4, w4, h4) = cv2.boundingRect(conts[3])

    # STEP 10: Slice the image into its 4 respective numbers
    number_one = frame[y1 - 15:y1 + h1 + 15, x1 - 15:x1 + w1 + 15]
    number_two = frame[y2 - 15:y2 + h2 + 15, x2 - 15:x2 + w2 + 15]
    number_three = frame[y3 - 15:y3 + h3 + 15, x3 - 15:x3 + w3 + 15]
    number_four = frame[y4 - 15:y4 + h4 + 15, x4 - 15:x4 + w4 + 15]

    # STEP 11: Send the sliced numbers to be detected
    first_val = detect_number(number_one)
    second_val = detect_number(number_two)
    third_val = detect_number(number_three)
    fourth_val = detect_number(number_four)

    return "{}{}{}{}".format(first_val, second_val, third_val, fourth_val)


pb = tf.keras.models.load_model("saved_model\\writing_model")


def detect_number(loc_img):
    """
        Given a frame of only a number, preprocess and determine what value it is

    :param loc_img: cut-out number to be determined
    :return: Value of the number in question
    """

    # STEP 12: Initialize variables for image detection
    loc_kernal = (5, 5)
    loc_high = 204
    loc_low = 177

    # STEP 13: Pre-process numbers for the image-detection process
    loc_mod_img = cv2.GaussianBlur(loc_img, loc_kernal, cv2.BORDER_DEFAULT)
    loc_mod_img = cv2.cvtColor(loc_mod_img, cv2.COLOR_BGR2GRAY)
    loc_canny = cv2.Canny(loc_mod_img, loc_low, loc_high, apertureSize=ap_size)
    loc_canny = cv2.cvtColor(loc_canny, cv2.COLOR_GRAY2RGB)
    np_array = np.array(loc_canny)
    # Normalize size to the same size of the training images (84wx70h)
    normal_frame = cv2.resize(np.copy(np_array), (70, 84), interpolation=cv2.INTER_AREA)
    normal_frame = tf.expand_dims(normal_frame, axis=0)

    # REFERENCE ON How to load a Tensorflow model using OpenCV -
    # https://www.tensorflow.org/tutorials/images/classification

    # STEP 14: Run image through Tensorflow Network to determine image value
    class_values = [8, 5, 4, 9, 1, 7, 6, 3, 2, 0]
    predictions = pb.predict(normal_frame)
    score = tf.nn.softmax(predictions[0])
    class_val = class_values[np.argmax(score)]

    return class_val


if __name__ == "__main__":

    # STEP 1: Take user input and read requested file
    file = input("Please enter a file name (including the file extension): ")
    img = cv2.imread("Images/{}".format(file))

    found_to_be_flawed = [
        363700,
        407396,
        636552,
        642712,
        647784,
        696167,
        879252,
        971376,
        1061154,
        1245480,
        1257522,
        1393770,
        1435800,
        1556388,
        2379636,
        2801676,
        2844756,
        3309528,
        3455299,
        3704364,
        4007250,
        4066296,
        4465898,
        4766328,
        5053335,
        5060276,
        6062723,
        7754859,
        8133352,
        8188644,
    ]

    if img is None:
        print("This file has not been found. Please make sure it is located in partA/Images/")
        exit()

    if int(file.replace(".png", "").replace(".jpg", "")) in found_to_be_flawed:
        print("WARNING: This image has been found to be one of 30 flawed images. Choosing another is suggested!")

    # STEP 2: Initialize variables for image pre-processing
    low = 177
    high = 204
    kernal = (3, 3)
    ap_size = 3
    cont_mode = 1
    retr_mode = 0

    # STEP 3: Pre-process image for canny-edge detection
    mod_img = cv2.GaussianBlur(img, kernal, cv2.BORDER_DEFAULT)
    mod_img = cv2.cvtColor(mod_img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(mod_img, low, high, apertureSize=ap_size)

    # STEP 4: Detect contours of numbers in the image after canny edge detection
    _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # STEP 5: Filter the found contours for the 4 strongest edges (assuming they will be the numbers)
    filtered_contours = []
    try:
        filtered_contours = get_4_largest(contours)

    except ValueError as e:
        print("Apologies! This file was incorrectly read. Please choose another file.")
        exit()

    # STEP 6: Execute TensorFlow Detection
    string_val = seperate_numbers(img, filtered_contours)

    # STEP X: Output detected number
    print("<{}, {}>".format(file.replace(".png", "").replace(".jpg", ""), string_val))

    cv2.waitKey()
