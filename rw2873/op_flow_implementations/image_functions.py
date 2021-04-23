import os
import cv2

def unpack_video(video_path: str, output_file_name: str):
    cam = cv2.VideoCapture(video_path)

    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except OSError:
        print('Error: Creating director for frames')

    curr_frame = 0

    while True:
        ret, frame = cam.read()

        if ret:
            name = './data/' + output_file_name \
                   + str(curr_frame) + '.jpg'
            print('Creating...' + output_file_name
                  + str(curr_frame) + '.jpg')

            cv2.imwrite(name, frame)

            curr_frame += 1
        else:
            break

    cam.release()
    cv2.destroyAllWindows()