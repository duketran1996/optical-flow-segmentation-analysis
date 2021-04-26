#########################################
#   Optical Flow Implementations        #
#   CS-GY 6643, Computer Vision         #
#   Professor James Fishbaugh           #
#   Author:        Russell Wustenberg   #
#   Collaborators: Duc Tran             #
#                  Andrew Weng          #
#                  Ye Xu                #
#########################################

import Lucas_Kanade as LK
import image_functions

def main():
    frame_1 = image_functions.open_image('./sources/images/marple13_20.jpg')
    frame_2 = image_functions.open_image('./sources/images/marple13_21.jpg')

    op_flow = LK.flow(frame_1, frame_2)

    image_functions.display_image(op_flow)

    return 0


if __name__ == '__main__':
    main()
