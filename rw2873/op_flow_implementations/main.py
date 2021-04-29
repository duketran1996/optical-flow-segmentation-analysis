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
import draw

def main():
    frame_1 = image_functions.open_image('./sources/images/marple13_20.jpg')
    frame_2 = image_functions.open_image('./sources/images/marple13_21.jpg')

    op_flow = LK.flow(frame_1, frame_2, 5)
    out_im = draw.draw_flow_intensity(op_flow)

    image_functions.display_image(out_im)

    return 0


if __name__ == '__main__':
    main()
