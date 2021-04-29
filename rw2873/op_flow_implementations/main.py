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
import rw2873_extra_credit_project as ec

def main():
    frame_1 = image_functions.open_image('./sources/images/marple13_20.jpg')
    frame_2 = image_functions.open_image('./sources/images/marple13_21.jpg')

    ec.ec_proj_main()

    return 0


if __name__ == '__main__':
    main()
