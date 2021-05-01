#   CS-GY 6643 Computer Vision
#   Professor James Fishbaugh
#   Russell Wustenberg (rw2873)
#   Extra Credit Project
#   May 2021

import Lucas_Kanade as LK
import Horne_Schunk as HS
import image_functions
import draw


def ec_proj_main():
    im_sphere_0 = image_functions.open_image('./sources/ec_project/sphere0.png')
    im_sphere_1 = image_functions.open_image('./sources/ec_project/sphere1.png')
    im_traffic_0 = image_functions.open_image('./sources/ec_project/traffic0.png')
    im_traffic_1 = image_functions.open_image('./sources/ec_project/traffic1.png')

    # Output these for paper examples
    im_sphere_0_dx = image_functions.get_derivative_x(im_sphere_0)
    im_sphere_0_dy = image_functions.get_derivative_y(im_sphere_0)
    im_sphere_0_dt = im_sphere_0 - im_sphere_1

    op_flow = LK.flow(im_traffic_0, im_traffic_1, 15, True)
    hsv = draw.draw_flow_hsv(op_flow)
    image_functions.display_image(hsv)

    op_flow = HS.flow(im_traffic_0, im_traffic_1, 1.0)
    hsv = draw.draw_flow_hsv(op_flow)
    image_functions.display_image(hsv)


    return