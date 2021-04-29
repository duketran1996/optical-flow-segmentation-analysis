#   CS-GY 6643 Computer Vision
#   Professor James Fishbaugh
#   Russell Wustenberg (rw2873)
#   Extra Credit Project
#   May 2021

import Lucas_Kanade as LK
import image_functions
import draw


def ec_proj_main():
    im_sphere_0 = image_functions.open_image('./sources/ec_project/sphere0.png')
    im_sphere_1 = image_functions.open_image('./sources/ec_project/sphere1.png')

    im_sphere_0_dx = image_functions.get_derivative_x(im_sphere_0)
    im_sphere_0_dy = image_functions.get_derivative_y(im_sphere_0)
    im_sphere_0_dt = im_sphere_0 - im_sphere_1

    op_flow = LK.flow(im_sphere_1, im_sphere_0, 5)
    out_im = draw.draw_flow_intensity(op_flow)

    image_functions.display_image(out_im)

    draw.flow_arrows(im_sphere_1, op_flow)


    return