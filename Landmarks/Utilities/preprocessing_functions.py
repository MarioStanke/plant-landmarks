import numpy as np
from PIL import Image
from . import affine_math_functions as amf
from sklearn.linear_model import LinearRegression

def affine_rotation_from_scale_data(img_dimension, scale_data, top_offset = 300):
    """
    Calculate rotational affinity for given scale points of an image.
    """
    
    left = scale_data['left']
    right = scale_data['right']
    x1 = left[0]
    y1 = left[1]
    x2 = right[0]
    y2 = right[1]
    
    l = np.array([x1,y1])
    r = np.array([x2,y2])
    
    angle = 0

    if (y2 != y1):

        alpha = np.arctan((y2-y1) / np.abs(x2-x1)) #calculate the angle
        angle = amf.rad_to_deg(alpha) #radian measure to degree
        angle = - angle #set sign to rotate accordingly (rotate function rotates counter-clockwise)

    middle = np.array( [img_dimension[0] / 2, img_dimension[1] / 2] ) #calculate the middle of the image, 
                                                                      # because rotate function rotates 
                                                                      # around the middle point
    return angle, middle



def crop_from_scale_affinity(img, angle, middle, scale_data, scale_padding_factor, top_offset = 300):
    """
    Use the rotation from affine_rotation_from_scale_data to determine a box in order to cut off the clothes-pegs.
    """
    
    l = np.array(scale_data['left'])
    r = np.array(scale_data['right'])
    
    l[1] = img.height-l[1] #get points in the scale with (0,0) in the upper left corner
    r[1] = img.height-r[1] 
    
    scale_data = np.array([ [l[0], l[1]], [r[0], r[1]] ])
    scale_data_rotated = amf.affine_rotation(angle, middle, scale_data) #get the rotated coordinates
    lnew = scale_data_rotated[0,:]
    rnew = scale_data_rotated[1,:]
    
    
    img = img.rotate(angle) #rotate the image 

    v = rnew - lnew #calculate distance between both scale points
    lnew = lnew - (scale_padding_factor - 1)/2*v # widen the width
    rnew = rnew + (scale_padding_factor - 1)/2*v 
    w = np.array([ v[1], -v[0] ]) #canonical choice for a vector perpendicular to v
    ulnew = lnew + w #add w to the furthest left point
    ulnew[1] = ulnew[1] - top_offset #increase the height such that no part of the shoot gets cut off

    box = (ulnew[0],ulnew[1],rnew[0],rnew[1]) #define box
    

    cropped = img.crop(box) #crop image

    return cropped, box



def affine_rotation_from_scale_data_crop(cropped):
    """
    Determine an affine rotation based on a regression over the green pixel points of a shoot.
    """
    im_rgb = cropped.convert("RGB")
    I = np.transpose(np.array(im_rgb), (1,0,2))
    I = I.astype(np.int64)

    # treshold for the colour channel in order to detect the white background
    wlim = 10

    # Determine the colour channel
    R = I[:,:,0]
    G = I[:,:,1]
    B = I[:,:,2]

    # Set relations of the channel to detect the white background
    W = np.max( np.c_[ np.abs(R-G)[:,:,np.newaxis], np.abs(B-G)[:,:,np.newaxis], np.abs(R-B)[:,:,np.newaxis] ], axis=2)

    # Calculate the framing mask to detect the pixels that belong to a shoot
    J = (R <= G) & (B <= G) & (W > wlim) 
    
    x,y = J.nonzero()
    reg = LinearRegression(fit_intercept = True).fit(x[:,np.newaxis],y) #calculate a linear regression over all these points
    
    reg_coef = reg.coef_[0]
    reg_data= (reg.coef_, reg.intercept_)
  
    angle = np.arctan(1 / np.abs(reg_coef)) * 180 / np.pi #calculate an angle basen on the regression coefficient
    angle = np.sign(reg_coef) * (90-angle) #correct the sign of the angle
    middle = np.array( [cropped.width / 2, cropped.height / 2] ) #get the middel point

    return angle, middle, reg_data