import numpy as np

def rad_to_deg(rad):
    """
    Translation of radian measure in degree.
    """

    return rad * 360 / (2*np.pi) 

def deg_to_rad(deg):
    """
    Translation of degree in radian measure.
    """

    return deg / 360 * (2*np.pi)


def affine_rotation(angle, middle, points):
    """
    Updates coordinates of points after the rotation around the given middle by the given angle.
    
    Applies the affine rotation :math:`\mathbb{A}_{\varphi}(v) = O_{\varphi}v + m` with rotationmatrix :math:`O_{\varphi} = \begin{pmatrix} \operatorname{cos}(\varphi) & \operatorname{sin}(\varphi) \\ - \operatorname{sin}(\varphi) & \operatorname{cos}(\varphi) \end{pmatrix}` around the middle point :math:`m` on several vectors :math:`v = \text{points}`.

    """
    
    phi = deg_to_rad(angle) #degree to radian measure
    
    O = np.array([[np.cos(phi), np.sin(phi)],[-np.sin(phi), np.cos(phi)]]).reshape([2,2]) #rotation matrix
    v = points - middle
    
    flat = len(v.shape) == 1
    
    if flat:
        v = v.reshape([1,2])
        
    #get new cooedinated by multiplication with rotation matrix and addition of the middle
    result = np.einsum('ij,kj -> ki', O, v) + middle      
    
    if flat:
        result = result.flatten()
    
    return result


def inverse_affine_rotation(angle, middle, points):
    """
    Updates coordinates of points after the inverse rotation around the given middle by the given angle.

    Apply the inverse affine rotation :math:`\mathbb{A}_{\varphi}(v) = O^\intercal_{\varphi}v + m`
with rotationmatrix :math:`O^\intercal_{\varphi} = \begin{pmatrix} \operatorname{cos}(\varphi) & -\operatorname{sin}(\varphi) \\ \operatorname{sin}(\varphi) & \operatorname{cos}(\varphi) \end{pmatrix}` around the middle point :math:`m` on several vectors :math:`v = \text{points}`.
    """
    
    phi = deg_to_rad(angle) #degree in radian measure
    
    O = np.array([[np.cos(phi), np.sin(phi)],[-np.sin(phi), np.cos(phi)]]).reshape([2,2]) #rotation matrix
    v = points - middle
    
    flat = len(v.shape) == 1
    
    if flat:
        v = v.reshape([1,2])
        
    #get new cooedinated by multiplication with rotation matrix and addition of the middle
    result = np.einsum('ij,kj -> ki', O.T, v) + middle     
    
    if flat:
        result = result.flatten()
    
    return result


def centroid(points):
    """
    Calculate the centroid of points, here landmarks.

    Centroid :math:`= \frac{1}{m}\cdot\sum\limits_{i=1}^{m} x_{i}`
    """

    return np.mean(points, axis=0)


def affine_translation(new_origin, points):
    """
    Translate the origin point to new_origin and update given points.
    """

    return points - new_origin