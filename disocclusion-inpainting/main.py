import sys, os
import time
import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from skimage.filters import threshold_otsu
from DisocclusionFill import DisoccllusionFill

def createargs():
    parser = argparse.ArgumentParser(description='Inpaint Disocclusion.')
    parser.add_argument('--textureImage', type = str)
    parser.add_argument('--depthImage', type = str)
    parser.add_argument('--disoccColor', nargs='+', type = int)
    parser.add_argument('--drnFill', type = int, default = 0)
    parser.add_argument('--outdir',  type = str, default = 'result')
    return parser.parse_args()

def main():
    #inputs
    """
    textfilename = 'tex_BAL.png'
    depthfilename = 'dep_BAL.png'
    texture = cv2.imread(textfilename)
    depth = cv2.imread(depthfilename, 0)
    """
    args = createargs()

    if (args.textureImage is not None) and (args.depthImage is not None):
        texture = cv2.imread(args.textureImage)
        if (type(texture) is not np.ndarray):
            print("Error: Unable to open texture image.")
            exit(-1)
        depth = cv2.imread(args.depthImage,0)
        if (type(depth) is not np.ndarray):
            print ("Error: Unable to open depth image.")
            exit(-1)

    else:
        print("Provide texture and depth images!")
        return

    if args.disoccColor is None:
        print("Provide diocclusionColor e.g. (255, 255, 255)!")
        return

    out_path = args.outdir
    if args.outdir is not None:
        out_path = args.outdir

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    print("out_path", out_path)

    #inpainting begins here
    start = time.clock()
    #create DisoccllusionFill object
    dF = DisoccllusionFill(texture, depth, args.disoccColor)
    #start Inpainting
    dF.inpaintDisocclusion()
    resultTexture = dF.inpaintedImage
    resultDepth = dF.inpaintedDepth
    end = time.clock()
    print("Inpainting Running Time:", end-start)

    #write result
    cv2.imwrite(os.path.join(out_path, 'inpainted_texture.png'), resultTexture)
    cv2.imwrite(os.path.join(out_path, 'inpainted_depth.png'), resultDepth)

    #Display result
    """
    plt.imshow(cv2.cvtColor(texture, cv2.COLOR_BGR2RGB))
    plt.title('Texture:Disocclusion(White)')
    plt.show()

    plt.imshow(depth, cmap='gray')
    plt.title('depth')
    plt.show()

    plt.imshow(cv2.cvtColor(resultTexture, cv2.COLOR_BGR2RGB))
    plt.title('InpaintedTextureImage')
    plt.show()
    """

if __name__ == "__main__":
    """
    Usage: python main.py --textureImage tex_BAL2.png --depthImage dep_BAL2.png --disoccColor 255 255 255
    """
    main()
