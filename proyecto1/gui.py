import cv2

from defs import SobelFilter, equalize_hsv_channel, BlurFilter, BLUR30

if __name__ == "__main__":
    vid = cv2.VideoCapture(0)
    while(True):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        ''' Border Detection '''
        # cv2.imshow('frame', SobelFilter(frame))
        '''Histogram Equalization'''
        # cv2.imshow('frame', equalize_hsv_channel(frame, 2))
        ''' Blurring Effect '''
        cv2.imshow('frame', BlurFilter(frame, blur_weights=BLUR30))

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
