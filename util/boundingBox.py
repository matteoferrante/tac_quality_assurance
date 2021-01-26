# bounding box class
import cv2
class BoundingBoxWidget(object):
    # this class enable the user to manual replace the ROI found by the algorithm
    roi = []

    def __init__(self, img, mean_r, points):
        self.original_image = img
        self.clone = self.original_image.copy()
        self.mean_r = mean_r
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # Bounding box reference points
        self.image_coordinates = []
        self.points=points

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x, y)]
            cv2.circle(self.clone, (x, y), self.mean_r, (255, 255, 255), 2)
            self.roi.append((x, y, self.mean_r))
            cv2.imshow("image", self.clone)



        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()
            self.roi = []

    def show_image(self):
        for i in self.points.values():
            cv2.circle(self.clone, (i[0], i[1]), i[2], (0, 0, 0), 1)
        return self.clone

    def get_roi(self):
        return self.roi

    def clean_roi(self):
        self.roi = []
