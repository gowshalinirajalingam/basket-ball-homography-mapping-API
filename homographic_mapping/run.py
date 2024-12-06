# You may need to restart your runtime prior to this, to let your installation take effect

# Some basic setup

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


im = cv2.imread("/interplay_v2/public/private/homographic_mapping/frame.jpg")
print('Frame shape:',im.shape)

    

cfg = get_cfg()
cfg.merge_from_file("/interplay_v2/public/private/homographic_mapping/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

predictor = DefaultPredictor(cfg)
players_output = predictor(im)

# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
instances = players_output["instances"]
print(instances)
pred_boxes = instances.get("pred_boxes")
pred_classes = instances.get("pred_classes")
print(pred_boxes)
print(pred_classes)

# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
v = v.draw_instance_predictions(players_output["instances"].to("cpu"))
# cv2.imshow(v.get_image()[:, :, ::-1])
cv2.imwrite('/interplay_v2/public/private/homographic_mapping/person_prediction.jpg', v.get_image()[:, :, ::-1])

src_pts = np.array([
    [3, 707],       # left bottom - bottom corner
    [951, 816],     # middle bottom corner
    [1914, 759],     # right bottom - bottom corner
    [1917, 641],     # right bottom - top corner
    [1459, 520],     # top right rorner
    [449, 501],     # top left corner
    [3, 593]        # left bottom - top corner
    ]) 
im_poly = im.copy()


# cv2.fillPoly(img_src, [src_pts], 255)
cv2.polylines(im_poly, [src_pts], isClosed=True, color=[255,0,0], thickness=2)

cv2.imwrite('/interplay_v2/public/private/homographic_mapping/polylines_in_frame.jpg', im_poly)





# Use the boxes info from the tensor prediction result
#
# x1,y1 ------
# |          |
# |          |
# |          |
# --------x2,y2
#
from shapely.geometry import Point, Polygon

def drawPlayers(im, pred_boxes, showResult=False):
  color = [255, 0, 0]   
  thickness = 1
  radius = 1

  i  = 0
  for box in pred_boxes:
    
    # Include only class Person
    if pred_classes[i] == 0:  
        
      x1 = int(box[0])
      y1 = int(box[1])
      x2 = int(box[2])
      y2 = int(box[3])

      xc = x1 + int((x2 - x1)/2)
      player_pos1 = (xc - 1, y2)
      player_pos2 = (xc + 1, y2 + 1)

      court = Polygon(src_pts)

      # Draw only players that are within the basketball court
      if Point(player_pos1).within(court):
        if showResult:
          print("[% 3d, % 3d]" %(xc, y2))

        cv2.rectangle(im, player_pos1, player_pos2, color, thickness)
        i = i + 1            

  if showResult:
    # cv2_imshow(im)
    cv2.imwrite('/interplay_v2/public/private/homographic_mapping/players_plots.jpg', im)


drawPlayers(im, pred_boxes, True)


img_dst = cv2.imread('/interplay_v2/public/private/homographic_mapping/court.jpg', cv2.IMREAD_UNCHANGED)
img_dst = cv2.resize(img_dst, (1920,1080), interpolation = cv2.INTER_AREA)

# img_dst = cv2.imread('./court.jpg')

# Four corners of the court + mid-court circle point in destination image 
# Start top-left corner and go anti-clock wise + mid-court circle point
dst_pts = np.array([
      [144,  1060],  # LEFT BOTTOM
      [969,  1065],  # MIDDLE BOTTOM
      [1769, 1063],  # RIGHT BOTTOM
      [1885, 875],   # TOP BOTTOM RIGHT  (4 o'clock)
      [1882,  49],   # TOP RIGHT
      [50,    43],   # TOP LEFT
      [50,    871]   # TOP - BOTTOM LEFT (7 o'clock)
    ])   

# dst_pts = np.array([
#     [43, 355],       # left bottom - bottom corner
#     [317, 351],      # middle bottom corner
#     [563, 351],      # right bottom - bottom corner
#     [629, 293],      # right bottom - top corner
#     [628, 3],        # top right rorner
#     [8, 4],          # top left corner
#     [2, 299]         # left bottom - top corner
#     ])   

cv2.polylines(img_dst, [dst_pts], isClosed=True, color=[255,0,0], thickness=2)
# cv2_imshow(img_dst)
cv2.imwrite('/interplay_v2/public/private/homographic_mapping/polylines_court.jpg', img_dst)

def homographyTransform(im, showResult=False):

  # Calculate Homography
  h, status = cv2.findHomography(src_pts, dst_pts)
  img_out = cv2.warpPerspective(im, h, (img_dst.shape[1], img_dst.shape[0]))
  
  if showResult:
    # cv2_imshow(img_out)
    cv2.imwrite('/interplay_v2/public/private/homographic_mapping/players_plots_zoomed.jpg', img_out)

    

  return img_out  

# Try out
img_out = homographyTransform(im, True) 




def getPlayersMask(im):
  lower_range = np.array([255,0,0])                         # Set the Lower range value of blue in BGR
  upper_range = np.array([255,155,155])                     # Set the Upper range value of blue in BGR
  mask = cv2.inRange(im, lower_range, upper_range)     # Create a mask with range
  result = cv2.bitwise_and(im, img_out, mask = mask)   # Performing bitwise and operation with mask in img variable
  # cv2_imshow(result)                              

  return cv2.inRange(result, lower_range, upper_range)  
  
# Try out  
mask = getPlayersMask(img_out)    
# cv2_imshow(mask)
cv2.imwrite('/interplay_v2/public/private/homographic_mapping/players_mask.jpg', mask)



def drawPlayersOnCourt(im, coord, color, radius=10):
  for pos in coord:
    center_coordinates = (pos[0], pos[1])
    cv2.circle(im, center_coordinates, radius, color, thickness=-1) 

  return im


# Draft method to draw lines between history player positions to show trail
def drawCoordinateLines(result, pts, currentFrame, player):
  
  for i in np.arange(1, len(pts)):
    
    # if either of the tracked points are None, ignore them
    if pts[i - 1] is None or pts[i] is None:
      continue

    thickness = int(np.sqrt(30 / float(i + 1)) * 2.5)
    print("player=%s" %player)
    x1 = pts[i - 1][0][0]
    x2 = pts[i - 1][0][1]
    print("x1=%d, x2=%d" %(x1, x2))
    y1 = pts[i][0][0]
    y2 = pts[i][0][1]
    print("y1=%d, y2=%d" %(y1, y2))
    print(" ---------------------- ")
    cv2.line(result, (x1, x2), (y1, y2), red_color, thickness)

  return result


import time
import progressbar
from time import sleep
from collections import deque
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import GenericMask
import imutils

vs = cv2.VideoCapture("/interplay_v2/public/private/homographic_mapping/mini-map-game-input.mp4")
totalFrames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
print('totalFrames',totalFrames)

grabbed = True
currentFrame = 0
start = time.time()
writer = None
output_format = 'vp80' #for webm

bar = progressbar.ProgressBar(maxval=totalFrames, \
      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

bar.start()

court_img = cv2.imread('/interplay_v2/public/private/homographic_mapping/court.jpg')
court_img = cv2.resize(court_img, (1920,1080), interpolation = cv2.INTER_AREA)
# cv2_imshow(court_img)


blue_color = (255,0,0)
red_color = (0,0,255)

# loop over frames from the video file stream (207)
while grabbed:     
  
  # read the next frame from the file
  (grabbed, frame) = vs.read()

  # cv2.imwrite('./frame.jpg', frame)


  if not grabbed:
    break

  # by default VideoCapture returns float instead of int
  width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(vs.get(cv2.CAP_PROP_FPS))
  codec = cv2.VideoWriter_fourcc(*output_format)

  if writer is None:
    writer = cv2.VideoWriter("/interplay_v2/public/private/homographic_mapping/mini-map-output2.webm", codec, fps, (width, height), True)

  if grabbed:
    # print(currentFrame)

    # Get player positions
    outputs = predictor(frame)  
    instances = outputs["instances"].to("cpu")
    boxes = instances.get("pred_boxes")

    court = court_img.copy()
    # Draw players on video frame
    drawPlayers(frame, boxes, False)
    
    img_out = homographyTransform(frame, False)
    # cv2_imshow(img_out)

    mask = getPlayersMask(img_out)
    
    # cv2_imshow(mask)

    # Get the contours from the players "dots" so we can reduce the coordinates
    # to the number of players on the court.
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # print('cnts',cnts)
  
    if cnts is not None:      
      for cnt in cnts:
        result = drawPlayersOnCourt(court, cnt[0], blue_color)
                               
        # writer.write(result)
        # cv2_imshow(result)

    currentFrame += 1
    bar.update(currentFrame)

    print('frame_num',currentFrame)
    # if currentFrame == 1:
    #   break

    writer.write(result)

  


# cv2_imshow(result)
    
writer.release()
vs.release()
bar.finish()

end = time.time()
elap = (end - start)
print("[INFO] process took {:.4f} seconds".format(elap))

print("Video created")


