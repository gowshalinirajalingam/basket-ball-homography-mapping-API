module.exports = function (RED) {


    function nodeFunciton(config) {
        RED.nodes.createNode(this, config);
        // node-specific code goes here
        //const python = this.context().global.get('python');
        let python = this.context().global.get('internal_python');
        if (typeof python == "undefined") {
            const require = this.context().global.get('require');
            const pythonBridge = require('python-bridge');
            python = pythonBridge(this);
            this.context().global.set("internal_python", python);
        }
        
        //GET VALUES FROM FRONT END
        let name = config.name;

        
        let node = this;
        this.on('input', function (msg, send, done) {

            python.child.parentClass = node;

            python.ex`
print("Basket ball players mapping Starts")

try:
    import psutil
    import json
    import pandas as pd
    import sys
    import os
    import time
    from pandas_profiling import ProfileReport
    
    
    #IMPORTS FOR HOMOGRAPHY MAPPING
    #SHOULD PLACE detectron2 folder INSIDE /interplay_v2/node_modules/python-bridge
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
    
    from shapely.geometry import Point, Polygon
    
    import time
    import progressbar
    from time import sleep
    from collections import deque
    from detectron2.utils.visualizer import ColorMode
    from detectron2.utils.visualizer import GenericMask
    import imutils

    #SET STATUS AS RUNNING
    node.status({'fill': 'blue','shape': 'dot','text': 'Running'})
    
    print("All libraries are imported")
    sys.stdout.flush()

    start = time.time()
    
    
    msg = ${msg}
    
except Exception as e:
    print(e)
    sys.stdout.flush()
    node.warn(e)
    
    

#REPRESENTING A PLAYER ON THE COURT USING CLUE CIRCLE
def drawPlayers(im, pred_classes, src_pts, pred_boxes, showResult=False):
    # Use the boxes info from the tensor prediction result
    #
    # x1,y1 ------
    # |          |
    # |          |
    # |          |
    # --------x2,y2
    #
  try:
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
  except Exception as e:
        print(e)
        sys.stdout.flush()
        node.warn(e)
        


#IMAGE TRANSFORMATIONS    
#The output image (img_out) shows the player dots within a 2D view of the court
def homographyTransform(im, src_pts, dst_pts, img_dst, showResult=False):

  try:
      # Calculate Homography
      h, status = cv2.findHomography(src_pts, dst_pts)
      img_out = cv2.warpPerspective(im, h, (img_dst.shape[1], img_dst.shape[0]))
      
      if showResult:
        # cv2_imshow(img_out)
        cv2.imwrite('/interplay_v2/public/private/homographic_mapping/players_plots_zoomed.jpg', img_out)
    
        
    
      return img_out  
  except Exception as e:
      print(e)
      sys.stdout.flush()
      node.warn(e)


def getPlayersMask(im):
    
  try:
      lower_range = np.array([255,0,0])                         # Set the Lower range value of blue in BGR
      upper_range = np.array([255,155,155])                     # Set the Upper range value of blue in BGR
      mask = cv2.inRange(im, lower_range, upper_range)     # Create a mask with range
      result = cv2.bitwise_and(im, im, mask = mask)   # Performing bitwise and operation with mask in img variable
      # cv2_imshow(result)                              
    
      return cv2.inRange(result, lower_range, upper_range)  
  except Exception as e:
      print(e)
      sys.stdout.flush()
      node.warn(e)
            
#DRAW PLAYERS IN THE court.jpg IMAGE
def drawPlayersOnCourt(im, coord, color, radius=10):
  try:
      for pos in coord:
        center_coordinates = (pos[0], pos[1])
        cv2.circle(im, center_coordinates, radius, color, thickness=-1) 
    
      return im
  except Exception as e:
      print(e)
      sys.stdout.flush()
      node.warn(e)


# Draft method to draw lines between history player positions to show trail
def drawCoordinateLines(result, pts, currentFrame, player):
  try:
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
  except Exception as e:
      print(e)
      sys.stdout.flush()
      node.warn(e)


def python_function(msg):

    try:
        
        start_time = time.time()

        print("python function starts")
        
        print('msg',msg)
        
        video_type = msg['payload']['video_type']
        video_link = msg['payload']['video_link']
        friendly_name = msg['payload']['friendly_name']
        file_original_name = msg['payload']['file_original_name']
        
        print('video_type:',video_type)
        print('video_link:',video_link)
        print('friendly_name:',friendly_name)
        print('file_original_name:',file_original_name)
        sys.stdout.flush()
        
        #HOMOGRAPHY MAPPING CODE STARTS
        project_path = '/interplay_v2/public/private/homographic_mapping'
        if(not(os.path.exists(project_path))):
            os.makedirs(project_path)
            
        #READ 1ST FRAME TO CREATE THE ANNOTATED IMAGE OF THE COURT AREA
        im = cv2.imread("/interplay_v2/public/private/homographic_mapping/frame.jpg")
        print('Frame shape:',im.shape)

        #RETRIEVING THE POSITION OF EACH PLAYER USING Mask R-CNN models STARTS
        cfg = get_cfg()
        cfg.merge_from_file("/interplay_v2/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
            
        predictor = DefaultPredictor(cfg)      #eturns a list of rectangle coordinates (pred_boxes) of each identified object.
        players_output = predictor(im)

        # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
        instances = players_output["instances"]
        print(instances)
        pred_boxes = instances.get("pred_boxes")
        pred_classes = instances.get("pred_classes")   #The object classes are stored in pred_classes, where person objects are marked as 0.
        print(pred_boxes)
        print(pred_classes)
        sys.stdout.flush()

        #RETRIEVING THE POSITION OF EACH PLAYER ENDS
        
        '''
        #VISUALIZE THE PLAYERS IN THE IMAGE STARTS
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
        v = v.draw_instance_predictions(players_output["instances"].to("cpu"))
        cv2.imwrite('/interplay_v2/public/private/homographic_mapping/person_prediction.jpg', v.get_image()[:, :, ::-1])
        #VISUALIZE THE PLAYERS IN THE IMAGE ENDS
        '''
        
        #PROVIDE THE PLOYGON COORDINATES OF THE COURT AND DRAW THAT IN THE FRAME IMAGE GIVEN. AUTOMATE IT
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

        '''
        #REPRESENTING A PLAYER ON THE COURT USING CLUE CIRCLE
        drawPlayers(im, pred_classes, src_pts, pred_boxes, True)
        '''
        

        #DRAW THE PLOYGON IN THE COURT IMAGE STARTS
        img_dst = cv2.imread('/interplay_v2/public/private/homographic_mapping/court.jpg', cv2.IMREAD_UNCHANGED)
        img_dst = cv2.resize(img_dst, (1920,1080), interpolation = cv2.INTER_AREA)


        dst_pts = np.array([
              [144,  1060],  # LEFT BOTTOM
              [969,  1065],  # MIDDLE BOTTOM
              [1769, 1063],  # RIGHT BOTTOM
              [1885, 875],   # TOP BOTTOM RIGHT  (4 o'clock)
              [1882,  49],   # TOP RIGHT
              [50,    43],   # TOP LEFT
              [50,    871]   # TOP - BOTTOM LEFT (7 o'clock)
            ])   
  

        cv2.polylines(img_dst, [dst_pts], isClosed=True, color=[255,0,0], thickness=2)
        # cv2_imshow(img_dst)
        cv2.imwrite('/interplay_v2/public/private/homographic_mapping/polylines_court.jpg', img_dst)
        #DRAW THE PLOYGON IN THE COURT IMAGE ENDS

        '''
        #IMAGE TRANSFORMATIONS    
        img_out = homographyTransform(im, src_pts, dst_pts, img_dst, True) 


        mask = getPlayersMask(img_out)    
        # cv2_imshow(mask)
        cv2.imwrite('/interplay_v2/public/private/homographic_mapping/players_mask.jpg', mask)

        '''

        



        #DETECTION FOR THE VIDEO STARTS
        if video_type == 'mp4':
            video_path = '/interplay_v2/public/private/homographic_mapping/'+file_original_name+'.mp4'
        elif video_type == 'rtsp':
            video_path = video_link
            
        print('video_path',video_path)
        sys.stdout.flush()
        
        node.warn("Players mapping started!")

        vs = cv2.VideoCapture(video_path)
        totalFrames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        print('totalFrames',totalFrames)
        sys.stdout.flush()


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
            
          f_start_time = time.time()
          
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

          output_path = "/interplay_v2/public/private/homographic_mapping/"+friendly_name+"_output.webm"
          if writer is None:
            writer = cv2.VideoWriter(output_path, codec, fps, (width, height), True)

          if grabbed:

            # RETRIEVING THE POSITION OF EACH PLAYER.
            outputs = predictor(frame)  
            instances = outputs["instances"].to("cpu")
            boxes = instances.get("pred_boxes")

            court = court_img.copy()
            
            
            #REPRESENTING A PLAYER ON THE COURT USING CLUE CIRCLE. Draw players on video frame
            drawPlayers(frame, pred_classes, src_pts, boxes, False)
            
            #IMAGE TRANSFORMATIONS. The output image (img_out) shows the player dots within a 2D view of the court 
            img_out = homographyTransform(frame, src_pts, dst_pts, img_dst, False)

            #GET MASK OF EACH PLAYERS. Dots in black image            
            mask = getPlayersMask(img_out)
            

            #RETRIEVE THE COORDINATES OF THE MASKED PLAYERS.
            # Get the contours from the players "dots" so we can reduce the coordinates to the number of players on the court.
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            # print('cnts',cnts)
          
            if cnts is not None:      
              for cnt in cnts:
                #DRAW PLAYERS IN THE court.jpg IMAGE
                result = drawPlayersOnCourt(court, cnt[0], blue_color)
                                       
                # writer.write(result)
                # cv2_imshow(result)

            currentFrame += 1
            bar.update(currentFrame)

            # print('frame_num',currentFrame)
            # sys.stdout.flush()

            # if currentFrame == 1:
            #   break

            writer.write(result)
            
            # calculate frames per second of running detections
            fps = 1.0 / (time.time() - f_start_time)
            print("FPS: %.2f" % fps)
            sys.stdout.flush()

          


        # cv2_imshow(result)
            
        writer.release()
        vs.release()
        bar.finish()

        end = time.time()
        elap = (end - start)
        print("[INFO] process took {:.4f} seconds".format(elap))

        print("Video created")
        sys.stdout.flush()

        
        end_time = time.time()
        print("Time taken to complete the mapping for the full video:", end_time-start_time,"s")
        sys.stdout.flush()

        #HOMOGRAPHY MAPPING CODE ENDS
        
        
        
        
        msg['status'] = "Mapping completed succcessfully!"
        msg['output_video_path'] = output_path 

    except Exception as e:
        print(e)
        sys.stdout.flush()
        node.warn(e)
    
    return msg



#CODE BEGINS HERE
print("Node name:",${name})
msg = python_function(msg)

print('"Node name ends:',${name})

#SET STATUS AS DONE
node.status({'fill': 'green','shape': 'dot','text': 'Done'})

#THE MSG WILL BE DISPLAYED IN THE DEBUG CONSOLE
node.warn(msg)

#THE MSG WILL SENT TO NEXT NODE
node.send(msg)
sys.stdout.flush()
    `;

            //msg.payload = name;
            // node.send(msg);





            /*
            if (done) {
                done();
            }
            */
        });
        //this.context().global.set('python', python);



    }

    RED.nodes.registerType("BasketBall-Players-Mapping", nodeFunciton);
};