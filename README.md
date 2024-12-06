# README #

* It will map the players positions in the court from 3D view to 2D view 
Reference: 
https://github.com/stephanj/basketballVideoAnalysis
https://github.com/stephanj/basketballVideoAnalysis/tree/master/mini-map-tutorial
https://www.linkedin.com/pulse/journey-towards-creating-basketball-mini-map-stephan-janssen/

* The working version is available in http://52.12.10.48/
(This is GPU_V7 vm container) 

*our colab code is available at: https://colab.research.google.com/drive/14YU2oVsfusdFAb6qLT5puqatg6ZtdmHx?usp=sharing
* in that colab code refer code under homography-mapping snippet



### What is this repository for? ###

When you give a input like below the node will give the mapped video as a result.
{
    "video_type": "mp4",
    "video_link": "",
    "friendly_name": "mini-map-game-sample-input",
    "file_original_name": "mini-map-game-input"
}

### How do I get set up? ###
* put homographic_mapping folder inside /interplay_v2/public/private
* put .js and .html files inside /interplay_v2/nodes/core/nodesrepositories/Quick AI/
* set the correct python path in .js file if needed.
* Place detectron2_repo folder inside /interplay_v2
* import flow.json file in interplay and test the node

* Refer https://docs.google.com/document/d/1I1u7QxsNSour8vuLADKC4kp0zibw2sFfqpO0pwU4_sM/edit?usp=sharing
for full configuration setup from scratch