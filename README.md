# circular-logo-detector
Detect circular logos in a video.

[![circular-logo-detector video](https://i.ytimg.com/vi/X0GXfhVPFu0/hqdefault.jpg)](https://youtu.be/X0GXfhVPFu0 "Logo detector video test")

Execute as follows : **detect.py -i positive.avi -o export.csv**
Name your logo: positive.png

#### What this script does:
1. Detect circles in each frame using Hough transform
2. Extract region of interest ROI which are simply the content of each circle
3. Extract local features of each ROI and compare them to those of the logo
4. If we have enough similarities, then:
    * Draw a box around that circle
    * Save the box coordinates in a csv file
    
In order to improve performance, this program do a threshold filter to the image before applying the Hough transform. As you can see in the next example, this helps to get rid of false positives.
[![Hough circle detector workflow](https://i.ibb.co/0nFKXqJ/Compare.jpg)](https://ibb.co/b62F31d)
