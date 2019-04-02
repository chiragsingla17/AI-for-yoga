import cv2
weightsFile= "/home/chirag/open pose/pose/mpi/pose_iter_160000.caffemodel"
protoFile= "/home/chirag/open pose/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
#frame = cv2.imread("/home/chirag/Downloads/single.jpeg")
# Read image
frame = cv2.imread("/home/chirag/Downloads/SIDE-LUNGE-YOGA-POSE-69.JPG")
 
# Specify the input image dimensions
inWidth = 368
inHeight = 368
 
# Prepare the frame to be fed to the network
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
 
# Set the prepared object as the input blob of the network
net.setInput(inpBlob)
output = net.forward()

H = output.shape[2]
W = output.shape[3]
# Empty list to store the detected keypoints
points = []
for i in range(18):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]
 
    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
     
    # Scale the point to fit on the original image
    x = (frame.shape[1]* point[0]) / W
    y = (frame.shape[0] * point[1]) / H
 
    if prob > 0.3 : 
        cv2.circle(frame, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
 
        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else :
        points.append(None)
frame = cv2.resize(frame,(480,720))
cv2.imshow("Output-Keypoints",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
