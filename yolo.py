import cv2
import numpy as np
import mysql.connector
from googleapiclient.discovery import build


# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = cv2.imread("image.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (448, 448), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing information on the screen
db_str =[]
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
        db_str.append(label)

cv2.imshow("we", img)
db_str = set(db_str)

db_str = " ".join(db_str)
db_str=db_str.title()
print(db_str)
query = "Recipe with " + db_str + "as Ingridents"
    #
    # mydb = mysql.connector.connect(host="localhost", user="root", password="Aezakmi",database="reciepe_recon", auth_plugin="mysql_native_password")
    # mycursor = mydb.cursor()
    # my_sql_query = "select `name`,link from reciepe_recon.recipe_rec where ingridents = '" + db_str +"'"

#    mycursor.execute(my_sql_query)
# for db in mycursor:
#    print(db)



api_key = "API KEY"

youtube = build('youtube', 'v3', developerKey=api_key)


request = youtube.search().list(
    part="snippet",
    q=query,
    maxResults="5").execute()

videos = request['items']

for video in videos:
    final = video['snippet']['title']
    videoid = video['id']['videoId']
    print(final)
    link = "LINK: https://www.youtube.com/watch?v=" + videoid
    print(link)
    print("\n")
    print("___________________________________________________________________________")
    print("\n")




