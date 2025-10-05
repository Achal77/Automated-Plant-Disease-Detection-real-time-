import cv2
import numpy as np
import tensorflow as tf
import os


model = tf.keras.models.load_model('C:\\Users\\LOQ\\OneDrive\\Desktop\\dataset\\model.h5')


train_dir = "C:\\Users\\LOQ\\OneDrive\\Desktop\\archive (2)\\New Plant Diseases Dataset(Augmented)\\New Plant Diseases Dataset(Augmented)\\train"
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
print("Detected Classes:", class_names)

img_size = (128, 128)

def preprocess(img):
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = preprocess(frame)
    preds = model.predict(img)
    class_idx = np.argmax(preds)
    conf = preds[0][class_idx]
    label = f"{class_names[class_idx]}: {conf*100:.2f}%"
    cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Plant Disease Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()