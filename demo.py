import cv2
import torch
from torchvision import transforms
from PIL import Image
import warnings
import torch.nn.functional as F
from mysoble_and_attention import Network
warnings.filterwarnings("ignore")
import numpy as np
import pyautogui  # 导入模拟键盘输入的库
torch.cuda.set_device(0)
model_path = "./model/model.pth"

# 定义一个映射，将预测的标签映射到相应的键盘按键
label_to_key = {
    'a': 'a',
    'b': 'b',
    'c': 'c',
    'd': 'd',
    'e': 'e',
    'f': 'f',
    'g': 'g',
    'h': 'h',
    'i': 'i',
    'j': 'j',
    'k': 'k',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'o': 'o',
    'p': 'p',
    'q': 'q',
    'r': 'r',
    's': 's',
    't': 't',
    'u': 'u',
    'v': 'v',
    'w': 'w',
    'x': 'x',
    'y': 'y',
    'z': 'z',
    '0': '0',
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9',
    '10': '10',
    'A': 'A',
    'B': 'B',
    'C': 'C',
    'D': 'D',
    'E': 'E',
    'F': 'F',
    'G': 'G',
    'H': 'H',
    'I': 'I',
    'J': 'J',
    'K': 'K',
    'L': 'L',
    'M': 'M',
    'N': 'N',
    'O': 'O',
    'P': 'P',
    'Q': 'Q',
    'R': 'R',
    'S': 'S',
    'T': 'T',
    'U': 'U',
    'V': 'V',
    'W': 'W',
    'X': 'X',
    'Y': 'Y',
    'Z': 'Z',
}

def load_model(model_path):
    model_info = torch.load(model_path, map_location=torch.device('cuda:0'))
    classes = model_info['classes']
    num_classes = len(classes)
    model = Network(num_classes,4)
    model.load_state_dict(model_info['model'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 默认设备
    model = model.to(device)  
    model.eval()
    return model, classes

def process_image(image, transform):
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = transform(image).unsqueeze(0)
    return image


def predict(image_path, model, labels, transform, plot_probabilities=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    image = process_image(image_path, transform)
    image1 = image[:, :, :, :image.shape[3] // 2]
    image2 = image[:, :, :, image.shape[3] // 2:]
    image1 = image1.to(device) 
    image2 = image2.to(device)
    output = model(image1,image2)
    prob = F.softmax(output, dim=1)
    predicted_index = torch.argmax(prob, dim=1)  
    predicted_label = labels[predicted_index.item()]  
    max_confidence = torch.max(prob).item()  
    return predicted_label, max_confidence


def main():
    model, classes = load_model(model_path)    
    transform = transforms.Compose([
        transforms.Resize((128,324)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    threshold = 0.6
    cam1 = cv2.VideoCapture(1)
    cam2 = cv2.VideoCapture(0)
    cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    recording = False
    gesture_list = []
    def remove_consecutive_duplicates(lst):
        return [v for i, v in enumerate(lst) if i == 0 or v != lst[i-1]]
    
    
    last_prediction = None
    while True:
        ret1, frame1 = cam2.read()
        ret2, frame2 = cam1.read()
        #frame1 = cv2.flip(frame1, -1) 
        frame2 = cv2.flip(frame2, -1)
        combined_frame = cv2.hconcat([frame1, frame2])
        if cv2.waitKey(1) & 0xFF == ord(' '):
            recording = not recording
            if recording:
                print("Recording started")
            else:
                print("Recording ended")
                gesture_list = remove_consecutive_duplicates(gesture_list)
                
                print(gesture_list)
                gesture_list = []

        if recording:
            results, confidence = predict(combined_frame, model, classes, transform, plot_probabilities=True)
            # 如果置信度低于阈值，跳过
            if confidence < threshold:
                continue
            gesture_list.append(results)  
            cv2.putText(combined_frame, f'Result: {results}, Confidence: {confidence}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
            # 只有当新的预测与上一次的预测不同时，我们才模拟键盘输入
            if results != last_prediction:
                if results in label_to_key:
                    pyautogui.write(label_to_key[results])
                last_prediction = results
        window_title = 'Recording' if recording else 'Not Recording'
        cv2.imshow(window_title, combined_frame)
        if not ret1 or not ret2:
            print("Failed to grab frame.")
            break
        # 按下'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 释放摄像头并关闭所有窗口
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()