import cv2
import google.generativeai as genai
from PIL import Image
import time
import os
import csv


API_KEY=""
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemma-3-27b-it') 

def extract_values_to_csv(video_path, output_file):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0: fps = 30 
    
    frame_count = 0
    
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Temps (s)', 'Tension (V)', 'Courant (A)', 'Watt (W)'])
        
        print(f"Analysis in progress... Data will be saved in: {output_file}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % int(fps) == 0:
                second = int(frame_count / fps)

                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                if(video_path == "./video_oakd.mp4"):
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_adj = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
                
                pil_image = Image.fromarray(frame_adj)

                prompt = "Analyze the screen. Give ONLY the two numbers separated by a comma (e.g., 5.21, 0.13). Do not add any letters."
                
                try:
                    response = model.generate_content([prompt, pil_image])
                    clean_res = response.text.strip().replace(' ', '')
                    
                    valeurs = clean_res.split(',')
                    
                    if len(valeurs) == 2:
                        tension, courant = valeurs[0], valeurs[1]
                        writer.writerow([second, tension, courant, float(tension)*float(courant)])
                        print(f"[{second}s] Saved: {tension}V, {courant}A, {float(tension)*float(courant)}W")
                    else:
                        print(f"[{second}s] Format error: {clean_res}")
                        
                except Exception as e:
                    print(f"Error at {second}s: {e}")

            frame_count += 1

    cap.release()
    print(f"Analysis complete. File ready: {os.path.abspath(output_file)}")

extract_values_to_csv("./video_onnx.mp4","resultats_mesures_onnx.csv")