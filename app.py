from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import mediapipe as mp
from keras import models
from PIL import ImageFont, ImageDraw, Image
import time
from collections import deque, Counter
import spacy, csv, nltk
from nltk.tag import pos_tag
from collections import Counter
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from googletrans import Translator

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('stopwords')

translator = Translator()

# โหลดโมเดลและตั้งค่า
model = models.load_model('tsl_model.h5')

# สมมติว่ามี actions list (ปรับตามโมเดลของคุณ)
actions = ['book', 'stay', 'speak', 'what', 'chicken_basil', 'fish', 'like', 'laugh',
 'buffalo', 'none', 'listen', 'drink', 'you', 'sleep', 'he', 'eat', 'school', 'me',
 'where', 'rice', 'house', 'student', 'today', 'run', 'walk']

colors = [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for _ in range(len(actions))]

# Font settings
font_path = "Datasets/SukhumvitSet-Medium.ttf"
try:
    font = ImageFont.truetype(font_path, 45)
    font_small = ImageFont.truetype(font_path, 35)
except:
    font = ImageFont.load_default()
    font_small = ImageFont.load_default()

class RealTimeSignRecognizer:
    def __init__(self, model, actions, target_sequence_length=30):
        self.model = model
        self.actions = actions
        self.target_sequence_length = target_sequence_length
        
        self.motion_keypoints = []
        
        self.sentence = []
        self.last_prediction = ""
        self.prediction_confidence = 0.0
        
        self.motion_detector = MotionDetector()
        self.gesture_state = "waiting"  # waiting, recording, processing
        self.no_motion_frames = 0
        self.max_no_motion_frames = 15  # หยุดนิ่ง 15 เฟรม = ท่าทางเสร็จ
        self.min_motion_frames = 10    # ต้องมีการเคลื่อนไหวอย่างน้อย 10 เฟรม
        
        self.frame_count = 0
        self.recording_start_time = 0
        
    def extract_data_fixed(self, results):
        """ดึงข้อมูล keypoints แบบ fixed"""
        data = []

        def get_fixed_landmarks(num_points, landmarks):
            fixed_data = []
            for i in range(num_points):
                if landmarks and i < len(landmarks.landmark):
                    landmark = landmarks.landmark[i]
                    fixed_data.extend([landmark.x, landmark.y, landmark.z])
                else:
                    fixed_data.extend([0.0, 0.0, 0.0])
            return fixed_data

        data.extend(get_fixed_landmarks(21, results.right_hand_landmarks))
        data.extend(get_fixed_landmarks(21, results.left_hand_landmarks))
        data.extend(get_fixed_landmarks(33, results.pose_landmarks))
        data.extend(get_fixed_landmarks(468, results.face_landmarks))

        return np.array(data, dtype=np.float32)
    
    def resample_keypoints(self, keypoints_list, target_length):
        """แบ่ง keypoints ให้เป็น target_length frames เท่าๆ กัน โดยใช้ linear interpolation"""
        if len(keypoints_list) == 0:
            return []
        
        if len(keypoints_list) == target_length:
            return keypoints_list
        
        # สร้าง indices สำหรับการ sampling แบบเท่าๆ กัน
        original_length = len(keypoints_list)
        indices = np.linspace(0, original_length - 1, target_length)
        
        resampled = []
        for idx in indices:
            if idx == int(idx):
                resampled.append(keypoints_list[int(idx)])
            else:
                lower_idx = int(np.floor(idx))
                upper_idx = int(np.ceil(idx))
                weight = idx - lower_idx
                
                if upper_idx < len(keypoints_list):
                    interpolated = (1 - weight) * keypoints_list[lower_idx] + weight * keypoints_list[upper_idx]
                    resampled.append(interpolated)
                else:
                    resampled.append(keypoints_list[lower_idx])
        
        return resampled
    
    def predict_gesture(self, keypoints_sequence):
        """ทำนายท่าทางจาก keypoints sequence"""
        if len(keypoints_sequence) != self.target_sequence_length:
            print(f"❌ Sequence length mismatch: {len(keypoints_sequence)} != {self.target_sequence_length}")
            return None, 0.0
            
        try:
            input_data = np.expand_dims(np.array(keypoints_sequence), axis=0)
            prediction = self.model.predict(input_data, verbose=0)[0]
            
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]
            
            return self.actions[predicted_class], confidence
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None, 0.0
    
    def process_frame(self, keypoints):
        """ประมวลผลเฟรมและจัดการสถานะการบันทึก"""
        self.frame_count += 1
        
        # ตรวจสอบความถูกต้องของข้อมูล keypoints
        if len(keypoints) != 543 * 3:  # 21+21+33+468 landmarks * 3 coordinates
            print(f"⚠️ Invalid keypoints size: {len(keypoints)}")
            return None
        
        # ตรวจจับการเคลื่อนไหว
        has_motion = self.motion_detector.detect_motion(keypoints)
        
        prediction_result = None
        
        if has_motion:
            self.no_motion_frames = 0
            
            if self.gesture_state == "waiting":
                self.gesture_state = "recording"
                self.motion_keypoints = []
                self.recording_start_time = time.time()
                print("🎬 เริ่มบันทึกท่าทาง...")
            
            if self.gesture_state == "recording":
                self.motion_keypoints.append(keypoints.copy())
                
                # แสดงความคืบหน้าการบันทึก
                if len(self.motion_keypoints) % 10 == 0:
                    print(f"📹 Recording... {len(self.motion_keypoints)} frames")
        
        else:
            self.no_motion_frames += 1
            
            if self.gesture_state == "recording":
                self.motion_keypoints.append(keypoints.copy())
            
            if (self.gesture_state == "recording" and 
                self.no_motion_frames >= self.max_no_motion_frames):
                
                if len(self.motion_keypoints) >= self.min_motion_frames:
                    self.gesture_state = "processing"
                    
                    resampled_keypoints = self.resample_keypoints(self.motion_keypoints, self.target_sequence_length)
                    
                    if len(resampled_keypoints) == self.target_sequence_length:
                        prediction, confidence = self.predict_gesture(resampled_keypoints)
                        
                        if prediction and confidence > 0.3:  # ลด threshold เพื่อให้ sensitive มากขึ้น
                            self.last_prediction = prediction
                            self.prediction_confidence = confidence
                            
                            # เพิ่มคำลงในประโยค
                            self.add_word_to_sentence(prediction)
                            
                            prediction_result = {
                                'word': prediction,
                                'confidence': confidence,
                                'frames_recorded': len(self.motion_keypoints),
                                'frames_resampled': len(resampled_keypoints),
                                'recording_time': time.time() - self.recording_start_time
                            }
                            
                            print(f"✅ ทำนายได้: {prediction} (confidence: {confidence:.3f})")
                        else:
                            print(f"❌ ความมั่นใจต่ำ: {prediction} ({confidence:.3f})" if prediction else "❌ ไม่สามารถทำนายได้")
                    else:
                        print(f"❌ ผิดพลาดในการแบ่งข้อมูล: {len(resampled_keypoints)} frames")
                
                else:
                    print(f"⚠️ ท่าทางสั้นเกินไป ({len(self.motion_keypoints)} < {self.min_motion_frames} frames)")
                
                self.reset_for_next_gesture()
        
        return prediction_result
    
    def add_word_to_sentence(self, word):
        """เพิ่มคำลงในประโยค"""
        if len(self.sentence) == 0 or self.sentence[-1] != word:
            self.sentence.append(word)
            print(f"📝 เพิ่มคำ: '{word}' -> ประโยค: '{' '.join(self.sentence)}'")
        else:
            print(f"⚠️ ไม่เพิ่มคำซ้ำ: '{word}'")
        
        if len(self.sentence) > 15:
            self.sentence = self.sentence[-15:]
            print("✂️ ตัดประโยคที่ยาวเกินไป")
    
    def reset_for_next_gesture(self):
        """รีเซ็ตสำหรับท่าทางใหม่"""
        self.gesture_state = "waiting"
        self.motion_keypoints = []
        self.no_motion_frames = 0
        self.recording_start_time = 0
        print("🔄 พร้อมสำหรับท่าทางใหม่")
    
    def clear_sentence(self):
        """ล้างประโยค"""
        self.sentence = []
        print("🗑️ ล้างประโยค")
    
    def get_final_sentence(self):
        """ได้ประโยคสุดท้าย"""
        return ' '.join(self.sentence)
    
    def get_display_info(self):
        """ได้ข้อมูลสำหรับแสดงผล"""
        return {
            'sentence': ' '.join(self.sentence),
            'current_prediction': self.last_prediction,
            'confidence': self.prediction_confidence,
            'state': self.gesture_state,
            'frame_count': self.frame_count,
            'recorded_frames': len(self.motion_keypoints),
            'no_motion_count': self.no_motion_frames,
            'target_frames': self.target_sequence_length
        }

class MotionDetector:
    def __init__(self, window_size=5):
        self.prev_keypoints = None
        self.motion_history = deque(maxlen=window_size)
        self.motion_threshold = 0.012
        
    def detect_motion(self, keypoints):
        """ตรวจจับการเคลื่อนไหวจาก keypoints"""
        if self.prev_keypoints is None:
            self.prev_keypoints = keypoints.copy()
            return False

        hand_r_size = 21 * 3     # มือขวา
        hand_l_size = 21 * 3     # มือซ้าย  
        pose_size = 33 * 3       # โพสท่า
        face_size = 468 * 3      # ใบหน้า
        
        start_idx = 0
        hand_r_curr = keypoints[start_idx:start_idx + hand_r_size]
        hand_r_prev = self.prev_keypoints[start_idx:start_idx + hand_r_size]
        start_idx += hand_r_size
        
        hand_l_curr = keypoints[start_idx:start_idx + hand_l_size]
        hand_l_prev = self.prev_keypoints[start_idx:start_idx + hand_l_size]
        start_idx += hand_l_size
        
        pose_curr = keypoints[start_idx:start_idx + pose_size]
        pose_prev = self.prev_keypoints[start_idx:start_idx + pose_size]
        start_idx += pose_size
        
        face_curr = keypoints[start_idx:]
        face_prev = self.prev_keypoints[start_idx:]
        
        hand_r_motion = np.mean(np.abs(hand_r_curr - hand_r_prev))
        hand_l_motion = np.mean(np.abs(hand_l_curr - hand_l_prev))
        pose_motion = np.mean(np.abs(pose_curr - pose_prev))
        face_motion = np.mean(np.abs(face_curr - face_prev))
        
        motion_score = (0.4 * hand_r_motion + 
                       0.4 * hand_l_motion + 
                       0.15 * pose_motion + 
                       0.05 * face_motion)
        
        has_motion = motion_score > self.motion_threshold
        
        self.motion_history.append(has_motion)
        
        self.prev_keypoints = keypoints.copy()
        
        if len(self.motion_history) >= 4:
            recent_motion = list(self.motion_history)[-4:]
            return sum(recent_motion) >= 2
        
        return has_motion

def process_video(video_path):
    """ประมวลผลวิดีโอและส่งคืนประโยคที่ตรวจจับได้"""
    print(f"🎬 เริ่มประมวลผลวิดีโอ: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดวิดีโอได้")
        return None, 0.0
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    recognizer = RealTimeSignRecognizer(model, actions)
    
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📊 วิดีโอข้อมูล: {total_frames} เฟรม, {fps:.2f} FPS, ความยาว {duration:.2f} วินาที")
    
    processed_frames = 0
    all_predictions = []
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ ไม่สามารถอ่านเฟรมได้")
                break
            
            processed_frames += 1
            
            if processed_frames % 30 == 0:
                progress = (processed_frames / total_frames) * 100
                print(f"📈 ความคืบหน้า: {progress:.1f}% ({processed_frames}/{total_frames})")
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            
            keypoints = recognizer.extract_data_fixed(results)
            
            prediction_result = recognizer.process_frame(keypoints)
            
            if prediction_result:
                all_predictions.append(prediction_result)
    
    final_sentence = recognizer.get_final_sentence()
    
    avg_confidence = 0.0
    if all_predictions:
        avg_confidence = sum(pred['confidence'] for pred in all_predictions) / len(all_predictions)
    
    cap.release()
    
    print(f"ประมวลผลวิดีโอเสร็จสิ้น")
    print(f"ทำนายได้ {len(all_predictions)} คำ")
    print(f"ประโยคสุดท้าย: '{final_sentence}'")
    print(f"ความมั่นใจเฉลี่ย: {avg_confidence:.3f}")
    
    return final_sentence, avg_confidence

def draw_thai_text(image, text, position, font, color=(255, 255, 255)):
    """วาดข้อความภาษาไทยบนภาพ"""
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

nlp = spacy.load("en_core_web_sm")
csv_file_path = 'research_new.csv'

# สร้าง listA และ listB
listA = []
corpus = []

# อ่านไฟล์ CSV และเก็บข้อมูลใน listA และ corpus
with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # ข้ามหัวข้อ

    for row in csv_reader:
        listA.append(row[0])
        corpus.append(row[1])

def lemmatize_text_input(text):
    lemmatized_text = [token.lemma_ for token in nlp(text)]
    return " ".join(lemmatized_text)

def lemmatize_corpus(list_corpus):
  lem_corpus = []
  for i in list_corpus:
    lemmatized_corpus = [token.lemma_ for token in nlp(i)]
    cor = " ".join(lemmatized_corpus)
    lem_corpus.append(cor)
  return lem_corpus


def func_tfidf_cosine_lem(query_sentence, corpus_lem, corpus_nl):
    ifidfvectorizer = TfidfVectorizer()
    vectorizer = ifidfvectorizer.fit_transform([query_sentence] + corpus_lem)
    vocab = ifidfvectorizer.get_feature_names_out()
    query_vector = vectorizer.toarray()[0]
    corpus_lem_vectors = vectorizer.toarray()[1:]
    similarities = cosine_similarity([query_vector], corpus_lem_vectors)
    df = pd.DataFrame(corpus_lem_vectors, columns=vocab)
    most_similar_index = similarities.argmax()
    text = "Not found"
    if similarities.argmax() != 0:
        text = corpus_lem[most_similar_index], corpus_nl[most_similar_index]
    return text, df, similarities.max()

lemmatized_query_corpus = lemmatize_corpus(corpus)

def tokenzie_data(data):
    new_data = []
    for i in range(len(data)):
        new_data.append(" ".join([j[1] for j in pos_tag(nltk.word_tokenize(data[i]))]))
    return new_data

new_data = pd.read_csv("new_lang_pair.csv")
new_data.drop(columns=["Unnamed: 0"], inplace=True)

keys = list(new_data["ASL"])
values = list(new_data["English"])

def find_most_similar_sentence_new_value(query_sentence):
    query_sentence = query_sentence.lower()

    query_sentence = [i[1] for i in pos_tag(nltk.word_tokenize(query_sentence))]
    query_sentence = list(map(lambda x: x[1] + str(query_sentence[:x[0]].count(x[1]) + 1) if query_sentence.count(x[1]) > 1 else x[1], enumerate(query_sentence)))
    for i in range(len(query_sentence)):
        if not any(j.isdigit() for j in query_sentence[i]):
            query_sentence[i] = query_sentence[i]+"1"
    query_sentence = " ".join(query_sentence)

    # print(query_sentence)

    most_similar_sentence_ifidf_cosine_lem, __, __ = func_tfidf_cosine_lem(query_sentence, values, values)

    result = most_similar_sentence_ifidf_cosine_lem

    return query_sentence, result[1]

def find_most_similar_sentence_new_key(query_sentence):
    query_sentence = query_sentence.lower()

    query_sentence = [i[1] for i in pos_tag(nltk.word_tokenize(query_sentence))]
    query_sentence = list(map(lambda x: x[1] + str(query_sentence[:x[0]].count(x[1]) + 1) if query_sentence.count(x[1]) > 1 else x[1], enumerate(query_sentence)))
    for i in range(len(query_sentence)):
        if not any(j.isdigit() for j in query_sentence[i]):
            query_sentence[i] = query_sentence[i]+"1"
    query_sentence = " ".join(query_sentence)

    # print(query_sentence)

    most_similar_sentence_ifidf_cosine_lem, __, __ = func_tfidf_cosine_lem(query_sentence, keys, values)

    result = most_similar_sentence_ifidf_cosine_lem

    return query_sentence, result[1]

def find_most_frequent_word(pattern, training_data):
    pattern_words = pattern.split()
    blank_indices = [i for i, word in enumerate(pattern_words) if word == '___']
    
    n = len(pattern_words)
    candidate_lists = {idx: [] for idx in blank_indices}  # Track words for each blank position

    for sentence in training_data:
        words = sentence.split()
        for i in range(len(words) - n + 1):
            ngram = words[i:i + n]
            if all(pw == w or pw == '___' for pw, w in zip(pattern_words, ngram)):
                for idx in blank_indices:
                    candidate_lists[idx].append(ngram[idx])

    # Find most frequent words for each blank
    most_frequent_words = []
    for idx in blank_indices:
        word_counts = Counter(candidate_lists[idx])
        most_common_word = word_counts.most_common(1)[0][0] if word_counts else "_"
        most_frequent_words.append(most_common_word)

    return most_frequent_words

def is_sublist(sublist, main_list):
    if not sublist:
        return True  # Empty list is always a sublist

    i = 0
    for j in range(len(main_list)):
        if main_list[j] == sublist[i]:
            i += 1
            if i == len(sublist):
                return True
    return False

def overlap_merge(lists):
    merged = lists[0][:]  # Start with the first list (make a copy)

    for lst in lists[1:]:
        # Find the maximum overlap between merged and lst
        max_overlap = 0
        for i in range(len(lst)):
            if merged[-(i + 1):] == lst[: i + 1]:  # Compare suffix of merged with prefix of lst
                max_overlap = i + 1  # Store the maximum matched overlap length

        # Append only the non-overlapping part of lst
        merged.extend(lst[max_overlap:])

    return merged

def correct_sentence(sentence):
    # Corrects grammar mistakes in a sentence, including auxiliary verbs, subject-verb agreement, noun singular/plural, and verb conjugation.
    
    # Helper functions
    def correct_auxiliary(token, subject):
        # Corrects auxiliary verbs based on subject agreement.
        if subject.text.lower() == "you":
            return "are"
        elif subject.text.lower() in ["he", "she", "it", "this", "that"]:
            return "is"
        elif subject.text.lower() in ["i"]:
            return "am"
        else:
            return "are"

    def correct_possessive_noun(token, next_token):
        # Corrects noun to possessive form.
        if next_token and next_token.text.lower() == "name" and token.text.lower() == "mother":
            return "mother's"  # Correct for possessive form
        return token.text

    def singularize_noun(noun):
        # Singularizes the noun if it's plural (handles regular pluralization).
        if noun.endswith("ies"):
            return noun[:-3] + "y"  # cities → city
        elif noun.endswith("es") and not noun.endswith("ss"):
            return noun[:-2]  # buses → bus
        elif noun.endswith("s"):
            return noun[:-1]  # dogs → dog
        return noun

    def conjugate_verb(verb, subject):
        # Conjugates verb based on subject.
        if subject.text.lower() in ["he", "she", "it"]:
            if verb == "have":
                return "has"  # Handle special case for "have" → "has"
            return verb + "s" if not verb.endswith("s") else verb  # eat → eats
        elif subject.text.lower() in ["i", "you", "we", "they"]:
            return verb  # Keep base form
        return verb

    # Process the sentence
    doc = nlp(sentence)
    corrected_words = []

    for token in doc:
        text = token.text.lower()
        pos = token.pos_
        dep = token.dep_

        # Identify the subject of the sentence
        subject = next((t for t in doc if t.dep_ == "nsubj"), None)

        # Handle auxiliary verbs (specifically "be")
        if text == "be" and subject:
            corrected_words.append(correct_auxiliary(token, subject))
            continue

        # Handle noun singularization and pluralization
        if pos == "NOUN":
            # Singularize nouns when necessary (e.g., "names" to "name")
            if token.text.lower() == "names":
                corrected_words.append(singularize_noun(token.text))
            else:
                corrected_words.append(correct_possessive_noun(token, doc[token.i + 1] if token.i + 1 < len(doc) else None))
            continue

        # Handle verb conjugation (e.g., "eat" to "eats")
        if pos == "VERB" and subject:
            corrected_words.append(conjugate_verb(text, subject))
            continue

        # Add token as is if no correction is needed
        corrected_words.append(token.text)

    # Check for subject-verb agreement after processing
    final_sentence = " ".join(corrected_words)
    if "mother's" in final_sentence and "are" in final_sentence:
        final_sentence = final_sentence.replace("are", "is")  # Correct "are" to "is" if "mother's" is singular

    return final_sentence

def sign_translator(text_used, stat):
    if stat == False:
        after_result = find_most_similar_sentence_new_value(text_used)
    elif stat == True:
        after_result = find_most_similar_sentence_new_key(text_used)
    match_word = dict(zip(after_result[0].split(" "), nltk.word_tokenize(text_used)))
    splitted_result_sentence = after_result[1].split(" ")
    result_sentence = [match_word[splitted_result_sentence[i]] if splitted_result_sentence[i] in match_word.keys() else "___" for i in range(len(splitted_result_sentence))]
    
    subsens = []

    for i in range(len(result_sentence)):
        for j in range(i+1, len(result_sentence)+1):
            temp = []
            temp.append(result_sentence[i])
            temp += result_sentence[i+1:j]
            if temp.count("___") > 1 or temp.count("___") == 0:
                break
            else:
                if len(temp) != 1:
                    subsens.append(temp)
    subsens.append(result_sentence)

    for i in range(len(subsens)):
        if "___" in subsens[i]:
            temp = find_most_frequent_word(" ".join(subsens[i]), lemmatized_query_corpus)
            for j in temp:
                subsens[i][subsens[i].index("___")] = j
            if subsens[i].count("_") != 0:
                subsens[i].remove("_")
        else:
            subsens[i] = subsens[i]

    for i in range(len(subsens)):
        for j in range(len(subsens)):
            if subsens[i] == subsens[j]:
                continue
            elif is_sublist(subsens[j], subsens[i]):
                subsens[j] = subsens[i]
    subsens = list(map(list,set(map(tuple,subsens))))

    res = " ".join(overlap_merge(subsens))


    if ("_" in res or ("?" in res and "?" != res[-1])) and stat == False:
        res = sign_translator(text_used, True)

    return correct_sentence(res)

# Flask App Configuration
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'webm', 'mp4', 'avi', 'mov', 'mkv'}

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """Home route to check if server is running"""
    return jsonify({
        'message': 'Flask Sign Language Recognition Server is running!',
        'endpoints': {
            '/predict': 'POST - Upload video for sign language recognition',
            '/health': 'GET - Health check'
        },
        'supported_actions': actions
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'message': 'Sign Language Recognition Server is running',
        'model_loaded': True,
        'supported_actions_count': len(actions)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Process uploaded video and return sign language translation result"""
    try:
        # Check if the request has the file part
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No video file provided'
            }), 400

        file = request.files['video']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No video file selected'
            }), 400

        # Check file extension
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            timestamp = str(int(time.time()))
            filename = f"{timestamp}_{filename}"
            
            # Save the file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get file size for logging
            file_size = os.path.getsize(filepath)
            
            print(f"วิดีโอที่ได้รับ: {filename}")
            print(f"ขนาดไฟล์: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
            print(f"บันทึกที่: {filepath}")
            
            # Process the video with sign language recognition
            print("🤖 เริ่มประมวลผลภาษามือ...")
            start_time = time.time()
            
            final_sentence, avg_confidence = process_video(filepath)
            
            processing_time = time.time() - start_time
            print(f"⏱️ เวลาในการประมวลผล: {processing_time:.2f} วินาที")
            
            # Clean up - delete the uploaded file
            try:
                os.remove(filepath)
                print(f"🗑️ ลบไฟล์ชั่วคราว: {filename}")
            except Exception as e:
                print(f"⚠️ ไม่สามารถลบไฟล์ได้: {e}")
                
            
            
            # Prepare response
            if final_sentence:
                full_sentence = sign_translator(f"{final_sentence} ?",False)
                translated = translator.translate(full_sentence, src='en', dest='th')
                
                response_data = {
                    'success': True,
                    'thai_text': translated.text,
                    'english_text': full_sentence,  # You can add English translation here
                    'confidence': float(round(avg_confidence, 3)),
                    'message': 'วิดีโอได้ถูกประมวลผลเรียบร้อยแล้ว',
                    'filename': filename,
                    'file_size': file_size,
                    'processing_time': round(processing_time, 2)
                }
            else:
                response_data = {
                    'success': True,
                    'thai_text': 'ไม่พบภาษามือที่สามารถจดจำได้',
                    'english_text': 'No recognizable sign language detected',
                    'confidence': float(0.0),
                    'message': 'ไม่พบท่าทางภาษามือที่สามารถจดจำได้ในวิดีโอ',
                    'filename': filename,
                    'file_size': file_size,
                    'processing_time': round(processing_time, 2)
                }
            
            print(f"ส่งผลลัพธ์: {response_data['thai_text']}")
            return jsonify(response_data)
        
        else:
            return jsonify({
                'success': False,
                'error': 'ประเภทไฟล์ไม่ถูกต้อง ประเภทที่รองรับ: ' + ', '.join(ALLOWED_EXTENSIONS)
            }), 400
            
    except Exception as e:
        print(f"❌ ข้อผิดพลาดในการประมวลผลวิดีโอ: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'ข้อผิดพลาดของเซิร์ฟเวอร์: {str(e)}'
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'ไฟล์ใหญ่เกินไป ขนาดสูงสุดคือ 100MB'
    }), 413

if __name__ == '__main__':
    print("🚀 เริ่ม Flask Sign Language Recognition Server...")
    print(f"📁 โฟลเดอร์อัพโหลด: {UPLOAD_FOLDER}")
    print(f"📋 ประเภทไฟล์ที่รองรับ: {ALLOWED_EXTENSIONS}")
    print(f"🤖 จำนวนท่าทางที่รองรับ: {len(actions)}")
    print("🌐 เซิร์ฟเวอร์จะรันที่: http://localhost:5001")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',  # Listen on all interfaces
        port=5001,
        debug=True
    )