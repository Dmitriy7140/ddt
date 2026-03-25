import cv2
import pandas as pd
import argparse
import time

# -----------------------------
# Константы состояний
# -----------------------------
EMPTY = "EMPTY"
OCCUPIED = "OCCUPIED"
APPROACH = "APPROACH"

# -----------------------------
# Аргументы
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True, help="Path to video file")
args = parser.parse_args()

# -----------------------------
# Видео
# -----------------------------
cap = cv2.VideoCapture(args.video)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# -----------------------------
# ROI выбор столика
# -----------------------------
ret, frame = cap.read()
if not ret:
    raise Exception("Не удалось прочитать видео")

roi = cv2.selectROI("Выбери столик", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Выбери столик")

x, y, w, h = roi

# -----------------------------
# Видео writer
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

# -----------------------------
# Детектор движения
# -----------------------------
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

# -----------------------------
# Переменные состояния
# -----------------------------
current_state = EMPTY
last_state = None
last_empty_time = None

events = []

frame_idx = 0

# -----------------------------
# Основной цикл
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = frame_idx / fps

    # ROI
    table_roi = frame[y:y+h, x:x+w]

    # Детекция движения
    fg_mask = bg_subtractor.apply(table_roi)

    # Убираем шум
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    motion_pixels = cv2.countNonZero(thresh)

    # Эвристика (настроечный параметр)
    motion_threshold = 500

    if motion_pixels > motion_threshold:
        detected_state = OCCUPIED
    else:
        detected_state = EMPTY

    # -----------------------------
    # Логика переходов
    # -----------------------------
    if detected_state != current_state:
        last_state = current_state
        current_state = detected_state

        # событие APPROACH
        if last_state == EMPTY and current_state == OCCUPIED:
            events.append({
                "event": APPROACH,
                "time": timestamp
            })

            if last_empty_time is not None:
                delay = timestamp - last_empty_time
                events.append({
                    "event": "DELAY",
                    "time": timestamp,
                    "delay": delay
                })

        # фиксация EMPTY
        if current_state == EMPTY:
            last_empty_time = timestamp
            events.append({
                "event": EMPTY,
                "time": timestamp
            })

        # фиксация OCCUPIED
        if current_state == OCCUPIED:
            events.append({
                "event": OCCUPIED,
                "time": timestamp
            })

    # -----------------------------
    # Визуализация
    # -----------------------------
    if current_state == EMPTY:
        color = (0, 255, 0)  # зеленый
    else:
        color = (0, 0, 255)  # красный

    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.putText(frame, current_state, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    out.write(frame)

    frame_idx += 1

# -----------------------------
# Аналитика
# -----------------------------
df = pd.DataFrame(events)

delays = df[df["event"] == "DELAY"]

if not delays.empty:
    avg_delay = delays["delay"].mean()
else:
    avg_delay = None

print("\n===== RESULT =====")
print(df)

if avg_delay:
    print(f"\nСреднее время: {avg_delay:.2f} сек")
else:
    print("\nНедостаточно данных для расчета")

# -----------------------------
# Сохранение отчета
# -----------------------------
df.to_csv("events.csv", index=False)

cap.release()
out.release()
cv2.destroyAllWindows()