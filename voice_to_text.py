import cv2
import json
import numpy as np
import pyaudio
import queue
import threading
import time
import os
import textwrap
from Pacing_info import PaceAnalyzer, CONFIDENCE_THRESHOLD
from model_refiner import ModelManager, MODEL_PATH

# --- Constants ---
# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH is now imported from model_refiner to ensure consistency.

# Audio
SAMPLE_RATE = 16000
CHUNK_SIZE = 8192

# Video & UI
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 540
UI_FONT = cv2.FONT_HERSHEY_SIMPLEX
UI_FONT_COLOR = (255, 255, 255)
UI_BG_COLOR = (0, 0, 0)
UI_PANEL_ALPHA = 0.6

# --- Helper Functions ---

def draw_text_with_outline(frame, text, pos, scale=0.8, color=(255, 255, 255), thickness=1):
    """Draws text with a black outline for better visibility."""
    x, y = pos
    # Draw the black outline
    cv2.putText(frame, text, (x, y), UI_FONT, scale, (0, 0, 0), thickness * 2, cv2.LINE_AA)
    # Draw the main text
    cv2.putText(frame, text, (x, y), UI_FONT, scale, color, thickness, cv2.LINE_AA)

def draw_ui(frame, stats):
    """Draws the main statistics panel on the frame."""
    # Create a semi-transparent background for the panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (340, 160), UI_BG_COLOR, -1)
    frame = cv2.addWeighted(overlay, UI_PANEL_ALPHA, frame, 1 - UI_PANEL_ALPHA, 0)

    # Display stats
    wpm_text = f"WPM: {stats.get('wpm', 0)}"
    pace_text = f"Pace: {stats.get('pacing_feedback', '...')}"
    clarity_text = f"Clarity: {stats.get('clarity_score', 100):.0f}%"
    words_text = f"Total Words: {stats.get('total_words', 0)}"
    fillers_text = f"Filler Words: {stats.get('filler_words', 0)}"

    draw_text_with_outline(frame, wpm_text, (20, 45))
    draw_text_with_outline(frame, pace_text, (20, 75))
    draw_text_with_outline(frame, clarity_text, (20, 105))
    draw_text_with_outline(frame, f"{words_text} | {fillers_text}", (20, 135), scale=0.7)
    return frame

# --- Worker Thread ---

def audio_worker(q, stop_event):
    """
    The heart of the application. Runs in a separate thread to handle all
    audio processing and speech analysis, preventing the UI from freezing.
    """
    try:
        # --- Initialization ---
        print("[Audio Thread] Initializing...")
        # ModelManager now handles model selection internally.
        model_manager = ModelManager()
        recognizer = model_manager.create_recognizer(SAMPLE_RATE)
        
        analyzer = PaceAnalyzer(window_size_seconds=10)
        
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                        input=True, frames_per_buffer=CHUNK_SIZE)
        stream.start_stream()
        
        print("[Audio Thread] Ready and listening.")
        q.put({'type': 'status', 'message': 'Ready'})
        
        last_stats_update = 0

        # --- Main Loop ---
        while not stop_event.is_set():
            data = stream.read(4096, exception_on_overflow=False)
            
            # Feed audio to the recognizer
            if recognizer.AcceptWaveform(data):
                res = json.loads(recognizer.Result())
                if res.get('text'):
                    final_words = res.get('result', [])
                    analyzer.process_final_result(final_words)
                    q.put({"type": "final", "words": final_words})
            else:
                partial_res = json.loads(recognizer.PartialResult())
                partial_text = partial_res.get('partial', '')
                if partial_text:
                    analyzer.process_partial_result(partial_text)
                    q.put({"type": "partial", "text": partial_text})

            # Periodically send a full analysis update to the main thread
            current_time = time.time()
            if current_time - last_stats_update > 0.25: # 4 updates per second
                stats = analyzer.get_analysis(current_time)
                q.put({"type": "stats", "data": stats})
                last_stats_update = current_time

    except Exception as e:
        # Report any fatal error to the main thread
        import traceback
        error_msg = f"Error in audio thread: {e}\n{traceback.format_exc()}"
        q.put({'type': 'error', 'message': error_msg})
    finally:
        # --- Cleanup ---
        print("[Audio Thread] Cleaning up...")
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        if 'p' in locals():
            p.terminate()
        print("[Audio Thread] Stopped.")

# --- Main Application ---

def main():
    """
    Sets up the main application window, camera, and threads.
    Handles UI rendering and user input.
    """
    # --- Initialization ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("FATAL: Cannot open camera. Exiting.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

    q = queue.Queue()
    stop_event = threading.Event()
    
    print("[Main Thread] Starting audio worker...")
    audio_thread = threading.Thread(target=audio_worker, args=(q, stop_event))
    audio_thread.start()

    # --- Application State ---
    final_transcript_words = []
    partial_transcript = ""
    current_stats = {}
    status_message = "Initializing..."

    # --- Main Loop ---
    try:
        while True:
            # Check for messages from the audio thread
            try:
                msg = q.get_nowait()
                if msg['type'] == 'stats':
                    current_stats = msg['data']
                elif msg['type'] == 'partial':
                    partial_transcript = msg['text']
                elif msg['type'] == 'final':
                    # Filter words by the same confidence threshold before adding to the transcript
                    confident_words = [word for word in msg['words'] if word.get('conf', 0) >= CONFIDENCE_THRESHOLD]
                    final_transcript_words.extend(confident_words)
                    partial_transcript = "" # Clear partial text on final result
                elif msg['type'] == 'status':
                    status_message = msg['message']
                elif msg['type'] == 'error':
                    print(f"FATAL ERROR from audio thread: {msg['message']}")
                    break
            except queue.Empty:
                pass # No new messages

            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame from camera. Exiting...")
                break
            
            frame = cv2.flip(frame, 1) # Flip horizontally for a mirror effect

            # Draw the UI
            frame = draw_ui(frame, current_stats)

            # --- Draw Captions ---
            # Limit transcript history to avoid filling the screen
            MAX_WORDS_ON_SCREEN = 20
            display_words = final_transcript_words[-MAX_WORDS_ON_SCREEN:]
            
            full_text = " ".join([w.get('word', '') for w in display_words])
            if partial_transcript:
                # Add an ellipsis to show it's a live transcript
                full_text += " " + partial_transcript + "..."

            # Wrap text to fit the window width
            wrapped_text = textwrap.wrap(full_text, width=60) # Adjust width as needed
            
            # Display the last 2 lines of captions
            caption_y_start = WINDOW_HEIGHT - 80
            line_height = 35
            for i, line in enumerate(wrapped_text[-2:]):
                y_pos = caption_y_start + (i * line_height)
                # Use the corrected helper function to draw captions
                draw_text_with_outline(frame, line.strip(), (20, y_pos), scale=0.9)

            # Show the main window
            cv2.imshow('Real-Time Speech Analysis', frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Main Thread] 'q' pressed. Shutting down.")
                break
    finally:
        # --- Cleanup ---
        print("[Main Thread] Stopping audio thread...")
        stop_event.set()
        audio_thread.join(timeout=5) # Wait for the thread to finish
        
        cap.release()
        cv2.destroyAllWindows()
        print("[Main Thread] Application closed.")

if __name__ == "__main__":
    # Check for model existence before starting, using the path from model_refiner.
    if not os.path.exists(MODEL_PATH):
        print(f"FATAL: Vosk model not found at '{MODEL_PATH}'")
        print("Please download the model and place it in the correct directory.")
    else:
        main()