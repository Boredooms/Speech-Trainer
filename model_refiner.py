import os
import json
from vosk import Model, SpkModel, KaldiRecognizer

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPK_MODEL_PATH = os.path.join(SCRIPT_DIR, "vosk-model-spk-0.4")

# --- Model Selection ---
# Switched to the 'zamia' model, which may offer better accuracy.
# The original model was "vosk-model-small-en-us-0.15".
MODEL_NAME = "vosk-model-small-en-us-zamia-0.5"
MODEL_PATH = os.path.join(SCRIPT_DIR, MODEL_NAME)

class ModelManager:
    """
    Manages loading the Vosk speech model, speaker model, and creating recognizers.
    This centralizes model and grammar configuration.
    """
    def __init__(self):
        """
        Initializes the manager and loads the main speech model.
        """
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Main model not found at '{MODEL_PATH}'")
        
        print(f"[ModelManager] Using model: {MODEL_NAME}")
        print("[ModelManager] Loading main speech model...")
        self.model = Model(MODEL_PATH)
        self.spk_model = self._load_speaker_model()
        self.grammar = self._get_interview_grammar()
        print("[ModelManager] Initialization complete.")

    def _load_speaker_model(self):
        """
        Loads the speaker adaptation model if it exists.
        """
        if os.path.exists(SPK_MODEL_PATH):
            print("[ModelManager] Loading speaker adaptation model...")
            try:
                spk_model = SpkModel(SPK_MODEL_PATH)
                print("[ModelManager] Speaker adaptation model loaded successfully.")
                return spk_model
            except Exception as e:
                print(f"[ModelManager] Warning: Error loading speaker model: {e}. Continuing without it.")
        else:
            print("[ModelManager] Speaker model not found. Proceeding without speaker adaptation.")
        return None

    def _get_interview_grammar(self):
        """
        Returns a JSON string of a predefined grammar for typical interview topics.
        This helps the recognizer favor these words, improving accuracy.
        """
        # interview_keywords = [
        #     "experience", "skills", "strengths", "weaknesses", "teamwork", "leadership",
        #     "problem-solving", "communication", "project", "challenge", "success",
        #     "career goals", "organization", "industry", "achievement", "qualification",
        #     "tell me about yourself", "why should we hire you", "questions for me",
        #     "[unk]"  # Allow for unknown words
        # ]
        # print("[ModelManager] Using interview-focused grammar.")
        # return json.dumps(interview_keywords)
        print("[ModelManager] No restrictive grammar being used.")
        return None

    def create_recognizer(self, sample_rate):
        """
        Creates a configured KaldiRecognizer instance.

        Args:
            sample_rate (int): The audio sample rate.

        Returns:
            KaldiRecognizer: A configured recognizer instance.
        """
        # Create recognizer without the restrictive grammar
        recognizer = KaldiRecognizer(self.model, sample_rate)
        
        # Set speaker model if it exists
        if self.spk_model:
            recognizer.SetSpkModel(self.spk_model)
            print("[ModelManager] Speaker model applied to recognizer.")

        recognizer.SetWords(True)
        print("[ModelManager] KaldiRecognizer created.")
        return recognizer

# --- Test block for standalone execution ---
if __name__ == '__main__':
    print("--- Testing ModelManager ---")
    try:
        # This test now uses the centrally defined MODEL_PATH
        if os.path.exists(MODEL_PATH):
            manager = ModelManager()
            recognizer = manager.create_recognizer(sample_rate=16000)
            if recognizer:
                print("\nSUCCESS: ModelManager initialized and created a recognizer.")
            else:
                print("\nERROR: Failed to create a recognizer.")
        else:
            print(f"\nSKIPPING TEST: Main model not found at '{MODEL_PATH}'.")
            print("Download the model to run the test.")
            
    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")

    print("\n--- Test Complete ---")