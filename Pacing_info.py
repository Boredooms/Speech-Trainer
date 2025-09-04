import time
from collections import deque, Counter

# --- Constants ---
WPM_IDEAL_MIN = 140
WPM_IDEAL_MAX = 160
PAUSE_THRESHOLD_SECONDS = 2.0
WPM_CALCULATION_MIN_DURATION_SECONDS = 1.5
REPETITION_PHRASE_LENGTH = 3
CONFIDENCE_THRESHOLD = 0.80  # Words with confidence below this will be ignored

# Expanded list of single-word fillers for more accurate detection,
# as multi-word phrase detection is not supported by the current logic.
FILLER_WORDS = {
    "uh", "um", "er", "ah", "hmm", "so", "well", "right", "literally", "okay",
    "anyway", "see", "just", "really", "like", "actually", "basically", "mean",
    "guess", "suppose", "think", "honest", "totally", "simply", "personally",
    "seriously", "truly", "virtually", "apparently"
}

class PaceAnalyzer:
    """
    A simplified and more robust analysis engine.
    - Session stats are updated only from final, accurate results to prevent double-counting.
    - Live counters no longer reset on pauses, providing a more stable real-time experience.
    """
    def __init__(self, window_size_seconds=10):
        self.window_size = window_size_seconds
        
        self.word_timestamps = deque()
        
        # Session-wide statistics, updated only from final results
        self.session_total_words = 0
        self.session_filler_words = 0
        self.session_total_confidence = 0.0
        self.session_words_with_confidence = 0
        self.session_repetitive_phrases = 0
        self.phrase_counts = Counter()

        # Live data for the current, unconfirmed utterance
        self.live_words = []
        self.live_filler_count = 0
        self.live_repetitive_phrase_count = 0
        self.last_word_time = 0
        self.smoothed_wpm = 0.0
        self.last_partial_text = ""

    def _cleanup(self, current_time):
        """Removes word timestamps older than the analysis window."""
        while self.word_timestamps and (current_time - self.word_timestamps[0] > self.window_size):
            self.word_timestamps.popleft()

    def process_final_result(self, word_list):
        """
        Processes a final, confirmed result. This is the single source of truth
        for updating all session-wide statistics.
        """
        current_time = time.time()
        self.last_word_time = current_time
        
        # The live utterance is now complete. Reset live data.
        self.live_words, self.live_filler_count, self.live_repetitive_phrase_count, self.last_partial_text = [], 0, 0, ""
        
        # Filter words by confidence before processing
        confirmed_words = [d for d in word_list if d.get('conf', 0) >= CONFIDENCE_THRESHOLD]
        
        final_words = [d.get('word', '').lower() for d in confirmed_words if d.get('word')]
        
        # Update session totals from the final, accurate words
        self.session_total_words += len(final_words)
        self.session_filler_words += sum(1 for word in final_words if word in FILLER_WORDS)
        
        if len(final_words) >= REPETITION_PHRASE_LENGTH:
            for i in range(len(final_words) - REPETITION_PHRASE_LENGTH + 1):
                phrase = tuple(final_words[i:i + REPETITION_PHRASE_LENGTH])
                if self.phrase_counts[phrase] > 0:
                    self.session_repetitive_phrases += 1
                self.phrase_counts[phrase] += 1

        for word_data in confirmed_words:
            confidence = word_data.get('conf')
            if confidence is not None:
                self.session_total_confidence += confidence
                self.session_words_with_confidence += 1
            
            self.word_timestamps.append(word_data.get('start', current_time))

    def process_partial_result(self, partial_text):
        """
        Updates the live analysis based on the latest partial transcript.
        """
        current_time = time.time()
        partial_text = partial_text.strip()
        if partial_text == self.last_partial_text: return

        self.last_word_time = current_time
        self.last_partial_text = partial_text
        new_words = partial_text.lower().split()
        
        diff = len(new_words) - len(self.live_words)
        if diff > 0:
            for _ in range(diff): self.word_timestamps.append(current_time)
        elif diff < 0:
            for _ in range(abs(diff)):
                if self.word_timestamps: self.word_timestamps.pop()
        
        self.live_words = new_words
        
        self.live_filler_count = sum(1 for word in self.live_words if word in FILLER_WORDS)
        
        self.live_repetitive_phrase_count = 0
        if len(self.live_words) >= REPETITION_PHRASE_LENGTH:
            for i in range(len(self.live_words) - REPETITION_PHRASE_LENGTH + 1):
                phrase = tuple(self.live_words[i:i + REPETITION_PHRASE_LENGTH])
                if self.phrase_counts[phrase] > 0:
                    self.live_repetitive_phrase_count += 1

    def get_analysis(self, current_time):
        """
        Returns a dictionary of all speech metrics.
        """
        self._cleanup(current_time)
        
        # On a pause, only reset the WPM calculator, not the live word counts.
        if self.last_word_time > 0 and current_time - self.last_word_time > PAUSE_THRESHOLD_SECONDS:
            self.word_timestamps.clear()
            self.last_word_time = 0

        word_count_in_window = len(self.word_timestamps)
        raw_wpm = 0
        if word_count_in_window > 2:
            duration = current_time - self.word_timestamps[0]
            if duration >= WPM_CALCULATION_MIN_DURATION_SECONDS:
                raw_wpm = (word_count_in_window / duration) * 60
        
        smoothing_factor = 0.3 if raw_wpm > 0 else 0.5
        self.smoothed_wpm = (smoothing_factor * raw_wpm) + ((1 - smoothing_factor) * self.smoothed_wpm)
        if self.smoothed_wpm < 5: self.smoothed_wpm = 0
        
        clarity = (self.session_total_confidence / self.session_words_with_confidence) if self.session_words_with_confidence > 0 else 1.0
            
        feedback = "..."
        if self.smoothed_wpm > 10:
            if self.smoothed_wpm < WPM_IDEAL_MIN: feedback = "A bit slow"
            elif self.smoothed_wpm > WPM_IDEAL_MAX: feedback = "A bit fast"
            else: feedback = "Ideal pace"

        # Display values are a combination of confirmed session data and current live data
        display_total_words = self.session_total_words + len(self.live_words)
        display_filler_words = self.session_filler_words + self.live_filler_count
        display_repetitive_phrases = self.session_repetitive_phrases + self.live_repetitive_phrase_count

        return {
            "wpm": int(self.smoothed_wpm),
            "pacing_feedback": feedback,
            "clarity_score": clarity * 100,
            "filler_words": display_filler_words,
            "total_words": display_total_words,
            "repetitive_phrases": display_repetitive_phrases
        }