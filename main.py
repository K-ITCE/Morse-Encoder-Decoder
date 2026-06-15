import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, medfilt
from scipy.io import wavfile
import os
from datetime import datetime

codes = {
  "a":".-",
  "b":"-...",
  "c":"-.-.",
  "d":"-..",
  "e":".",
  "f":"..-.",
  "g":"--.",
  "h":"....",
  "i":"..",
  "j":".---",
  "k":"-.-",
  "l":".-..",
  "m":"--",
  "n":"-.",
  "o":"---",
  "p":".--.",
  "q":"--.-",
  "r":".-.",
  "s":"...",
  "t":"-",
  "u":"..-",
  "v":"...-",
  "w":".--",
  "x":"-..-",
  "y":"-.--",
  "z":"--..",
  "1":".----",
  "2":"..---",
  "3":"...--",
  "4":"....-",
  "5":".....",
  "6":"-....",
  "7":"--...",
  "8":"---..",
  "9":"----.",
  "0":"-----",
  ".":".-.-.-",
  ",":"--..--",
  "?":"..--..",
  "/":"-..-.",
  "@":".--.-.",
  "'":".----.",
  "\"":".-..-.",
  "-":"-....-",
  "+":".-.-.",
  "=":"-...-",
  "(":"-.--..",
  ")":"-.--.-",
  "&":".-...",
  ":":"---...",
  ";":"-.-.-.",
  "!":"-.-.--"
}

def estimate_timing_parameters(intervals, verbose=False):
    """Simplified timing estimation using percentiles instead of KMeans."""
    tone_intervals = intervals[intervals['value'] == 1]
    tone_durs = tone_intervals['end'] - tone_intervals['start']
    
    silence_intervals = intervals[intervals['value'] == 0]
    silence_durs = silence_intervals['end'] - silence_intervals['start']
    
    if len(tone_durs) == 0:
        raise ValueError("No tone blocks found")
    
    # Find dot duration as the smallest/most common tone
    sorted_tones = sorted(tone_durs)
    
    # Find the gap between tone clusters (dots vs dashes)
    # Look for the biggest jump in sorted tone durations
    max_gap = 0
    gap_idx = 0
    for i in range(1, len(sorted_tones)):
        gap = sorted_tones[i] - sorted_tones[i-1]
        if gap > max_gap:
            max_gap = gap
            gap_idx = i
    
    # Dots are all tones before the largest gap
    unit = np.median(sorted_tones[:gap_idx])
    dot_max = unit * 1.2
    dash_min = unit * 1.5
    
    if len(silence_durs) == 0:
        return unit, dot_max, dash_min, unit, unit * 3
        
    # Split silences into two clusters (intra-char vs inter-letter)
    sorted_silences = sorted(silence_durs)
    midpoint_idx = len(sorted_silences) // 2
    short_gap_max = np.median(sorted_silences[:midpoint_idx]) * 2.0
    med_gap_max = np.median(sorted_silences[midpoint_idx:]) * 1.5
    
    if verbose:
        print(f"Estimated unit (dot length): {unit:.4f} s")
        print(f"  Dot: <= {dot_max:.4f} s")
        print(f"  Dash: >= {dash_min:.4f} s")
    
    return unit, dot_max, dash_min, short_gap_max, med_gap_max

def text_to_morse(input_text):
  counter = 0
  result = ""
  while counter < len(input_text):
    if input_text[counter] == " ":
      result += '/'
    else:
      try:
        result += f'{codes[input_text[counter].lower()]} '
      except:
        result += f'$ '
      
    counter +=1  
  return result

def morse_to_text(input_code):
  counter = 0
  result = ""
  while counter < len(input_code):
    code = ""
    while counter < len(input_code) and input_code[counter] != " " and input_code[counter] != "/":
      code += input_code[counter]
      counter+=1
    
    for letter,key_code in codes.items():
      if key_code == code:
        result += letter
        break
    
    if counter < len(input_code):
      if input_code[counter] == " ":
          counter += 1
      elif input_code[counter] == "/":
          result += " "
          counter += 1
  return result

def morse_to_audio(morse_code, output_file, dot_duration=0.1, tone_freq=1000, sr=22050):
  """
  Convert Morse code string to audio file (WAV) with corresponding beeps.
  
  Parameters:
  - morse_code: String like ".... . .-.. .-.. ---" (dots, dashes, spaces, slashes)
  - output_file: Path to save the WAV file
  - dot_duration: Duration of a dot in seconds (default 0.1s)
  - tone_freq: Frequency of the beep tone in Hz (default 800 Hz)
  - sr: Sample rate (default 22050 Hz)
  
  Returns: None (saves WAV file)
  """
  dash_duration = dot_duration * 3  # A dash is 3x longer than a dot
  intra_char_gap = dot_duration    # Gap within a character
  inter_letter_gap = dot_duration * 3  # Gap between letters
  inter_word_gap = dot_duration * 7    # Gap between words
  
  # Generate sine wave tone with fade in/out envelope
  def generate_tone(duration, freq, sr, fade_duration=0.005):
    """
    Generate a sine wave tone with fade in/out envelope to reduce clicks.
    
    Parameters:
    - duration: Duration of the tone in seconds
    - freq: Frequency in Hz
    - sr: Sample rate
    - fade_duration: Fade in/out duration in seconds (default 0.005s = 5ms)
    """
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples)
    tone = np.sin(2 * np.pi * freq * t)
    
    # Create envelope for fade in/out
    envelope = np.ones(n_samples, dtype=np.float32)
    fade_samples = int(sr * fade_duration)
    
    # Fade in (linear)
    if fade_samples > 0 and fade_samples < n_samples:
      envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    
    # Fade out (linear)
    if fade_samples > 0 and fade_samples < n_samples:
      envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    
    # Apply envelope to the tone
    tone = tone * envelope
    return tone.astype(np.float32)
  
  # Generate silence
  def generate_silence(duration, sr):
    return np.zeros(int(sr * duration), dtype=np.float32)
  
  audio = []
  i = 0
  while i < len(morse_code):
    char = morse_code[i]
    
    if char == '.':
      audio.append(generate_tone(dot_duration, tone_freq, sr))
      audio.append(generate_silence(intra_char_gap, sr))
    elif char == '-':
      audio.append(generate_tone(dash_duration, tone_freq, sr))
      audio.append(generate_silence(intra_char_gap, sr))
    elif char == ' ':
      # Space between letters - remove one intra-char gap already added, add inter-letter gap
      if audio:
        audio.pop()  # Remove the last intra-char gap
      audio.append(generate_silence(inter_letter_gap, sr))
    elif char == '/':
      # Slash for word boundary
      if audio:
        audio.pop()  # Remove the last inter-letter gap
      audio.append(generate_silence(inter_word_gap, sr))
    
    i += 1
  
  # Concatenate all audio segments
  audio_data = np.concatenate(audio)
  
  # Normalize to prevent clipping
  max_val = np.max(np.abs(audio_data))
  if max_val > 1.0:
    audio_data = audio_data / max_val
  
  # Convert to 16-bit PCM
  audio_data = (audio_data * 32767).astype(np.int16)
  
  # Write to WAV file
  wavfile.write(output_file, sr, audio_data)

def text_to_audio(text, output_file, dot_duration=0.05, tone_freq=600, sr=22050):
  """
  Convert text to Morse audio file (complete pipeline).
  
  Parameters:
  - text: Input text to convert
  - output_file: Path to save the WAV file
  - dot_duration: Duration of a dot in seconds
  - tone_freq: Frequency of the beep tone in Hz
  - sr: Sample rate in Hz
  """
  morse_code = text_to_morse(text)
  morse_to_audio(morse_code, output_file, dot_duration, tone_freq, sr)
  print(f"✓ Morse code: {morse_code}")
  return morse_code

def audio_to_morse(file, smooth_kernel=101, threshold=None, plot=False):
    """
    Convert a Morse code audio file to a Morse string (dots, dashes, spaces, slashes).
    Returns a string like: "-- .-- -.- .- -- . .. ... ...."
    """
    # 1. Load and compute envelope (Hilbert + median filter)
    y, sr = librosa.load(file, mono=True)
    analytic = hilbert(y)
    envelope = np.abs(analytic)
    envelope = medfilt(envelope, kernel_size=smooth_kernel)
    if threshold is None:
        threshold = 0.3 * np.max(envelope)
    binary = (envelope > threshold).astype(int)
    time = np.arange(len(binary)) / sr

    # 2. Extract contiguous 0/1 intervals
    diffs = np.diff(binary.astype(int), prepend=1 - binary[0])
    change_idx = np.where(diffs != 0)[0]
    starts = time[change_idx]
    ends = time[np.append(change_idx[1:] - 1, len(binary) - 1)]
    values = binary[change_idx]
    intervals = pd.DataFrame({'start': starts, 'end': ends, 'value': values}).dropna()

    # 3. Determine timing parameters (simplified percentile-based approach)
    unit, dot_max, dash_min, short_gap_max, med_gap_max = estimate_timing_parameters(intervals, verbose=False)

    # 4. Build Morse string
    morse_parts = []      # will contain letters and '/'
    current_letter = []

    for idx, row in intervals.iterrows():
        dur = row['end'] - row['start']
        if row['value'] == 1:      # tone
            if dur <= dot_max:
                symbol = '.'
                current_letter.append('.')
            else:                  # dash
                symbol = '-'
                current_letter.append('-')
        else:                       # silence
            if dur <= short_gap_max:
                symbol = 'gap-intra'
                continue
            elif dur <= med_gap_max:
                symbol = 'gap-letter'
                if current_letter:
                    morse_parts.append(''.join(current_letter))
                    current_letter = []
            else:
                symbol = 'gap-word'
                if current_letter:
                    morse_parts.append(''.join(current_letter))
                    current_letter = []
                morse_parts.append('/')

    # Append last letter if any
    if current_letter:
        morse_parts.append(''.join(current_letter))

    # Join with spaces between letters (slashes remain as they are)
    morse_string = ' '.join(morse_parts).replace('/ ', '/').replace(' /', '/')
    
    if plot:
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        
        # Plot 1: Raw audio signal
        axes[0].plot(time, y, alpha=0.7, color='blue')
        axes[0].set_ylabel('Raw Audio')
        axes[0].set_title('Raw Audio Signal')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Hilbert envelope with threshold
        axes[1].plot(time, envelope, color='red', linewidth=1.5, label='Envelope')
        axes[1].axhline(threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title('Envelope (Hilbert Transform) + Threshold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Binary signal
        axes[2].plot(time, binary, drawstyle='steps-post', color='green', linewidth=2)
        axes[2].set_ylabel('Binary Signal')
        axes[2].set_title('Binary Signal (Thresholded)')
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Interval durations in chronological order
        interval_durs = intervals['end'] - intervals['start']
        interval_colors = ['red' if v == 1 else 'blue' for v in intervals['value']]
        
        axes[3].bar(range(len(intervals)), interval_durs, color=interval_colors, alpha=0.6)
        
        # Mark thresholds
        axes[3].axhline(dot_max, color='purple', linestyle='--', linewidth=2, label=f'Dot/Dash threshold ({dot_max:.4f}s)')
        axes[3].axhline(short_gap_max, color='cyan', linestyle=':', linewidth=2, label=f'Intra-letter gap ({short_gap_max:.4f}s)')
        axes[3].axhline(med_gap_max, color='brown', linestyle=':', linewidth=2, label=f'Inter-letter gap ({med_gap_max:.4f}s)')
        
        axes[3].set_ylabel('Duration (s)')
        axes[3].set_xlabel('Interval Index (in order)')
        axes[3].set_title('All Intervals with Classification')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('plots/morse_analysis.png', dpi=100, bbox_inches='tight')
        print("✓ Plots saved to plots/morse_analysis.png")
        plt.close()

    return morse_string

def audio_to_text(file, plot=False):
    """
    Convert a Morse code audio file directly to English text.
    Uses your own `morsetotext` function for the final decoding.
    """
    morse_str = audio_to_morse(file, plot=plot)
    return morse_to_text(morse_str)

if __name__ == "__main__":
    while True:
        print("\n--- Morse Code Converter ---")
        print("1. Encode (text to audio)")
        print("2. Decode (audio to text)")
        print("0. Exit")
        choice = input("Enter choice: ")
        
        if choice == '1':
            text = input("Enter text message: ")
            speed = input("Speed multiplier (higher = faster) [1.0]: ")
            freq = input("Pitch in Hz [600]: ")
            
            try:
                dot_duration = 0.1 / float(speed) if speed else 0.1
                tone_freq = int(freq) if freq else 600
            except:
                dot_duration = 0.1
                tone_freq = 600
            
            timestamp = datetime.now().strftime("%H%M%S")
            output_file = f"wav/morse_{timestamp}.wav"
            text_to_audio(text, output_file, dot_duration=dot_duration, tone_freq=tone_freq)
            print(f"✓ Audio saved to {output_file}")
        elif choice == '2':
            filepath = input("Enter WAV file path: ")
            plot = input("Save plots? (y/[n]): ").lower() == 'y'
            
            if os.path.exists(filepath):
                try:
                    result = audio_to_text(filepath, plot=plot)
                    print(f"Decoded text: {result}")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("File not found.")
        elif choice == '0':
            break
        else:
            print("Invalid choice.")