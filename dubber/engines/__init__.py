"""Model engines behind small interfaces; each has a light stand-in so the whole pipeline runs (and is tested) on a CPU.

==============  ===========================================  ==========================================
stage           real engine                                  stand-in (CPU, no model)
==============  ===========================================  ==========================================
VAD             Silero VAD                                   frame-energy detector
separation      TIGER-DnR / Mel-Band RoFormer                "none": no stems, the original is ducked
ASR             faster-whisper large-v3-turbo                "none" (subtitles only) / "mock"
diarization     pyannote community-1 (HF token)              MFCC clustering of the lines
translation     Opus-MT (Marian)                             "mock": marks the text
TTS             Qwen3-TTS (FA2 -> CUDA Graphs -> SDPA)       "mock": speech-like tone of the right length
==============  ===========================================  ==========================================
"""
