Architecture & Design Notes

Two defined versions


V1 (this build): phone camera only. AI detects face/beard,
generates the green/orange/red trim map, shows it live on the phone
camera during trimming (re-aligned to your face every frame), and
gives visual + voice guidance. No physical hardware, no blade
tracking — guidance is based on face position and overall zone
proportions, not exact blade location.
V2 (future): adds LED markers physically attached to the manual
trimmer. The phone camera detects the markers directly, giving actual
blade position/direction instead of inferring it from unmarked video.
This meaningfully de-risks the hardest CV problem in the whole
project (see below) but requires actual hardware (LEDs on the
trimmer) and is a separate phase, not built here.


Why V1 still doesn't do "blade tracking"

Markerless blade/hand tracking from video alone — figuring out exactly
where an ordinary, unmarked trimmer blade is in 3D space — is a hard,
largely unsolved consumer-CV problem. V1 sidesteps it entirely: it
tracks your face frame-to-frame (a solved problem via MediaPipe) and
re-projects the zone map onto your current head position/rotation.
Guidance is phrased in terms of "how much red/yellow is currently
visible" rather than "your blade is 3mm off the line" — that pinpoint
accuracy is what V2's LED markers are for.

Module responsibility map

ModuleResponsibilityRisk levelcapture/image_input.pyLoad photo or grab one webcam frameLowcapture/live_camera.pyContinuous webcam feed generatorLowcapture/calibration.pyFace centered / lighting / distance checksLow-Medium (thresholds unvalidated)detection/face_landmarks.pyMediaPipe Face Mesh wrapperLowdetection/beard_segmentation.pyHeuristic HSV-based beard maskHigh — validate before trustingdetection/density_analysis.pyBeard coverage/patchiness heuristicHigh — unvalidated proxy metricstyles/style_catalogue.pyLoad style definitions from JSONLowstyles/recommendation_engine.pyFace-shape + density -> suggested stylesHigh — rule-based face-shape classification is known-unreliable at edgesstyles/custom_style_designer.pyManual mouse-drawn boundary toolLow (no AI involved)zones/zone_generator.pyKeep/blend/remove mask logicMediumtracking/face_alignment_tracker.pyRe-align zones to current frame's face positionMedium — similarity transform on 4 points, sensitive to landmark noiseoverlay/visualizer.py / overlay/live_overlay.pyDraw colored overlay (static / live)Lowguidance/visual_guidance.pyOn-screen text guidanceLowguidance/voice_guidance.pyTTS announcements (pyttsx3)Lowprogress/touchup_tracker.py"Still need touch-up?" decisionMedium — threshold unvalidatedapp/live_session.pyOrchestrates the full V1 loopN/A (wiring only)comparison/before_after.pySide-by-side before/after imageLow

Validation plan before demoing


Collect 10-15 real test photos: varied lighting, beard densities,
skin tones.
Run segment_beard_region on each, visually inspect the mask — do
NOT assume correctness from code running without exceptions.
Tune BEARD_HSV_LOWER / BEARD_HSV_UPPER in config/settings.py.
Run recommend_styles against the same photos and sanity-check the
face-shape labels against what a person would actually call them —
this is the most likely module to look "confidently wrong."
Only after that: tune zone margins, touch-up threshold, and test the
live session (--live) end-to-end with real trimming.


Explicit non-goals for V1 (deferred to V2 or out of scope)


No LED-marker blade tracking (V2).
No predictive "about to over-trim" warning tied to blade motion — V1
can only report current zone state, not forecast a future cut.
No trained segmentation or face-shape model — both are heuristics,
flagged as such in code comments.