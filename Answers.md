# 1. About This Repository

This repository contains answers to a task focused on extracting social interaction cues from egocentric (child-mounted) video footage. It includes:

- A proposal for detecting meaningful child–parent social interactions.
- A sample Python script demonstrating action recognition using a pretrained R3D-18 model from PyTorch.
- A discussion of the limitations of using generic action recognition models (e.g., Kinetics-400) in this context.
- Suggested adjustments to adapt the sample code for better detection of socially meaningful behaviors in videos from child-mounted cameras.

# 2. Proposed approach to extract relevant information regarding social interaction from a video:
To extract social interaction cues from a child-mounted camera, we propose the following:

1. Establish a robust definition of social interaction from developmental psychology to guide our coding, and then proceed to develop the processing pipeline. In past work for instance (e.g., Bohn et al.), interaction has been characterized by mutual engagement—like gaze coordination, vocal turn-taking, shared object attention, or mirroring gestures.

2. Apply separate models to extract simple, observable signals such as: face and body detection to identify who is in the scene, tracking to follow individuals over time, gaze estimation to determine where each person is looking, gesture recognition to capture hand and body movements, and audio processing to detect who is speaking and when.

3. Integrate the detected basic features to identify meaningful social interactions. For instance, if two individuals’ gaze vectors intersect or both point toward the same object, label it as joint attention. If one person performs a gesture (e.g., raising a hand) and shortly afterward another person performs a similar gesture, label it as gesture mirroring. If a parent speaks and the child responds, mark this as vocal turn-taking.
   
4. Discard scenes where no meaningful social behavior occurs, to focus only on episodes with genuine interaction.
   
5. Output a timeline of labeled interactions to quantify metrics such as total interaction time, the frequency of different interaction types, and who interacts most with the child.


# 3.Overview and Key Features of Sample Code

The Python sample code in this repository was used to perform action recognition on short clips extracted from a video file. It leverages a pretrained R3D-18 model from PyTorch’s torchvision library, trained on the Kinetics-400 dataset, to classify actions within overlapping 16-frame segments of the input video. In the following, we list some of the code's key features:

- Loads and preprocesses video frames, converting them to the format required by the R3D-18 model.  
- Implements a sliding window approach to generate overlapping clips for finer temporal resolution in action detection.  
- Uses a pretrained convolutional neural network (R3D-18) for action classification.  
- Outputs a timeline of detected actions with associated confidence scores for each processed clip.  

# 4. Challenges

- Most of the 400 Kinetics-400 classes are not designed to capture two‐person or child‐parent interactions.
- Most of the classes were not relevant in the egocentric (head-cam) view and did not correspond to the actual interactions between the kid and his family members in the video.
- Because the camera is mounted on the child’s head, faces or bodies of other participants are often only partially visible, moving in and out of frame. Kinetics-trained models were calibrated on third‐person views with full bodies and standard angles.
- Gestures like pointing, reaching, or looking at someone (all crucial for social interaction) were misclassified or missed entirely.
- This code doesn't recognize the different individuals in the video, it doesn't recognize who is the parent and who is the kid
- Social interaction often hinges on who is speaking to whom. Here, we ignore the audio track entirely. A lot of “talk” happens offscreen or out of view, so an action model alone will miss it.
- With 16‐frame clips (about 0.5 seconds each) and an 8‐frame stride, we got a coarse, punctuated timeline. Real interactions—like joint attention or back-and-forth speech—happen over seconds, not isolated 0.5 s windows.

# 5. Adjustments

The following includes possible adjustments to the previous code:

- Fine-tune a pre-trained network on an egocentric dataset of child–parent interactions to enable recognition of socially meaningful behaviors beyond those captured in general datasets like Kinetics-400.
- Run a face or body detector on each frame to localize people in the video.
- Track bounding boxes over time to keep people's identities consistent throughout the video as they move and interact.
- Use diarization toolkits to segment who is speaking when, and align each speaker’s utterances with their likely target - for instance, identifying instances of parent-to-child speech.
- Group the 0.5 s clips into 3–5 s sliding windows instead of processing them isolated.
- Use higher confidence thresholds to filter out spurious detections
- Revise predicted social interactions and assigning a generic label (e.g.,“non-social”) to outputs that do not match a curated set of socially relevant behaviors instead of retaining the “unknown” class.
- Incorporate predictions from additional social cue models—such as gaze, head pose, and presence or count of people, and classify an event as a social interaction only when these cues overlap in time and space.
- Produce a structured list that includes, for example, the start and end times of the action, the individuals involved in the scene, and the type of interaction.

