import torch
import torchvision.transforms as transforms
from torchvision.models.detection import YoloV8
import cv2
# Load YOLOv8 model
model = YoloV8(pretrained=True)

# Define the object tracking function
def track_objects(video_path, target_class, num_targets):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Prepare output video writer
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    output_video = cv2.VideoWriter('/content/output_tracked.mp4',
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   30, (frame_width, frame_height))

    # Initialize variables
    detected_objects = {}
    tracked_object_id = None
    tracked_object_class = 'car'

    while True:
        ret, frame = cap.read()
        count += 1
        if not ret:
            break

        # Call the detect method and store the returned values in a single variable
        detections = od.detect(frame)

        # Extract the individual values from the detections variable
        class_ids, confidences, boxes = detections
        if len(boxes) == 0:
            print("No objects detected.")
            continue

        # Iterate over the detections and assign object IDs
        for i, box in enumerate(boxes):
            class_id = class_ids[i]
            class_name = od.classes[class_id]
            (x, y, w, h) = box

            # Assign a unique object ID based on the class name
            if class_name not in detected_objects:
                detected_objects[class_name] = len(detected_objects) + 1

            object_id = detected_objects[class_name]

            # Store the object ID along with the box coordinates
            boxes[i] = (x, y, w, h, object_id)

            # Check if the object is the target object
            if class_name == tracked_object_class and object_id == 4:
                tracked_object_id = object_id

            # Draw bounding box and class label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show the frame with the detected object
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break

        # Calculate the reward, next state, and done flag based on your logic
        action = 0  # Replace with your logic
        reward = calculate_reward(action, tracked_object_id, tracking_objects)
        next_state = []  # Replace with your logic
        done = False  # Replace with your logic

        # Store the transition in the replay buffer
        replay_memory.append((state, action, reward, next_state, done))

        # Print the reward for each transition
        print("Reward:", reward)

        output_video.write(frame)

    output_video.release()
    cap.release()
    cv2.destroyAllWindows()

    # Print the summary of detected objects
    print("Detected Objects:")
    for class_name, object_id in detected_objects.items():
        print(f"Object ID: {object_id}, Class: {class_name}")

    # Print the tracked object
    print("\nTracked Object:")
    if tracked_object_id is not None:
        print(f"Tracked Object ID: {tracked_object_id}, Class: {tracked_object_class}")
    else:
        print(f"No object of class {tracked_object_class} found.")
