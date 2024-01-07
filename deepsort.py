from deep_sort import DeepSort

class DeepSort:
    def __init__(self):
        self.deepsort = DeepSort()

    def update(self, frame, bboxes, class_ids):
        # Perform object tracking using DeepSORT
        tracked_targets = self.deepsort.update(frame, bboxes, class_ids)
        return tracked_targets
