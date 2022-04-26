import cv2
import EmotionClassifier
import logging

class ImageCapture:
    def __init__(self, logging_level=logging.INFO):
        logging.basicConfig(level=logging_level, format='%(levelname)s: %(name)s:  %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.debug('Configured Logging')
        self.fc = EmotionClassifier.EmotionClassifier()
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.text_properties = {
            'font': cv2.FONT_HERSHEY_SIMPLEX,
            'bottomLeftCornerOfText': (10, 50),
            'fontScale': 1,
            'fontColor': (255, 255, 255),
            'thickness': 1,
            'lineType': cv2.LINE_4
        }
        self.logger.debug('Finished Initialization')

    def live_image_overlay(self):
        self.logger.debug('New Frame')
        # Get the camera feed
        ret, raw_frame = self.video_capture.read()

        # Flip to be more natural
        raw_frame = cv2.flip(raw_frame, 1)

        # Convert the image to grayscale
        gray_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

        active_region_top_left = (455, 100)
        active_region_bottom_right = (805, 450)
        active_area_frame = gray_frame[active_region_top_left[1]:active_region_bottom_right[1], active_region_top_left[0]:active_region_bottom_right[0]]

        # Draw the rectangle used for facial classification
        gray_frame = cv2.rectangle(
            gray_frame,
            (active_region_top_left[0]-1, active_region_top_left[1]-1),             # Start Point
            (active_region_bottom_right[0]+1, active_region_bottom_right[1]+1),     # End Point
            (255, 255, 255),                                                        # Color
            1                                                                       # Thickness
        )

        # Run the classifier
        emotions = self.fc.get_emotion(active_area_frame)
        self.logger.debug('Emotion List: %s', emotions)
        perceived_emotion = max(emotions, key=emotions.get)
        self.logger.info('Perceived Emotion: %s', perceived_emotion)

        # Write the result from image classification to the active area image
        cv2.putText(active_area_frame,
                    'Emotion: ' + perceived_emotion,
                    self.text_properties['bottomLeftCornerOfText'],
                    self.text_properties['font'],
                    self.text_properties['fontScale'],
                    self.text_properties['fontColor'],
                    self.text_properties['thickness'],
                    self.text_properties['lineType'])

        # Display the frames
        cv2.imshow('Webcam', gray_frame)
        cv2.imshow('Active Area', active_area_frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.logger.debug('Q Key Pressed')
            raise Exception

    def __del__(self):
        # Free the image capture
        self.video_capture.release()