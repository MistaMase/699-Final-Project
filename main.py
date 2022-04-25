import logging

import ImageCapture

if __name__ == '__main__':
    # Configure logging
    logging_level = logging.INFO
    logging.basicConfig(level=logging_level, format='%(levelname)s: %(name)s:  %(message)s')
    logger = logging.getLogger(__name__)
    ic = ImageCapture.ImageCapture()
    ic.live_image_overlay()

